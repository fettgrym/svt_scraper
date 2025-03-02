import requests
from bs4 import BeautifulSoup
import sqlite3
import re
import time
import random
import schedule
import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service
from contextlib import contextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("news_scraper")

# Load configuration
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default configuration if file not found
        return {
            "base_url": "https://www.svt.se/",
            "db_name": "news_articles.db",
            "keyword_limit": 50,
            "max_pages": 5,
            "driver_path": "chromedriver.exe",
            "scraping_interval_hours": 12,
            "retry_attempts": 3,
            "retry_delay": 5,
            "request_delay_min": 1,
            "request_delay_max": 3
        }

CONFIG = load_config()
BASE_URL = CONFIG["base_url"]
DB_NAME = CONFIG["db_name"]
KEYWORD_LIMIT = CONFIG["keyword_limit"]
MAX_PAGES = CONFIG["max_pages"]
DRIVER_PATH = CONFIG["driver_path"]
MAX_RETRIES = CONFIG["retry_attempts"]
RETRY_DELAY = CONFIG["retry_delay"]
REQUEST_DELAY_MIN = CONFIG["request_delay_min"]
REQUEST_DELAY_MAX = CONFIG["request_delay_max"]

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# Initialize NLP tools
STOP_WORDS = set(stopwords.words('swedish'))
lemmatizer = WordNetLemmatizer()
sentiment_analyzer = SentimentIntensityAnalyzer()

# Database connection context manager
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def perform_database_migrations():
    """Handle database schema updates"""
    with get_db_connection() as conn:
        c = conn.cursor()
        try:
            # Check if the sentiment_score column exists
            c.execute("PRAGMA table_info(articles)")
            columns = [info[1] for info in c.fetchall()]
            
            # Add missing columns
            if 'sentiment_score' not in columns:
                logger.info("Performing migration: Adding sentiment_score column")
                c.execute("ALTER TABLE articles ADD COLUMN sentiment_score REAL DEFAULT 0.0")
                
                # Update existing records with sentiment scores
                c.execute("SELECT id, title, content FROM articles WHERE sentiment_score IS NULL")
                articles = c.fetchall()
                
                for article in articles:
                    sentiment = analyze_sentiment(f"{article['title']} {article['content']}")
                    c.execute("UPDATE articles SET sentiment_score = ? WHERE id = ?", 
                             (sentiment, article['id']))
                    
                logger.info(f"Updated sentiment scores for {len(articles)} existing articles")
                
            conn.commit()
            logger.info("Database migrations completed successfully")
            
        except sqlite3.Error as e:
            logger.error(f"Database migration error: {str(e)}")
            conn.rollback()

def setup_database():
    """Initialize the database schema with proper indexes"""
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Articles table
        c.execute('''CREATE TABLE IF NOT EXISTS articles
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      url TEXT UNIQUE,
                      title TEXT,
                      content TEXT,
                      published_date DATETIME,
                      scraped_date DATETIME,
                      sentiment_score REAL)''')
        
        # Keywords table
        c.execute('''CREATE TABLE IF NOT EXISTS keywords
                     (article_id INTEGER,
                      keyword TEXT,
                      count INTEGER,
                      FOREIGN KEY(article_id) REFERENCES articles(id))''')
        
        conn.commit()
        
        # Run migrations to update the schema if needed
        perform_database_migrations()
        
        # Create indexes for better performance
        c.execute('''CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_keywords_article ON keywords(article_id)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_articles_date ON articles(published_date)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_articles_scraped ON articles(scraped_date)''')
        
        conn.commit()
        logger.info("Database setup complete")

def is_valid_article_url(url):
    """Determine if a URL is a valid news article"""
    if not url or not url.startswith(BASE_URL):
        return False
    
    # Must contain news path
    if '/nyheter/' not in url:
        return False
    
    # Exclude unwanted sections
    excluded = ['/video/', '/direkt/', '/sport/', '/play/', '/barn/']
    for exclude in excluded:
        if exclude in url:
            return False
            
    return True

def rate_limit():
    """Add random delay between requests to avoid being blocked"""
    delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
    time.sleep(delay)

def init_webdriver(headless=True):
    """Initialize and return a configured Chrome WebDriver"""
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-notifications')
    options.add_argument('--disable-popup-blocking')
    options.add_argument('--disable-infobars')
    options.add_argument('--start-maximized')
    
    if headless:
        options.add_argument('--headless')
    
    service = Service(DRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def handle_cookie_consent(driver):
    """Handle cookie consent popup if present"""
    try:
        # Wait for cookie button to be present and clickable
        cookie_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '/html/body/div[1]/div[2]/div/div[3]/button[2]'))
        )
        cookie_button.click()
        logger.info("Cookie consent handled")
        return True
    except Exception as e:
        logger.warning(f"Failed to handle cookie consent: {str(e)}")
        return False

def get_article_urls():
    """Scrape article URLs with pagination using Selenium"""
    logger.info("Starting URL collection...")
    driver = init_webdriver(headless=False)  # Set to True for production
    driver.get(BASE_URL)
    
    # Handle cookie consent popup
    handle_cookie_consent(driver)
    
    urls = set()
    page_count = 0

    try:
        while page_count < MAX_PAGES:
            try:
                # Wait for articles to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//main//ul/li//a'))
                )
                
                # Extract current page URLs
                articles = driver.find_elements(By.XPATH, '//main//ul/li//a')
                new_urls_count = 0
                
                for article in articles:
                    url = article.get_attribute('href')
                    if is_valid_article_url(url) and url not in urls:
                        urls.add(url)
                        new_urls_count += 1
                
                logger.info(f"Found {new_urls_count} new URLs (total: {len(urls)})")
                
                # Try to find and click pagination or "load more" button
                try:
                    # First try explicit pagination
                    pagination = driver.find_elements(By.CSS_SELECTOR, ".pagination a, .pagination button")
                    if pagination and len(pagination) > page_count:
                        driver.execute_script("arguments[0].click();", pagination[page_count])
                        logger.info(f"Clicked pagination element {page_count+1}")
                    else:
                        # Try generic "load more" button
                        load_more_xpath = "//button[contains(., 'Visa fler')]"  # "Show more" in Swedish
                        load_more_button = driver.find_element(By.XPATH, load_more_xpath)
                        driver.execute_script("arguments[0].click();", load_more_button)
                        logger.info("Clicked 'load more' button")
                        
                    # Allow time for new content to load
                    time.sleep(2)
                    page_count += 1
                    
                    # If no new URLs were found, we might be at the end
                    if new_urls_count == 0:
                        logger.info("No new URLs found, breaking pagination loop")
                        break
                        
                except NoSuchElementException:
                    logger.info("No more pages to load")
                    break

            except TimeoutException:
                logger.warning("Timed out waiting for page to load")
                break

    except Exception as e:
        logger.error(f"Error collecting URLs: {str(e)}")
    finally:
        driver.quit()

    logger.info(f"Total unique URLs collected: {len(urls)}")
    return list(urls)

def scrape_article(url, driver):
    """Scrape individual article content using Selenium"""
    logger.info(f"Scraping article: {url}")
    
    try:
        driver.get(url)
        
        # Handle cookie consent if needed
        handle_cookie_consent(driver)
        
        # Wait for the article content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "h1"))
        )

        # Extract title
        title = driver.find_element(By.TAG_NAME, "h1").text
        logger.info(f"Found title: {title}")

        # Extract content
        try:
            # Try to get the main article container
            content_div = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//main//article'))
            )
            paragraphs = content_div.find_elements(By.TAG_NAME, "p")
            content = ' '.join([p.text for p in paragraphs if p.text.strip()])
        except NoSuchElementException:
            # Fallback to finding all paragraphs within main
            paragraphs = driver.find_elements(By.XPATH, '//main//p')
            content = ' '.join([p.text for p in paragraphs if p.text.strip()])
        
        if not content:
            logger.warning(f"No content found for article: {url}")
            return None

        # Extract published date
        try:
            date_element = driver.find_element(By.TAG_NAME, "time")
            published_date = datetime.fromisoformat(date_element.get_attribute('datetime'))
        except NoSuchElementException:
            # If no date element is found, use current time
            logger.warning(f"No date element found for article: {url}")
            published_date = datetime.now()
            
        # Calculate sentiment score
        sentiment_score = analyze_sentiment(f"{title} {content}")
            
        return {
            'title': title,
            'content': content,
            'published_date': published_date,
            'url': url,
            'sentiment_score': sentiment_score
        }

    except TimeoutException:
        logger.error(f"Timeout while scraping {url}")
        return None
    except NoSuchElementException as e:
        logger.error(f"Element not found while scraping {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return None

def scrape_with_retry(url, driver, max_retries=MAX_RETRIES, delay=RETRY_DELAY):
    """Attempt to scrape an article with retries on failure"""
    for attempt in range(max_retries):
        try:
            article = scrape_article(url, driver)
            if article:
                return article
            else:
                logger.warning(f"Attempt {attempt+1} returned None for {url}")
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {url}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    logger.error(f"All {max_retries} attempts failed for {url}")
    return None

def process_keywords(text):
    """Extract and count keywords from text with lemmatization"""
    if not text:
        return {}
        
    # Tokenize text
    words = word_tokenize(text.lower())
    
    # Filter and lemmatize words
    filtered_words = [lemmatizer.lemmatize(word) for word in words 
                     if word.isalnum() and word not in STOP_WORDS and len(word) > 2]
    
    # Count occurrences
    word_counts = defaultdict(int)
    for word in filtered_words:
        word_counts[word] += 1
        
    return dict(word_counts)

def analyze_sentiment(text):
    """Analyze sentiment of text and return compound score"""
    if not text:
        return 0.0
        
    sentiment = sentiment_analyzer.polarity_scores(text)
    return sentiment['compound']  # Return the compound score between -1 and 1

def store_article(article):
    """Store article and keywords in database with transaction handling"""
    if not article or not article.get('content'):
        logger.warning("Cannot store article: missing data")
        return False

    with get_db_connection() as conn:
        try:
            c = conn.cursor()
            
            # Insert article
            c.execute('''INSERT OR IGNORE INTO articles 
                         (url, title, content, published_date, scraped_date, sentiment_score)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                      (article['url'], article['title'], article['content'],
                       article['published_date'], datetime.now(), article['sentiment_score']))
            
            if c.rowcount > 0:
                article_id = c.lastrowid
                # Process and store keywords
                keywords = process_keywords(f"{article['title']} {article['content']}")
                for word, count in keywords.items():
                    c.execute('''INSERT INTO keywords (article_id, keyword, count)
                                 VALUES (?, ?, ?)''', (article_id, word, count))
                logger.info(f"Stored article: {article['title']} (ID: {article_id})")
                conn.commit()
                return True
            else:
                logger.info(f"Duplicate article skipped: {article['title']}")
                return False
                
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            conn.rollback()
            return False

def get_existing_urls():
    """Get list of URLs already in the database"""
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute('SELECT url FROM articles')
        return {row['url'] for row in c.fetchall()}

def scraping_job():
    """Main scraping job that collects and processes articles"""
    logger.info("--- Starting scraping job ---")
    start_time = time.time()
    
    try:
        # Get existing URLs to avoid re-scraping
        existing_urls = get_existing_urls()
        logger.info(f"Found {len(existing_urls)} existing articles in database")
        
        # Get new URLs
        all_urls = get_article_urls()
        new_urls = [url for url in all_urls if url not in existing_urls]
        logger.info(f"Found {len(new_urls)} new articles to scrape")
        
        if not new_urls:
            logger.info("No new articles to scrape, job completed")
            return
            
        # Initialize Chrome driver
        driver = init_webdriver()
        
        try:
            # Process each URL
            articles_processed = 0
            articles_stored = 0
            
            for url in new_urls:
                # Add rate limiting
                rate_limit()
                
                # Scrape with retry
                article = scrape_with_retry(url, driver)
                articles_processed += 1
                
                # Store if successful
                if article:
                    if store_article(article):
                        articles_stored += 1
                
                # Log progress
                if articles_processed % 10 == 0:
                    logger.info(f"Progress: {articles_processed}/{len(new_urls)} articles processed")
                    
        finally:
            driver.quit()
            
        # Final stats
        elapsed_time = time.time() - start_time
        logger.info(f"Scraping job completed in {elapsed_time:.2f} seconds")
        logger.info(f"Articles processed: {articles_processed}, successfully stored: {articles_stored}")
            
    except Exception as e:
        logger.error(f"Scraping job failed: {str(e)}")

def generate_report(time_period):
    """Generate keyword and sentiment report for specified time period"""
    logger.info(f"Generating {time_period} report")
    
    periods = {
        'daily': 1,
        'weekly': 7,
        'monthly': 30,
        'all_time': 0
    }
    
    if time_period not in periods:
        logger.error(f"Invalid time period: {time_period}")
        return
    
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Keywords query
        keyword_query = '''
            SELECT keyword, SUM(count) as total
            FROM keywords
            JOIN articles ON articles.id = keywords.article_id
            WHERE scraped_date >= ?
            GROUP BY keyword
            ORDER BY total DESC
            LIMIT ?
        '''
        
        # Sentiment query
        sentiment_query = '''
            SELECT AVG(sentiment_score) as avg_sentiment,
                   COUNT(*) as article_count
            FROM articles
            WHERE scraped_date >= ?
        '''
        
        # Execute queries
        if time_period == 'all_time':
            c.execute('SELECT keyword, SUM(count) as total FROM keywords GROUP BY keyword ORDER BY total DESC LIMIT ?', 
                     (KEYWORD_LIMIT,))
            keyword_results = c.fetchall()
            
            c.execute('SELECT AVG(sentiment_score) as avg_sentiment, COUNT(*) as article_count FROM articles')
            sentiment_results = c.fetchone()
        else:
            delta = datetime.now() - timedelta(days=periods[time_period])
            
            c.execute(keyword_query, (delta, KEYWORD_LIMIT))
            keyword_results = c.fetchall()
            
            c.execute(sentiment_query, (delta,))
            sentiment_results = c.fetchone()
        
        # Generate report file
        filename = f"{time_period}_report_{datetime.now().strftime('%Y-%m-%d')}.txt"
        with open(filename, 'w') as f:
            # Header
            f.write(f"{time_period.capitalize()} News Report - {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("="*50 + "\n\n")
            
            # Sentiment section
            f.write("SENTIMENT ANALYSIS\n")
            f.write("-"*20 + "\n")
            f.write(f"Average sentiment score: {sentiment_results['avg_sentiment']:.4f}\n")
            f.write(f"Articles analyzed: {sentiment_results['article_count']}\n\n")
            sentiment_label = "Positive" if sentiment_results['avg_sentiment'] > 0 else "Negative" if sentiment_results['avg_sentiment'] < 0 else "Neutral"
            f.write(f"Overall sentiment: {sentiment_label}\n\n")
            
            # Keywords section
            f.write("TOP KEYWORDS\n")
            f.write("-"*20 + "\n")
            for idx, (keyword, count) in enumerate(keyword_results, 1):
                f.write(f"{idx}. {keyword}: {count}\n")
        
        logger.info(f"Generated report: {filename}")
        
        # Return most common words for possible visualization
        return keyword_results

def generate_trend_data():
    """Generate trend data for visualization"""
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Get sentiment trend by day
        c.execute('''
            SELECT DATE(published_date) as date, AVG(sentiment_score) as avg_sentiment
            FROM articles
            WHERE published_date > date('now', '-30 days')
            GROUP BY DATE(published_date)
            ORDER BY date
        ''')
        sentiment_trend = c.fetchall()
        
        # Get top 5 keywords for past month
        c.execute('''
            SELECT keyword, SUM(count) as total
            FROM keywords
            JOIN articles ON articles.id = keywords.article_id
            WHERE published_date > date('now', '-30 days')
            GROUP BY keyword
            ORDER BY total DESC
            LIMIT 5
        ''')
        top_keywords = [row['keyword'] for row in c.fetchall()]
        
        # Get trend for top keywords
        keyword_trends = {}
        for keyword in top_keywords:
            c.execute('''
                SELECT DATE(articles.published_date) as date, SUM(keywords.count) as count
                FROM keywords
                JOIN articles ON articles.id = keywords.article_id
                WHERE keyword = ? AND published_date > date('now', '-30 days')
                GROUP BY DATE(articles.published_date)
                ORDER BY date
            ''', (keyword,))
            keyword_trends[keyword] = c.fetchall()
        
        # Export to JSON for visualization
        trends = {
            'sentiment': [{'date': row['date'], 'value': row['avg_sentiment']} for row in sentiment_trend],
            'keywords': {k: [{'date': row['date'], 'count': row['count']} for row in v] for k, v in keyword_trends.items()}
        }
        
        with open('trends.json', 'w') as f:
            json.dump(trends, f, indent=2)
            
        logger.info("Generated trends data for visualization")

def schedule_jobs():
    """Schedule regular jobs"""
    scraping_interval = CONFIG.get("scraping_interval_hours", 12)
    logger.info(f"Scheduling scraping job every {scraping_interval} hours")
    
    # Schedule scraping
    schedule.every(scraping_interval).hours.do(scraping_job)
    
    # Schedule reports
    schedule.every().day.at("00:00").do(generate_report, 'daily')
    schedule.every().monday.at("00:00").do(generate_report, 'weekly')
    
    # FIX: Correct syntax for monthly scheduling
    schedule.every().month_at("00:00").do(generate_report, 'monthly')
    
    # Schedule trend data generation
    schedule.every().day.at("01:00").do(generate_trend_data)
    
    logger.info("All jobs scheduled")
    
    # Run continuously
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in scheduled job: {str(e)}")
            time.sleep(60)  # Wait a minute before resuming

def main():
    """Main entry point"""
    setup_database()
    
    # Parse command line arguments if any
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "report":
            # Generate a specific report
            period = sys.argv[2] if len(sys.argv) > 2 else "daily"
            generate_report(period)
        elif sys.argv[1] == "trends":
            # Generate trend data
            generate_trend_data()
        else:
            # Run a scraping job
            scraping_job()
    else:
        # Default: run a scraping job and then start scheduler
        logger.info("Running initial scraping job...")
        scraping_job()
        
        logger.info("Starting scheduler...")
        schedule_jobs()

if __name__ == "__main__":
    main()