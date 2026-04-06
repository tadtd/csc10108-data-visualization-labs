import time
import random
import logging
import os

# Configure logging
LOG_FILE = "logs/member_2.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def crawl_listing(category):
    """
    Mock function to crawl product listings for a category.
    """
    retries = 3
    while retries > 0:
        try:
            logging.info(f"Crawling listing for category: {category}")
            time.sleep(random.uniform(1, 3))
            
            print(f"Successfully crawled listing for {category}")
            return True
        except Exception as e:
            retries -= 1
            logging.error(f"Error crawling {category}: {e}. Retries left: {retries}")
            time.sleep(2)
    return False

if __name__ == "__main__":
    import json
    with open('config/categories_member_2.json', 'r') as f:
        categories = json.load(f)
    
    for cat in categories:
        crawl_listing(cat)
