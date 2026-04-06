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

def crawl_review(product_id):
    """
    Mock function to crawl product reviews.
    """
    retries = 3
    while retries > 0:
        try:
            logging.info(f"Crawling reviews for product_id: {product_id}")
            time.sleep(random.uniform(1, 3))
            
            print(f"Successfully crawled reviews for product {product_id}")
            return True
        except Exception as e:
            retries -= 1
            logging.error(f"Error crawling reviews for {product_id}: {e}. Retries left: {retries}")
            time.sleep(2)
    return False

if __name__ == "__main__":
    crawl_review("67890")
