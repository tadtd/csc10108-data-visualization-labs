import time
import random
import logging
import os

# Configure logging
LOG_FILE = "logs/member_1.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def crawl_detail(product_id):
    """
    Mock function to crawl product details.
    """
    retries = 3
    while retries > 0:
        try:
            logging.info(f"Crawling details for product_id: {product_id}")
            time.sleep(random.uniform(1, 3))
            
            print(f"Successfully crawled details for product {product_id}")
            return True
        except Exception as e:
            retries -= 1
            logging.error(f"Error crawling details for {product_id}: {e}. Retries left: {retries}")
            time.sleep(2)
    return False

if __name__ == "__main__":
    # Mock data
    crawl_detail("12345")
