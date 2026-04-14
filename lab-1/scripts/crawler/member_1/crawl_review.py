import argparse
import logging
import os
import random

from tiki_client import build_review_record, fetch_reviews

LOG_FILE = "logs/member_1.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def crawl_review(product_id: int, max_reviews: int, per_page: int) -> None:
    logging.info("Crawling reviews for product_id: %s", product_id)
    reviews = fetch_reviews(product_id, max_reviews, per_page, (1.0, 2.5))
    for review in reviews:
        print(build_review_record(review, product_id))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch reviews for a product.")
    parser.add_argument("product_id", type=int, help="Tiki product id")
    parser.add_argument("--max-reviews", type=int, default=10, help="Max reviews to fetch")
    parser.add_argument("--per-page", type=int, default=5, help="Reviews per page")
    args = parser.parse_args()
    crawl_review(args.product_id, args.max_reviews, args.per_page)


if __name__ == "__main__":
    random.seed(42)
    main()
