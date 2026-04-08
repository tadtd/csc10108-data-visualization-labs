import argparse
import logging
import os
import random

from tiki_client import build_product_record, fetch_product_detail

LOG_FILE = "logs/member_1.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def crawl_detail(product_id: int, category: str) -> None:
    logging.info("Crawling details for product_id: %s", product_id)
    detail = fetch_product_detail(product_id, (1.0, 2.5))
    if not detail:
        logging.warning("No detail for product_id: %s", product_id)
        return
    product = build_product_record(detail, ranking=0, category_name=category)
    print(product)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a single product detail.")
    parser.add_argument("product_id", type=int, help="Tiki product id")
    parser.add_argument("--category", default="", help="Category name fallback")
    args = parser.parse_args()
    crawl_detail(args.product_id, args.category)


if __name__ == "__main__":
    random.seed(42)
    main()
