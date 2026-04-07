import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from tiki_client import (
    build_product_record,
    build_review_record,
    fetch_product_detail,
    fetch_reviews,
    search_products,
)

LOG_FILE = "logs/member_1.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

PRODUCT_COLUMNS = [
    "product_id",
    "title",
    "price",
    "original_price",
    "discount_percent",
    "sold_count",
    "ranking",
    "rating",
    "review_count",
    "author",
    "publisher",
    "category",
    "publish_year",
    "page_count",
    "description",
    "is_bestseller",
    "is_top100",
    "is_combo",
    "has_gift",
]

REVIEW_COLUMNS = ["review_id", "product_id", "rating_score", "content", "created_at"]


def load_categories(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        return [str(item) for item in data]
    return []


def build_rankings(product_ids: List[int]) -> Dict[int, int]:
    return {product_id: rank for rank, product_id in enumerate(product_ids, start=1)}


def crawl_category(
    category: str,
    max_items: int,
    per_page: int,
    max_pages: int,
    sleep_range: Tuple[float, float],
) -> List[int]:
    logging.info("Crawling listing for category: %s", category)
    query = category
    return search_products(query, max_items, per_page, max_pages, sleep_range)


def build_authors_table(products: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(products)
    if df.empty:
        return pd.DataFrame(columns=["author_name", "total_books", "total_sales", "average_rating"])
    df["author"] = df["author"].fillna("")
    df = df[df["author"].str.strip() != ""]
    if df.empty:
        return pd.DataFrame(columns=["author_name", "total_books", "total_sales", "average_rating"])

    summary = (
        df.groupby("author", dropna=True)
        .agg(
            total_books=("product_id", "count"),
            total_sales=("sold_count", "sum"),
            average_rating=("rating", "mean"),
        )
        .reset_index()
        .rename(columns={"author": "author_name"})
    )
    summary["average_rating"] = summary["average_rating"].fillna(0)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl Tiki books data for member_1.")
    default_config = Path(__file__).resolve().parents[3] / "config" / "categories_member_1.json"
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to category config.",
    )
    default_out_dir = Path(__file__).resolve().parents[3] / "data" / "raw" / "member_1"
    parser.add_argument(
        "--out-dir",
        default=str(default_out_dir),
        help="Output directory for CSV files.",
    )
    parser.add_argument("--max-per-category", type=int, default=1000, help="Max products per category.")
    parser.add_argument("--listing-limit", type=int, default=40, help="Listing page size.")
    parser.add_argument("--max-pages", type=int, default=25, help="Max listing pages per category.")
    parser.add_argument("--max-reviews", type=int, default=50, help="Max reviews per product.")
    parser.add_argument("--review-limit", type=int, default=20, help="Review page size.")
    parser.add_argument("--sleep-min", type=float, default=1.0, help="Min sleep between requests.")
    parser.add_argument("--sleep-max", type=float, default=2.5, help="Max sleep between requests.")
    args = parser.parse_args()

    sleep_range = (args.sleep_min, args.sleep_max)
    os.makedirs(args.out_dir, exist_ok=True)

    categories = load_categories(args.config)
    if not categories:
        logging.error("No categories found in %s", args.config)
        return

    all_products: List[Dict[str, object]] = []
    all_reviews: List[Dict[str, object]] = []

    for category in categories:
        logging.info("\n=== Category: %s ===", category)
        product_ids = crawl_category(
            category,
            args.max_per_category,
            args.listing_limit,
            args.max_pages,
            sleep_range,
        )
        if not product_ids:
            logging.warning("No products found for category: %s", category)
            continue
        logging.info("Found %s products for category: %s", len(product_ids), category)

        rankings = build_rankings(product_ids)

        for index, product_id in enumerate(product_ids, start=1):
            try:
                logging.info("[%s/%s] Fetching product_id=%s", index, len(product_ids), product_id)
                detail = fetch_product_detail(product_id, sleep_range)
                if not detail:
                    logging.warning("Empty detail for product_id: %s", product_id)
                    continue
                product = build_product_record(detail, rankings[product_id], category)
                if product.get("product_id"):
                    all_products.append(product)

                reviews = fetch_reviews(product_id, args.max_reviews, args.review_limit, sleep_range)
                logging.info("Fetched %s reviews for product_id=%s", len(reviews), product_id)
                for review in reviews:
                    all_reviews.append(build_review_record(review, product_id))
            except Exception as exc:  # noqa: BLE001
                logging.error("Failed product_id %s: %s", product_id, exc)

    products_df = pd.DataFrame(all_products, columns=PRODUCT_COLUMNS)
    reviews_df = pd.DataFrame(all_reviews, columns=REVIEW_COLUMNS)
    authors_df = build_authors_table(all_products)

    products_df.to_csv(os.path.join(args.out_dir, "products.csv"), index=False)
    reviews_df.to_csv(os.path.join(args.out_dir, "reviews.csv"), index=False)
    authors_df.to_csv(os.path.join(args.out_dir, "authors.csv"), index=False)

    logging.info(
        "Saved %s products, %s reviews, %s authors",
        len(products_df),
        len(reviews_df),
        len(authors_df),
    )


if __name__ == "__main__":
    random.seed(42)
    main()
