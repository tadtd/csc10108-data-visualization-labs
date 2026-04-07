import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests


def resolve_root_dir() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents]:
        if (candidate / "config" / "categories_member_2.json").exists():
            return candidate
    return script_path.parents[3]


ROOT_DIR = resolve_root_dir()
RAW_DIR = ROOT_DIR / "data" / "raw" / "member_2"
LOG_FILE = ROOT_DIR / "logs" / "member_2.log"

PRODUCTS_FILE = RAW_DIR / "products.csv"
REVIEWS_FILE = RAW_DIR / "reviews.csv"

TIKI_REVIEW_API = "https://tiki.vn/api/v2/reviews"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
SLEEP_RANGE = (1, 3)
SAFE_MAX_WORKERS = 8

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def build_headers() -> dict:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://tiki.vn/",
    }


def request_json(url: str, params: dict) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(random.uniform(*SLEEP_RANGE))
            response = requests.get(
                url,
                params=params,
                headers=build_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logging.error("Review request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise


def extract_review_rows(product_id: int, items: list) -> list[dict]:
    rows = []
    for item in items:
        review_id = item.get("id")
        if not review_id:
            continue
        rows.append(
            {
                "review_id": int(review_id),
                "product_id": int(product_id),
                "rating_score": float(item.get("rating") or 0),
                "content": str(item.get("content", "")),
                "created_at": str(item.get("created_at", "")),
            }
        )
    return rows


def crawl_reviews_for_product(product_id: int, max_pages: int, per_page: int) -> list[dict]:
    rows = []
    for page in range(1, max_pages + 1):
        params = {
            "product_id": product_id,
            "limit": per_page,
            "page": page,
            "include": "comments",
            "sort": "score|desc,id|desc",
            "spid": 0,
        }
        try:
            payload = request_json(TIKI_REVIEW_API, params)
        except Exception:
            logging.exception("Failed review crawl for product_id=%s page=%s", product_id, page)
            break

        data = payload.get("data", [])
        if not data:
            break

        rows.extend(extract_review_rows(product_id, data))
    return rows


def crawl_review(
    limit_products: int | None = None,
    max_pages: int = 2,
    per_page: int = 20,
    workers: int | None = None,
) -> pd.DataFrame:
    if not PRODUCTS_FILE.exists():
        raise FileNotFoundError(f"Missing products file: {PRODUCTS_FILE}. Run crawl_detail.py first.")

    products_df = pd.read_csv(PRODUCTS_FILE)
    if "product_id" not in products_df.columns:
        raise ValueError("products_member_2.csv must contain product_id column")

    product_ids = products_df["product_id"].dropna().astype(int).tolist()
    if limit_products is not None and limit_products > 0:
        product_ids = product_ids[:limit_products]

    all_rows = []
    max_workers = workers or min(SAFE_MAX_WORKERS, (os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [
            executor.submit(crawl_reviews_for_product, product_id, max_pages, per_page)
            for product_id in product_ids
        ]
        for future in as_completed(futures):
            all_rows.extend(future.result())

    reviews_df = pd.DataFrame(
        all_rows,
        columns=["review_id", "product_id", "rating_score", "content", "created_at"],
    )
    if not reviews_df.empty:
        reviews_df = reviews_df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)

    reviews_df.to_csv(REVIEWS_FILE, index=False)
    logging.info("Saved reviews=%s rows to %s", len(reviews_df), REVIEWS_FILE)
    print(f"Saved reviews: {REVIEWS_FILE} | rows={len(reviews_df)}")
    return reviews_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl reviews from product IDs in products CSV.")
    parser.add_argument("--limit-products", type=int, default=None, help="Optional cap on number of products.")
    parser.add_argument("--max-pages", type=int, default=2, help="Number of review pages per product.")
    parser.add_argument("--per-page", type=int, default=20, help="Reviews per page.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(SAFE_MAX_WORKERS, (os.cpu_count() or 1)),
        help="Number of parallel workers for review crawling.",
    )
    args = parser.parse_args()
    crawl_review(
        limit_products=args.limit_products,
        max_pages=args.max_pages,
        per_page=args.per_page,
        workers=args.workers,
    )
