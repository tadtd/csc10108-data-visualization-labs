import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import TooManyRedirects


def resolve_root_dir() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents]:
        if (candidate / "config" / "categories_member_2.json").exists():
            return candidate
    return script_path.parents[3]


ROOT_DIR = resolve_root_dir()
RAW_DIR = ROOT_DIR / "data" / "raw" / "member_2"
LOG_FILE = ROOT_DIR / "logs" / "member_2.log"

LISTING_FILE = RAW_DIR / "listing_member_2.csv"
PRODUCTS_FILE = RAW_DIR / "products_member_2.csv"
AUTHORS_FILE = RAW_DIR / "authors_member_2.csv"

TIKI_DETAIL_API = "https://tiki.vn/api/v2/products/{product_id}"
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


def request_json(url: str, params: dict | None = None) -> dict:
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
            logging.error("Detail request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            error_text = str(exc).lower()
            # Back off harder for anti-bot patterns to reduce redirect loops and 429 bursts.
            if isinstance(exc, TooManyRedirects) or "too many requests" in error_text:
                time.sleep(2 * attempt + random.uniform(1, 2))
            if attempt == MAX_RETRIES:
                raise


def parse_sold_count(payload: dict) -> int:
    quantity_sold = payload.get("quantity_sold")
    if isinstance(quantity_sold, dict):
        text = str(quantity_sold.get("text", ""))
    else:
        text = str(quantity_sold or "")

    lowered = text.lower().replace("đã bán", "").replace("+", "").strip()
    if not lowered:
        return 0

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([kKmM]?)", lowered)
    if not match:
        return 0

    value = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        value *= 1000
    elif suffix == "m":
        value *= 1_000_000
    return int(value)


def parse_spec_value(specifications: list, key_candidates: list[str]) -> str:
    key_candidates = [x.lower() for x in key_candidates]
    for spec in specifications or []:
        for attr in spec.get("attributes", []):
            attr_name = str(attr.get("name", "")).strip().lower()
            if attr_name in key_candidates:
                return str(attr.get("value", "")).strip()
    return ""


def as_int(value: str) -> int | None:
    if value is None:
        return None
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def parse_badges(payload: dict) -> tuple[bool, bool]:
    badge_texts = []
    for badge in payload.get("badges_new", []) or []:
        badge_texts.append(str(badge.get("code", "")))
        badge_texts.append(str(badge.get("text", "")))
    all_text = " ".join(badge_texts).lower()
    return ("best" in all_text), ("top" in all_text and "100" in all_text)


def build_product_row(payload: dict, fallback_category: str, fallback_ranking: int) -> dict:
    specifications = payload.get("specifications", []) or []
    breadcrumbs = payload.get("breadcrumbs", []) or []
    categories = [item.get("name") for item in breadcrumbs if item.get("name")]

    author_name = ""
    authors = payload.get("authors")
    if isinstance(authors, list) and authors:
        author_name = str(authors[0].get("name", "")).strip()

    publisher = parse_spec_value(specifications, ["nhà xuất bản", "publisher"])
    publish_year = as_int(parse_spec_value(specifications, ["năm xuất bản", "publish year", "publication year"]))
    page_count = as_int(parse_spec_value(specifications, ["số trang", "number of pages", "page count"]))

    is_bestseller, is_top100 = parse_badges(payload)
    title = str(payload.get("name", ""))

    description = str(payload.get("short_description", "") or "")
    long_desc = str(payload.get("description", "") or "")
    if long_desc and long_desc not in description:
        description = f"{description} {long_desc}".strip()

    gifts = payload.get("gifts", [])
    has_gift = bool(gifts)

    return {
        "product_id": int(payload.get("id", 0)),
        "title": title,
        "price": float(payload.get("price") or 0),
        "original_price": float(payload.get("original_price") or payload.get("price") or 0),
        "discount_percent": int(payload.get("discount_rate") or 0),
        "sold_count": parse_sold_count(payload),
        "ranking": int(fallback_ranking or 0),
        "rating": float(payload.get("rating_average") or 0),
        "review_count": int(payload.get("review_count") or 0),
        "author": author_name,
        "publisher": publisher,
        "category": categories[-1] if categories else fallback_category,
        "publish_year": publish_year,
        "page_count": page_count,
        "description": description,
        "is_bestseller": bool(is_bestseller),
        "is_top100": bool(is_top100),
        "is_combo": ("combo" in title.lower()),
        "has_gift": has_gift,
    }


def build_authors_table(products_df: pd.DataFrame) -> pd.DataFrame:
    if products_df.empty or "author" not in products_df.columns:
        return pd.DataFrame(columns=["author_name", "total_books", "total_sales", "average_rating"])

    author_df = products_df.copy()
    author_df["author"] = author_df["author"].fillna("").astype(str).str.strip()
    author_df = author_df[author_df["author"] != ""]
    if author_df.empty:
        return pd.DataFrame(columns=["author_name", "total_books", "total_sales", "average_rating"])

    grouped = (
        author_df.groupby("author", as_index=False)
        .agg(
            total_books=("product_id", "nunique"),
            total_sales=("sold_count", "sum"),
            average_rating=("rating", "mean"),
        )
        .rename(columns={"author": "author_name"})
    )
    return grouped


def fetch_single_detail(input_row: dict) -> dict | None:
    product_id = int(input_row["product_id"])
    category = str(input_row.get("category", ""))
    ranking = int(input_row.get("rank_in_category") or 0)
    try:
        payload = request_json(TIKI_DETAIL_API.format(product_id=product_id))
        return build_product_row(payload, category, ranking)
    except Exception:
        logging.exception("Failed to crawl detail for product_id=%s", product_id)
        return None


def crawl_detail(
    limit: int | None = None,
    workers: int | None = None,
    keep_listing: bool = False,
    retry_rounds: int = 3,
) -> pd.DataFrame:
    if not LISTING_FILE.exists():
        raise FileNotFoundError(f"Missing listing file: {LISTING_FILE}. Run crawl_listing.py first.")

    listing_df = pd.read_csv(LISTING_FILE)
    if limit is not None and limit > 0:
        listing_df = listing_df.head(limit)

    records = listing_df.to_dict(orient="records")
    max_workers = workers or min(SAFE_MAX_WORKERS, (os.cpu_count() or 1))

    record_by_id: dict[int, dict] = {}
    for rec in records:
        pid = int(rec["product_id"])
        record_by_id[pid] = rec

    success_rows: dict[int, dict] = {}
    pending_ids = set(record_by_id.keys())

    for round_idx in range(max(1, retry_rounds)):
        if not pending_ids:
            break

        round_workers = max(1, max_workers // (2**round_idx))
        logging.info(
            "Detail crawl round=%s pending=%s workers=%s",
            round_idx + 1,
            len(pending_ids),
            round_workers,
        )

        round_pending = list(pending_ids)
        with ThreadPoolExecutor(max_workers=round_workers) as executor:
            futures = [
                executor.submit(fetch_single_detail, record_by_id[pid])
                for pid in round_pending
            ]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue
                pid = int(result["product_id"])
                success_rows[pid] = result
                pending_ids.discard(pid)

        logging.info(
            "Detail crawl round=%s completed success=%s remaining=%s",
            round_idx + 1,
            len(success_rows),
            len(pending_ids),
        )

    rows = list(success_rows.values())

    products_df = pd.DataFrame(rows)
    if products_df.empty:
        products_df = pd.DataFrame(
            columns=[
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
        )

    products_df = products_df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)

    products_df.to_csv(PRODUCTS_FILE, index=False)
    authors_df = build_authors_table(products_df)
    authors_df.to_csv(AUTHORS_FILE, index=False)

    if not keep_listing and LISTING_FILE.exists():
        LISTING_FILE.unlink()
        logging.info("Removed intermediate listing file: %s", LISTING_FILE)

    logging.info("Saved products=%s rows to %s", len(products_df), PRODUCTS_FILE)
    logging.info("Saved authors=%s rows to %s", len(authors_df), AUTHORS_FILE)
    print(f"Saved products: {PRODUCTS_FILE} | rows={len(products_df)}")
    print(f"Saved authors:  {AUTHORS_FILE} | rows={len(authors_df)}")
    return products_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl detail data for product IDs from listing CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of product IDs to crawl.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(SAFE_MAX_WORKERS, (os.cpu_count() or 1)),
        help="Number of parallel workers for detail crawling.",
    )
    parser.add_argument("--keep-listing", action="store_true", help="Keep listing_member_2.csv after detail crawl.")
    parser.add_argument("--retry-rounds", type=int, default=3, help="Retry rounds for failed product detail requests.")
    args = parser.parse_args()
    crawl_detail(
        limit=args.limit,
        workers=args.workers,
        keep_listing=args.keep_listing,
        retry_rounds=args.retry_rounds,
    )
