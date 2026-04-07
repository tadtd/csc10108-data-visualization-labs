import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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
OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "member_2"
CONFIG_PATH = ROOT_DIR / "config" / "categories_member_2.json"
LOG_FILE = ROOT_DIR / "logs" / "member_2.log"

LISTING_FILE = OUTPUT_DIR / "listing_member_2.csv"
TIKI_LISTING_API = "https://tiki.vn/api/v2/products"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
SLEEP_RANGE = (1, 3)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_categories(config_path: Path) -> list[dict]:
    if not config_path.exists():
        raise FileNotFoundError(f"Missing category config: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = []
    for item in data:
        if isinstance(item, dict):
            raw_queries = item.get("queries")
            if isinstance(raw_queries, list):
                queries = [str(q).strip() for q in raw_queries if str(q).strip()]
            else:
                base_q = item.get("query") or item.get("name", "")
                queries = [str(base_q).strip()] if str(base_q).strip() else []

            categories.append(
                {
                    "name": item.get("name", "unknown"),
                    "queries": queries,
                }
            )
        else:
            q = str(item).strip()
            categories.append({"name": q, "queries": [q] if q else []})
    return categories


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
            logging.error("Listing request failed (attempt %s/%s): %s", attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                raise


def extract_listing_rows(category_name: str, items: list, page: int) -> list[dict]:
    rows = []
    for idx, item in enumerate(items, start=1):
        product_id = item.get("id")
        if not product_id:
            continue

        primary_category_name = str(item.get("primary_category_name") or "")
        category_ids = item.get("category_ids") or []
        if "sách" not in primary_category_name.lower() and not category_ids:
            continue

        url_path = item.get("url_path")
        rows.append(
            {
                "product_id": int(product_id),
                "title": item.get("name", ""),
                "category": category_name,
                "source_primary_category": primary_category_name,
                "source_category_ids": "|".join(str(x) for x in category_ids),
                "url": f"https://tiki.vn/{url_path}" if url_path else "",
                "rank_in_category": (page - 1) * len(items) + idx,
            }
        )
    return rows


def crawl_category_listing(category_name: str, queries: list[str], target_count: int, per_page: int = 40) -> list[dict]:
    sort_options = ["top_seller", "newest", "price,asc", "price,desc"]
    max_pages_per_combo = 12

    logging.info("Start category=%s target=%s queries=%s", category_name, target_count, len(queries))
    rows = []
    seen_ids: set[int] = set()

    for query in queries:
        if len(rows) >= target_count:
            break

        for sort in sort_options:
            if len(rows) >= target_count:
                break

            for page in range(1, max_pages_per_combo + 1):
                if len(rows) >= target_count:
                    break

                params = {
                    "q": query,
                    "limit": per_page,
                    "page": page,
                    "sort": sort,
                    "aggregations": 2,
                    "version": "home-persionalized",
                    "trackity_id": "",
                }
                try:
                    payload = request_json(TIKI_LISTING_API, params)
                except Exception:
                    logging.exception(
                        "Request error category=%s query=%s sort=%s page=%s",
                        category_name,
                        query,
                        sort,
                        page,
                    )
                    break

                items = payload.get("data", [])
                if not items:
                    break

                candidate_rows = extract_listing_rows(category_name, items, page)
                added = 0
                for row in candidate_rows:
                    pid = row["product_id"]
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    row["rank_in_category"] = len(rows) + 1
                    rows.append(row)
                    added += 1
                    if len(rows) >= target_count:
                        break

                logging.info(
                    "category=%s query=%s sort=%s page=%s added=%s total=%s",
                    category_name,
                    query,
                    sort,
                    page,
                    added,
                    len(rows),
                )

    return rows[:target_count]


def crawl_listing(total_target: int = 5000, workers: int | None = None) -> pd.DataFrame:
    print(f"ROOT_DIR={ROOT_DIR}")
    print(f"CONFIG_PATH={CONFIG_PATH}")
    print(f"OUTPUT_DIR={OUTPUT_DIR}")

    categories = load_categories(CONFIG_PATH)
    if not categories:
        raise ValueError(f"No categories found in {CONFIG_PATH}")

    # Oversample per category because the same product can appear across multiple category queries.
    per_category_target = max(1, int((total_target / len(categories)) * 1.4))
    all_rows = []
    max_workers = workers or (os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(categories)))) as executor:
        future_map = {
            executor.submit(
                crawl_category_listing,
                category_name=cat["name"],
                queries=cat.get("queries", []),
                target_count=per_category_target,
            ): cat
            for cat in categories
        }
        for future in as_completed(future_map):
            cat = future_map[future]
            cat_rows = future.result()
            all_rows.extend(cat_rows)
            print(f"[{cat['name']}] collected: {len(cat_rows)}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        logging.warning("No listing data collected")
        df.to_csv(LISTING_FILE, index=False)
        return df

    df = df.drop_duplicates(subset=["product_id"]).reset_index(drop=True)
    if len(df) > total_target:
        df = df.head(total_target)

    df.to_csv(LISTING_FILE, index=False)
    logging.info("Listing saved to %s with %s rows", LISTING_FILE, len(df))
    print(f"Saved listing: {LISTING_FILE} | rows={len(df)}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl listing data for member_2 categories.")
    parser.add_argument("--target-total", type=int, default=5000, help="Target number of listing rows.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers (logical CPU threads).")
    args = parser.parse_args()
    crawl_listing(total_target=args.target_total, workers=args.workers)
