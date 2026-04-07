import json
import logging
import random
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

BASE_URL = "https://tiki.vn/api/v2"
PRODUCTS_URL = f"{BASE_URL}/products"
REVIEWS_URL = f"{BASE_URL}/reviews"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
]

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "vi-VN,vi;q=0.9,en;q=0.8",
    "referer": "https://tiki.vn/",
}


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return normalized.lower()


def request_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    retries: int = 3,
    timeout: int = 15,
    sleep_range: Tuple[float, float] = (1.0, 2.5),
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            headers = dict(DEFAULT_HEADERS)
            headers["user-agent"] = random.choice(USER_AGENTS)
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logging.warning("Request failed (%s/%s): %s", attempt + 1, retries, exc)
            time.sleep(random.uniform(*sleep_range))
    if last_error:
        raise last_error
    return {}


def parse_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_sold_count(quantity_sold: Any) -> int:
    if isinstance(quantity_sold, dict):
        value = quantity_sold.get("value")
        if value is not None:
            return int(value)
        text = quantity_sold.get("text", "")
    else:
        text = str(quantity_sold or "")

    normalized = normalize_text(text)
    if not normalized:
        return 0
    normalized = normalized.replace("+", "").replace(" ", "")
    if "k" in normalized:
        try:
            return int(float(normalized.replace("k", "")) * 1000)
        except ValueError:
            return 0
    return parse_int(normalized) or 0


def extract_attributes(detail: Dict[str, Any]) -> Dict[str, Optional[str]]:
    author = None
    publisher = None
    publish_year = None
    page_count = None

    authors = detail.get("authors") or []
    if authors:
        names = [item.get("name") for item in authors if item.get("name")]
        if names:
            author = ", ".join(names)

    publisher = detail.get("publisher", {}).get("name") or detail.get("publisher")

    specs = detail.get("specifications") or []
    for group in specs:
        for attr in group.get("attributes", []) or []:
            name = normalize_text(attr.get("name"))
            value = attr.get("value")
            if not value:
                continue
            if not author and "tac gia" in name:
                author = str(value)
            elif not publisher and "nha xuat ban" in name:
                publisher = str(value)
            elif not publish_year and "nam xuat ban" in name:
                publish_year = str(value)
            elif not page_count and "so trang" in name:
                page_count = str(value)

    return {
        "author": author,
        "publisher": publisher,
        "publish_year": publish_year,
        "page_count": page_count,
    }


def extract_category(detail: Dict[str, Any], fallback: Optional[str]) -> str:
    category = detail.get("categories", {}).get("name")
    if category:
        return category
    breadcrumbs = detail.get("breadcrumbs") or []
    if breadcrumbs:
        name = breadcrumbs[-1].get("name")
        if name:
            return name
    return fallback or ""


def has_gift_flag(title: str, description: str) -> bool:
    normalized = normalize_text(" ".join([title, description]))
    return any(keyword in normalized for keyword in ["qua tang", "tang kem", "gift"])


def is_combo_product(title: str) -> bool:
    normalized = normalize_text(title)
    return "combo" in normalized or "set" in normalized


def is_bestseller_product(detail: Dict[str, Any]) -> bool:
    if detail.get("is_bestseller") is True:
        return True
    badges = detail.get("badges") or []
    for badge in badges:
        text = normalize_text(badge.get("text") or "")
        if "bestseller" in text or "ban chay" in text:
            return True
    return False


def search_products(
    query: str,
    max_items: int,
    per_page: int,
    max_pages: int,
    sleep_range: Tuple[float, float],
) -> List[int]:
    product_ids: List[int] = []
    for page in range(1, max_pages + 1):
        params = {
            "q": query,
            "page": page,
            "limit": per_page,
        }
        payload = request_json(PRODUCTS_URL, params=params, sleep_range=sleep_range)
        items = payload.get("data", [])
        if not items:
            break
        for item in items:
            product_id = item.get("id") or item.get("product_id")
            if product_id is None:
                continue
            product_ids.append(int(product_id))
            if len(product_ids) >= max_items:
                return product_ids
        time.sleep(random.uniform(*sleep_range))
    return product_ids


def fetch_product_detail(product_id: int, sleep_range: Tuple[float, float]) -> Dict[str, Any]:
    payload = request_json(f"{PRODUCTS_URL}/{product_id}", sleep_range=sleep_range)
    return payload or {}


def fetch_reviews(
    product_id: int,
    max_reviews: int,
    per_page: int,
    sleep_range: Tuple[float, float],
) -> List[Dict[str, Any]]:
    reviews: List[Dict[str, Any]] = []
    page = 1
    while len(reviews) < max_reviews:
        params = {
            "product_id": product_id,
            "page": page,
            "limit": per_page,
        }
        payload = request_json(REVIEWS_URL, params=params, sleep_range=sleep_range)
        items = payload.get("data", [])
        if not items:
            break
        for item in items:
            reviews.append(item)
            if len(reviews) >= max_reviews:
                break
        page += 1
        time.sleep(random.uniform(*sleep_range))
    return reviews


def build_product_record(
    detail: Dict[str, Any],
    ranking: int,
    category_name: str,
) -> Dict[str, Any]:
    attrs = extract_attributes(detail)
    title = detail.get("name") or ""
    description = detail.get("description") or detail.get("short_description") or ""
    price = parse_float(detail.get("price")) or 0.0
    original_price = parse_float(detail.get("original_price")) or price
    discount_percent = parse_int(detail.get("discount_rate"))
    if discount_percent is None and original_price:
        discount_percent = int(round((1 - price / original_price) * 100)) if original_price else 0
    if discount_percent is not None and discount_percent < 0:
        discount_percent = 0
    sold_count = parse_sold_count(detail.get("quantity_sold"))
    rating = parse_float(detail.get("rating_average")) or 0.0
    review_count = parse_int(detail.get("review_count")) or 0

    publish_year = parse_int(attrs.get("publish_year"))
    page_count = parse_int(attrs.get("page_count"))

    category = extract_category(detail, category_name)

    return {
        "product_id": parse_int(detail.get("id")) or 0,
        "title": title,
        "price": price,
        "original_price": original_price,
        "discount_percent": discount_percent or 0,
        "sold_count": sold_count,
        "ranking": ranking,
        "rating": rating,
        "review_count": review_count,
        "author": attrs.get("author") or "",
        "publisher": attrs.get("publisher") or "",
        "category": category,
        "publish_year": publish_year,
        "page_count": page_count,
        "description": description,
        "is_bestseller": is_bestseller_product(detail),
        "is_top100": ranking <= 100,
        "is_combo": is_combo_product(title),
        "has_gift": has_gift_flag(title, description),
    }


def build_review_record(review: Dict[str, Any], product_id: int) -> Dict[str, Any]:
    return {
        "review_id": parse_int(review.get("id")) or 0,
        "product_id": product_id,
        "rating_score": parse_float(review.get("rating")) or 0.0,
        "content": review.get("content") or "",
        "created_at": parse_int(review.get("created_at")) or 0,
    }


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
