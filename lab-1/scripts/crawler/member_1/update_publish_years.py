import argparse
import csv
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import requests

DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://tiki.vn/",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_3_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0 Safari/537.36",
]


def parse_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip().lower()


def parse_spec_value(specifications: list, key_candidates: list[str]) -> str:
    key_candidates = [x.lower().strip() for x in key_candidates]
    for spec in specifications or []:
        for attr in spec.get("attributes", []):
            attr_name = str(attr.get("name", "")).strip().lower()
            if attr_name in key_candidates:
                return str(attr.get("value", "")).strip()
    return ""


def parse_publish_year(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if match:
        return int(match.group(1))
    return None


def request_json(url: str, timeout: int, retries: int, sleep_range: tuple[float, float]) -> dict:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            headers = dict(DEFAULT_HEADERS)
            headers["User-Agent"] = random.choice(USER_AGENTS)
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logging.warning("Request failed (%s/%s): %s", attempt + 1, retries, exc)
            time.sleep(random.uniform(*sleep_range))
    if last_error:
        raise last_error
    return {}


def extract_publish_year(payload: dict) -> Optional[int]:
    # Try top-level keys first.
    publish_year = parse_publish_year(payload.get("publish_year"))
    if publish_year is None:
        publish_year = parse_publish_year(payload.get("publication_date"))
    if publish_year is not None:
        return publish_year

    # Fallback to specifications attributes (same approach as member_2).
    specs = payload.get("specifications") or []
    spec_value = parse_spec_value(
        specs,
        [
            "năm xuất bản",
            "ngày xuất bản",
            "publish year",
            "publication year",
            "publication date",
            "published date",
        ],
    )
    if spec_value:
        return parse_publish_year(spec_value)
    return None


def read_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    content = path.read_text(encoding="utf-8").strip()
    return int(content) if content.isdigit() else 0


def write_checkpoint(path: Path, row_index: int) -> None:
    path.write_text(str(row_index), encoding="utf-8")


def fetch_publish_year(
    product_id: str,
    timeout: int,
    retries: int,
    sleep_min: float,
    sleep_max: float,
) -> Optional[int]:
    time.sleep(random.uniform(sleep_min, sleep_max))
    url = f"https://tiki.vn/api/v2/products/{product_id}"
    payload = request_json(url, timeout=timeout, retries=retries, sleep_range=(sleep_min, sleep_max))
    return extract_publish_year(payload)


def update_file(
    csv_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    sleep_min: float,
    sleep_max: float,
    timeout: int,
    retries: int,
    max_rows: Optional[int],
    resume: bool,
    workers: int,
    batch_size: int,
) -> None:
    start_row = read_checkpoint(checkpoint_path) if resume else 0
    total = 0
    updated = 0
    append_mode = resume and output_path.exists() and start_row > 0
    with csv_path.open(newline="", encoding="utf-8") as src, output_path.open(
        "a" if append_mode else "w",
        newline="",
        encoding="utf-8",
    ) as dst:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError("Missing header in CSV.")
        fieldnames = list(reader.fieldnames)
        if "publish_year" not in fieldnames:
            raise ValueError("CSV does not contain publish_year column.")
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        if not append_mode:
            writer.writeheader()

        def process_batch(batch_rows: list[tuple[int, dict]]) -> None:
            nonlocal updated
            futures: dict = {}
            results: dict[int, Optional[int]] = {}
            for row_index, row_data in batch_rows:
                current = (row_data.get("publish_year") or "").strip()
                if current:
                    continue
                product_id = row_data.get("product_id")
                if not product_id:
                    continue
                futures[executor.submit(
                    fetch_publish_year,
                    product_id,
                    timeout,
                    retries,
                    sleep_min,
                    sleep_max,
                )] = row_index

            for future in as_completed(futures):
                row_index = futures[future]
                try:
                    results[row_index] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logging.warning("Failed to fetch publish_year at row %s: %s", row_index, exc)
                    results[row_index] = None

            for row_index, row_data in batch_rows:
                publish_year = results.get(row_index)
                if publish_year is not None:
                    row_data["publish_year"] = str(publish_year)
                    updated += 1
                writer.writerow(row_data)
                if resume:
                    write_checkpoint(checkpoint_path, row_index)

        batch_rows: list[tuple[int, dict]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for row in reader:
                total += 1
                if max_rows and total > max_rows:
                    break
                if total <= start_row:
                    continue
                batch_rows.append((total, row))
                if len(batch_rows) >= batch_size:
                    process_batch(batch_rows)
                    batch_rows = []

            if batch_rows:
                process_batch(batch_rows)

    logging.info("Processed %s rows. Updated publish_year for %s rows.", total, updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill publish_year for member_1 products.csv")
    parser.add_argument(
        "--input",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "raw"
            / "member_1"
            / "products.csv"
        ),
        help="Path to products.csv",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "raw"
            / "member_1"
            / "products.updated.csv"
        ),
        help="Output CSV path",
    )
    parser.add_argument("--sleep-min", type=float, default=1.0)
    parser.add_argument("--sleep-max", type=float, default=2.5)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=0, help="Limit rows for a quick test.")
    parser.add_argument("--workers", type=int, default=6, help="Number of concurrent workers.")
    parser.add_argument("--batch-size", type=int, default=60, help="Rows per batch before writing.")
    parser.add_argument(
        "--checkpoint",
        default=str(
            Path(__file__).resolve().parents[2]
            / "data"
            / "raw"
            / "member_1"
            / "products.updated.checkpoint"
        ),
        help="Checkpoint file for resume.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    max_rows = args.max_rows if args.max_rows > 0 else None
    update_file(
        csv_path=Path(args.input),
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        sleep_min=args.sleep_min,
        sleep_max=args.sleep_max,
        timeout=args.timeout,
        retries=args.retries,
        max_rows=max_rows,
        resume=args.resume,
        workers=max(1, args.workers),
        batch_size=max(1, args.batch_size),
    )


if __name__ == "__main__":
    random.seed(42)
    main()
