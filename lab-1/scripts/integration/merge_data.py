import os

import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case."""
    normalized = df.copy()
    normalized.columns = (
        pd.Index(normalized.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return normalized


def read_csv_utf8(file_path: str) -> pd.DataFrame:
    """Read a CSV using UTF-8 and fall back to UTF-8 with BOM when needed."""
    try:
        return pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="utf-8-sig")


def discover_csv_files(raw_dirs: list[str]) -> dict[str, list[str]]:
    """Discover all CSV files grouped by dataset name (filename without extension)."""
    datasets: dict[str, list[str]] = {}

    for raw_dir in raw_dirs:
        if not os.path.isdir(raw_dir):
            print(f"[warn] Directory not found: {raw_dir}")
            continue

        for filename in os.listdir(raw_dir):
            if not filename.lower().endswith(".csv"):
                continue

            dataset_name = os.path.splitext(filename)[0].lower()
            file_path = os.path.join(raw_dir, filename)
            datasets.setdefault(dataset_name, []).append(file_path)

    return datasets


def merge_dataset(dataset_name: str, file_paths: list[str]) -> pd.DataFrame:
    """Read, normalize, and merge CSV files for a single dataset."""
    if not file_paths:
        print(f"[{dataset_name}] No files found. Returning empty DataFrame.")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for file_path in file_paths:
        df = read_csv_utf8(file_path)
        print(f"[{dataset_name}] Loaded {file_path} with shape {df.shape}")

        normalized_df = normalize_column_names(df)
        print(f"[{dataset_name}] Normalized columns for {file_path}: {normalized_df.shape}")

        frames.append(normalized_df)

    total_rows_before_merge = sum(frame.shape[0] for frame in frames)
    merged_df = pd.concat(frames, ignore_index=True, sort=False)

    print(
        f"[{dataset_name}] Merged {len(frames)} file(s). "
        f"Rows before merge: {total_rows_before_merge}; shape after merge: {merged_df.shape}"
    )

    return merged_df


def merge_data(raw_dirs: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge raw datasets from members into unified products and reviews DataFrames.

    - products.csv files are concatenated
    - reviews.csv files are concatenated
    - authors.csv files are intentionally skipped (recomputed later)
    """
    if raw_dirs is None:
        raw_dirs = ["data/raw/member_1", "data/raw/member_2"]

    datasets = discover_csv_files(raw_dirs)

    discovered = ", ".join(sorted(datasets.keys())) if datasets else "none"
    print(f"Discovered CSV datasets: {discovered}")

    if "authors" in datasets:
        print("[authors] Found authors.csv files but skipped by design (will be recomputed later).")

    merged_products = merge_dataset("products", datasets.get("products", []))
    merged_reviews = merge_dataset("reviews", datasets.get("reviews", []))

    return merged_products, merged_reviews


if __name__ == "__main__":
    products_df, reviews_df = merge_data()
    print(f"[result] merged_products shape: {products_df.shape}")
    print(f"[result] merged_reviews shape: {reviews_df.shape}")
