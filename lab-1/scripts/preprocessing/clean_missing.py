import pandas as pd


def _remove_duplicates(df: pd.DataFrame, subset_col: str, dataset_name: str) -> tuple[pd.DataFrame, int]:
    """Drop duplicated rows by a key column and return removed count."""
    before = len(df)
    if subset_col in df.columns:
        deduped = df.drop_duplicates(subset=[subset_col], keep="first").copy()
    else:
        deduped = df.copy()
        print(f"[{dataset_name}] Column '{subset_col}' not found. Skip duplicate removal by key.")

    removed = before - len(deduped)
    return deduped, removed


def _drop_missing_required(df: pd.DataFrame, required_cols: list[str], dataset_name: str) -> pd.DataFrame:
    """Drop rows with missing values in required columns that are present."""
    existing_cols = [col for col in required_cols if col in df.columns]
    if not existing_cols:
        print(f"[{dataset_name}] No required columns found for missing-value filtering.")
        return df.copy()

    return df.dropna(subset=existing_cols).copy()


def _coerce_products_types(products: pd.DataFrame) -> pd.DataFrame:
    """Ensure product_id=int, price=float, sold_count=int for products dataset."""
    cleaned = products.copy()

    if "product_id" in cleaned.columns:
        cleaned["product_id"] = pd.to_numeric(cleaned["product_id"], errors="coerce")
        cleaned = cleaned.dropna(subset=["product_id"]).copy()
        cleaned["product_id"] = cleaned["product_id"].astype("int64")

    if "price" in cleaned.columns:
        cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
        cleaned = cleaned.dropna(subset=["price"]).copy()
        cleaned["price"] = cleaned["price"].astype("float64")

    if "sold_count" in cleaned.columns:
        cleaned["sold_count"] = pd.to_numeric(cleaned["sold_count"], errors="coerce").fillna(0)
        cleaned["sold_count"] = cleaned["sold_count"].astype("int64")

    return cleaned


def _coerce_reviews_types(reviews: pd.DataFrame) -> pd.DataFrame:
    """Ensure product_id=int for reviews dataset when available."""
    cleaned = reviews.copy()

    if "product_id" in cleaned.columns:
        cleaned["product_id"] = pd.to_numeric(cleaned["product_id"], errors="coerce")
        cleaned = cleaned.dropna(subset=["product_id"]).copy()
        cleaned["product_id"] = cleaned["product_id"].astype("int64")

    return cleaned


def clean_merged_datasets(
    merged_products: pd.DataFrame,
    merged_reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean merged products and reviews DataFrames.

    Steps:
    1) Remove duplicates by product_id/review_id
    2) Drop rows with missing values in required columns
    3) Enforce required data types
    4) Print duplicate stats and final shapes
    """
    products = merged_products.copy()
    reviews = merged_reviews.copy()

    products, product_dupes_removed = _remove_duplicates(products, "product_id", "products")
    reviews, review_dupes_removed = _remove_duplicates(reviews, "review_id", "reviews")

    products = _drop_missing_required(products, ["product_id", "title", "price"], "products")
    reviews = _drop_missing_required(reviews, ["product_id"], "reviews")

    cleaned_products = _coerce_products_types(products)
    cleaned_reviews = _coerce_reviews_types(reviews)

    print(f"[products] Duplicates removed (by product_id): {product_dupes_removed}")
    print(f"[reviews] Duplicates removed (by review_id): {review_dupes_removed}")
    print(f"[products] Final shape: {cleaned_products.shape}")
    print(f"[reviews] Final shape: {cleaned_reviews.shape}")

    return cleaned_products, cleaned_reviews
