import pandas as pd


def _resolve_author_column(products: pd.DataFrame) -> str:
    """Find the author column name used in the products dataset."""
    for candidate in ["author", "author_name"]:
        if candidate in products.columns:
            return candidate
    raise ValueError("Missing author column. Expected one of: 'author', 'author_name'.")


def _normalize_author_names(author_series: pd.Series) -> pd.Series:
    """Trim whitespace and standardize empty names as missing values."""
    normalized = author_series.astype("string").str.strip()
    normalized = normalized.where(normalized.notna() & (normalized != ""), pd.NA)
    return normalized


def rebuild_authors_table(
    cleaned_products: pd.DataFrame,
    missing_author_strategy: str = "label_unknown",
) -> pd.DataFrame:
    """
    Build authors summary table from cleaned products.

    Output columns:
    - author_name
    - total_books
    - total_sales
    - average_rating

    Parameters:
    - cleaned_products: input products DataFrame
    - missing_author_strategy:
        - 'label_unknown': replace missing author names with 'Unknown'
        - 'drop': remove rows with missing author names
    """
    if not isinstance(cleaned_products, pd.DataFrame):
        raise TypeError("cleaned_products must be a pandas DataFrame.")

    required_numeric_cols = ["product_id", "sold_count", "rating"]
    missing_cols = [col for col in required_numeric_cols if col not in cleaned_products.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    author_col = _resolve_author_column(cleaned_products)

    df = cleaned_products[[author_col, "product_id", "sold_count", "rating"]].copy()
    df[author_col] = _normalize_author_names(df[author_col])

    if missing_author_strategy == "drop":
        df = df.dropna(subset=[author_col]).copy()
    elif missing_author_strategy == "label_unknown":
        df[author_col] = df[author_col].fillna("Unknown")
    else:
        raise ValueError("missing_author_strategy must be either 'label_unknown' or 'drop'.")

    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
    df["sold_count"] = pd.to_numeric(df["sold_count"], errors="coerce").fillna(0)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    df = df.dropna(subset=["product_id"]).copy()

    authors_df = (
        df.groupby(author_col, as_index=False)
        .agg(
            total_books=("product_id", "count"),
            total_sales=("sold_count", "sum"),
            average_rating=("rating", "mean"),
        )
        .rename(columns={author_col: "author_name"})
        .sort_values(by="total_sales", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )

    authors_df["total_books"] = authors_df["total_books"].astype("int64")
    authors_df["total_sales"] = authors_df["total_sales"].round().astype("int64")
    authors_df["average_rating"] = authors_df["average_rating"].astype("float64")

    return authors_df


def rebuild_authors(
    cleaned_products: pd.DataFrame,
    missing_author_strategy: str = "label_unknown",
) -> pd.DataFrame:
    """Wrapper that rebuilds authors summary from cleaned products."""
    return rebuild_authors_table(
        cleaned_products=cleaned_products,
        missing_author_strategy=missing_author_strategy,
    )
