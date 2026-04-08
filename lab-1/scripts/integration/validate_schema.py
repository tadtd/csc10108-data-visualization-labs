import pandas as pd


def _print_check_result(check_name: str, passed: bool, details: str) -> None:
    """Print one validation line with pass/fail status."""
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {check_name}: {details}")


def _ensure_dataframe(name: str, value: pd.DataFrame) -> None:
    """Validate that an input is a pandas DataFrame."""
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")


def check_required_product_columns(
    cleaned_products: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> tuple[bool, str]:
    """Check required columns exist in products."""
    if required_columns is None:
        required_columns = ["product_id", "title", "price", "sold_count"]

    missing = [col for col in required_columns if col not in cleaned_products.columns]
    if missing:
        return False, f"Missing columns: {missing}"

    return True, f"All required columns present: {required_columns}"


def check_no_nulls_in_products(
    cleaned_products: pd.DataFrame,
    not_null_columns: list[str] | None = None,
) -> tuple[bool, str]:
    """Check there are no nulls in critical products columns."""
    if not_null_columns is None:
        not_null_columns = ["product_id", "title", "price"]

    missing_for_check = [col for col in not_null_columns if col not in cleaned_products.columns]
    if missing_for_check:
        return False, f"Cannot validate nulls because columns are missing: {missing_for_check}"

    null_counts = cleaned_products[not_null_columns].isna().sum()
    failing = null_counts[null_counts > 0]
    if not failing.empty:
        details = ", ".join(f"{col}={int(count)}" for col, count in failing.items())
        return False, f"Found null values: {details}"

    return True, "No null values found in required columns"


def check_review_product_referential_integrity(
    cleaned_products: pd.DataFrame,
    cleaned_reviews: pd.DataFrame,
) -> tuple[bool, str]:
    """Ensure every reviews.product_id exists in products.product_id."""
    if "product_id" not in cleaned_products.columns:
        return False, "products.product_id column is missing"
    if "product_id" not in cleaned_reviews.columns:
        return False, "reviews.product_id column is missing"

    valid_product_ids = set(cleaned_products["product_id"].dropna().tolist())
    invalid_mask = ~cleaned_reviews["product_id"].isin(valid_product_ids)
    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        sample_invalid = cleaned_reviews.loc[invalid_mask, "product_id"].head(5).tolist()
        return False, f"{invalid_count} review rows have unknown product_id. Sample: {sample_invalid}"

    return True, "All review.product_id values exist in products.product_id"


def check_products_minimum_size(
    cleaned_products: pd.DataFrame,
    minimum_rows: int = 5000,
) -> tuple[bool, str]:
    """Check products dataset has at least minimum_rows rows."""
    row_count = len(cleaned_products)
    if row_count < minimum_rows:
        return False, f"Row count is {row_count}, required at least {minimum_rows}"

    return True, f"Row count is {row_count} (>= {minimum_rows})"


def validate_datasets(
    cleaned_products: pd.DataFrame,
    cleaned_reviews: pd.DataFrame,
    authors_df: pd.DataFrame,
    minimum_product_rows: int = 5000,
) -> bool:
    """
    Run quality checks before saving processed datasets.

    Critical checks:
    1. Required product columns exist
    2. No nulls in product_id/title/price
    3. Review/product referential integrity
    4. Minimum products dataset size
    """
    _ensure_dataframe("cleaned_products", cleaned_products)
    _ensure_dataframe("cleaned_reviews", cleaned_reviews)
    _ensure_dataframe("authors_df", authors_df)

    checks: list[tuple[str, bool, str]] = []

    required_ok, required_detail = check_required_product_columns(cleaned_products)
    checks.append(("Required Columns In Products", required_ok, required_detail))

    nulls_ok, nulls_detail = check_no_nulls_in_products(cleaned_products)
    checks.append(("No Nulls In Critical Product Fields", nulls_ok, nulls_detail))

    ref_ok, ref_detail = check_review_product_referential_integrity(cleaned_products, cleaned_reviews)
    checks.append(("Referential Integrity reviews.product_id -> products.product_id", ref_ok, ref_detail))

    size_ok, size_detail = check_products_minimum_size(cleaned_products, minimum_rows=minimum_product_rows)
    checks.append(("Minimum Products Dataset Size", size_ok, size_detail))

    print("Validation report:")
    for check_name, passed, details in checks:
        _print_check_result(check_name, passed, details)

    print(f"[INFO] authors_df shape: {authors_df.shape}")

    failed_critical_checks = [check_name for check_name, passed, _ in checks if not passed]
    if failed_critical_checks:
        raise ValueError(
            "Critical validation check(s) failed: " + ", ".join(failed_critical_checks)
        )

    print("Validation passed")
    return True


def validate_schema(
    cleaned_products: pd.DataFrame,
    cleaned_reviews: pd.DataFrame,
    authors_df: pd.DataFrame,
    minimum_product_rows: int = 5000,
) -> bool:
    """Wrapper for the project pipeline contract."""
    return validate_datasets(
        cleaned_products=cleaned_products,
        cleaned_reviews=cleaned_reviews,
        authors_df=authors_df,
        minimum_product_rows=minimum_product_rows,
    )


if __name__ == "__main__":
    print("Use validate_datasets(cleaned_products, cleaned_reviews, authors_df) in the pipeline.")
