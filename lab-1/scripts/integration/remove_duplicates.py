import pandas as pd

from scripts.preprocessing.clean_missing import clean_merged_datasets


def remove_duplicates(
    merged_products: pd.DataFrame,
    merged_reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean merged datasets and return cleaned products/reviews.

    This keeps the pipeline contract at remove_duplicates(), while applying
    duplicate removal, missing-value filtering, and dtype coercion.
    """
    print("[remove_duplicates] Cleaning merged datasets...")
    cleaned_products, cleaned_reviews = clean_merged_datasets(merged_products, merged_reviews)
    print("[remove_duplicates] Cleaning completed.")
    return cleaned_products, cleaned_reviews
