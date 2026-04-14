import logging
from pathlib import Path

import pandas as pd

from scripts.integration.merge_data import merge_data
from scripts.integration.remove_duplicates import remove_duplicates
from scripts.integration.validate_schema import validate_schema
from scripts.preprocessing.build_final_dataset import rebuild_authors


LOGGER = logging.getLogger("data_pipeline")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _save_final_datasets(
    cleaned_products: pd.DataFrame,
    cleaned_reviews: pd.DataFrame,
    authors_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist final datasets as UTF-8 CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_products.to_csv(output_dir / "products.csv", index=False, encoding="utf-8")
    cleaned_reviews.to_csv(output_dir / "reviews.csv", index=False, encoding="utf-8")
    authors_df.to_csv(output_dir / "authors.csv", index=False, encoding="utf-8")


def _remove_orphan_reviews(
    cleaned_products: pd.DataFrame,
    cleaned_reviews: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Remove reviews whose product_id does not exist in products."""
    if "product_id" not in cleaned_products.columns:
        raise ValueError("Missing required column in products: product_id")
    if "product_id" not in cleaned_reviews.columns:
        raise ValueError("Missing required column in reviews: product_id")

    valid_product_ids = set(cleaned_products["product_id"].dropna().tolist())
    before_count = len(cleaned_reviews)
    filtered_reviews = cleaned_reviews[cleaned_reviews["product_id"].isin(valid_product_ids)].copy()
    removed_count = before_count - len(filtered_reviews)
    return filtered_reviews, removed_count


def run_pipeline(
    output_dir: str = "data/raw",
    auto_clean_orphan_reviews: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run full data integration pipeline and return final DataFrames."""
    output_path = Path(output_dir)

    LOGGER.info("START step: merge_data")
    merged_products, merged_reviews = merge_data()
    LOGGER.info("END step: merge_data | products=%s reviews=%s", merged_products.shape, merged_reviews.shape)

    LOGGER.info("START step: remove_duplicates")
    cleaned_products, cleaned_reviews = remove_duplicates(merged_products, merged_reviews)
    LOGGER.info(
        "END step: remove_duplicates | products=%s reviews=%s",
        cleaned_products.shape,
        cleaned_reviews.shape,
    )

    LOGGER.info("START step: rebuild_authors")
    authors_df = rebuild_authors(cleaned_products)
    LOGGER.info("END step: rebuild_authors | authors=%s", authors_df.shape)

    if auto_clean_orphan_reviews:
        LOGGER.info("START step: auto_clean_orphan_reviews")
        cleaned_reviews, orphan_removed = _remove_orphan_reviews(cleaned_products, cleaned_reviews)
        LOGGER.info(
            "END step: auto_clean_orphan_reviews | removed=%s reviews=%s",
            orphan_removed,
            cleaned_reviews.shape,
        )
    else:
        LOGGER.info("SKIP step: auto_clean_orphan_reviews | mode disabled")

    LOGGER.info("START step: validate_schema")
    validate_schema(cleaned_products, cleaned_reviews, authors_df)
    LOGGER.info("END step: validate_schema")

    LOGGER.info("START step: save_final_datasets")
    _save_final_datasets(cleaned_products, cleaned_reviews, authors_df, output_path)
    LOGGER.info("END step: save_final_datasets | output_dir=%s", output_path.as_posix())

    LOGGER.info(
        "FINAL SHAPES | products=%s reviews=%s authors=%s",
        cleaned_products.shape,
        cleaned_reviews.shape,
        authors_df.shape,
    )

    return cleaned_products, cleaned_reviews, authors_df


def main() -> None:
    _configure_logging()
    try:
        run_pipeline(output_dir="data/raw", auto_clean_orphan_reviews=True)
    except Exception:
        LOGGER.exception("Pipeline execution failed")
        raise


if __name__ == "__main__":
    main()
