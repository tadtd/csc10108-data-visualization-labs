from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.config import PRODUCTS_PATH
from dashboard.utils import load_data, to_bool


@dataclass
class DiscountPublisherRetriever:
  products_path: Path = PRODUCTS_PATH

  def load_products(self) -> pd.DataFrame:
    products = load_data(str(self.products_path))
    return self._prepare_products(products)

  def _prepare_products(self, products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    numeric_columns = [
      "price",
      "original_price",
      "discount_percent",
      "sold_count",
      "rating",
      "review_count",
      "publish_year",
      "page_count",
    ]
    for column in numeric_columns:
      if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["price", "discount_percent", "sold_count", "rating", "review_count"]:
      df[column] = df[column].fillna(0)

    df["publisher_clean"] = df.get("publisher", pd.Series(index=df.index, dtype="object")).fillna("Không rõ")
    df["publisher_clean"] = df["publisher_clean"].replace("", "Không rõ")

    category_source = df.get("category_binned", pd.Series(index=df.index, dtype="object")).fillna("")
    if "category" in df.columns:
      category_source = category_source.mask(category_source.eq(""), df["category"].fillna(""))
    df["genre"] = category_source.replace("", "Không rõ").astype(str).str.strip()

    for column in ["is_combo", "is_bestseller", "is_top100", "has_gift"]:
      if column in df.columns:
        df[column] = to_bool(df[column])
      else:
        df[column] = False

    df["rating_group"] = np.where(df["rating"] >= 4.5, ">= 4.5", "< 4.5")
    df["revenue_est"] = df["price"] * df["sold_count"]
    df["price_band"] = pd.cut(
      df["price"],
      bins=[-1, 100000, 200000, 350000, 500000, np.inf],
      labels=["<=100k", "100k-200k", "200k-350k", "350k-500k", ">500k"],
    )
    df["disc_band"] = pd.cut(
      df["discount_percent"],
      bins=[-1, 0, 10, 20, 30, 100],
      labels=["0%", "1-10%", "11-20%", "21-30%", ">30%"],
    )
    return df

  def with_publisher_group(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cloned = df.copy()
    publisher_sales = (
      cloned.groupby("publisher_clean", as_index=False)["sold_count"]
      .sum()
      .sort_values("sold_count", ascending=False)
    )
    top_publishers = publisher_sales.head(5)["publisher_clean"].tolist()
    cloned["publisher_group"] = np.where(cloned["publisher_clean"].isin(top_publishers), "Top 5 NXB", "Khác")
    return cloned, top_publishers

  def filter_products(
    self,
    products: pd.DataFrame,
    genres: list[str],
    price_range: tuple[float, float],
    rating_range: tuple[float, float],
    min_reviews: int,
  ) -> tuple[pd.DataFrame, list[str]]:
    filtered = products.copy()
    if genres:
      filtered = filtered[filtered["genre"].isin(genres)]

    min_price, max_price = price_range
    filtered = filtered[(filtered["price"] >= min_price) & (filtered["price"] <= max_price)]

    min_rating, max_rating = rating_range
    filtered = filtered[(filtered["rating"] >= min_rating) & (filtered["rating"] <= max_rating)]
    filtered = filtered[filtered["review_count"] >= min_reviews]

    filtered, top_publishers = self.with_publisher_group(filtered)
    return filtered, top_publishers

  def compute_kpis(self, filtered: pd.DataFrame) -> dict[str, float]:
    if filtered.empty:
      return {"total_products": 0, "total_sold": 0, "total_revenue": 0.0, "avg_discount": 0.0}
    return {
      "total_products": int(filtered["product_id"].nunique()),
      "total_sold": float(filtered["sold_count"].sum()),
      "total_revenue": float(filtered["revenue_est"].sum()),
      "avg_discount": float(filtered["discount_percent"].mean()),
    }

  def price_band_summary(self, filtered: pd.DataFrame) -> pd.DataFrame:
    return (
      filtered.groupby("price_band", observed=False, as_index=False)
      .agg(avg_sold=("sold_count", "mean"), product_count=("product_id", "nunique"))
      .dropna(subset=["price_band"])
      .sort_values("price_band")
    )

  def discount_band_summary(self, filtered: pd.DataFrame) -> pd.DataFrame:
    return (
      filtered.groupby("disc_band", observed=False, as_index=False)
      .agg(avg_sold=("sold_count", "mean"), avg_revenue=("revenue_est", "mean"))
      .dropna(subset=["disc_band"])
      .sort_values("disc_band")
    )

  def rating_group_summary(self, filtered: pd.DataFrame) -> pd.DataFrame:
    return filtered.groupby("rating_group", as_index=False).agg(
      avg_sold=("sold_count", "mean"),
      avg_revenue=("revenue_est", "mean"),
      product_count=("product_id", "nunique"),
    )

  def publisher_group_summary(self, filtered: pd.DataFrame) -> pd.DataFrame:
    return filtered.groupby("publisher_group", as_index=False).agg(
      avg_sold=("sold_count", "mean"),
      avg_revenue=("revenue_est", "mean"),
      product_count=("product_id", "nunique"),
    )

  def numeric_correlation(self, filtered: pd.DataFrame) -> pd.DataFrame:
    frame = filtered[["sold_count", "price", "discount_percent", "rating", "review_count"]].copy()
    if len(frame) < 5:
      return pd.DataFrame()
    return frame.corr(numeric_only=True)