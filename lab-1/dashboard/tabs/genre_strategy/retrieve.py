from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.utils import load_data, to_bool
from dashboard.config import PRODUCTS_PATH, AUTHORS_PATH

INVALID_GENRES = {"root", "other"}


@dataclass
class GenreStrategyRetriever:
  products_path: Path = PRODUCTS_PATH
  authors_path: Path = AUTHORS_PATH

  def load_products(self) -> pd.DataFrame:
    products = load_data(str(self.products_path))
    return self._prepare_products(products)

  def load_authors(self) -> pd.DataFrame:
    return load_data(str(self.authors_path))

  def _prepare_products(self, products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()

    numeric_columns = [
      "price",
      "original_price",
      "discount_percent",
      "sold_count",
      "ranking",
      "rating",
      "review_count",
      "publish_year",
      "page_count",
    ]
    for column in numeric_columns:
      if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["price", "sold_count", "discount_percent", "review_count", "rating"]:
      if column in df.columns:
        df[column] = df[column].fillna(0)

    boolean_columns = ["is_bestseller", "is_top100", "is_combo", "has_gift"]
    for column in boolean_columns:
      if column in df.columns:
        df[column] = to_bool(df[column])
      else:
        df[column] = False

    category_source = df["category_binned"].fillna("")
    if "category" in df.columns:
      category_source = category_source.mask(category_source.eq(""), df["category"].fillna(""))
    df["genre"] = category_source.replace("", "Không rõ").astype(str).str.strip()
    normalized_genre = df["genre"].str.casefold()
    df = df[~normalized_genre.isin(INVALID_GENRES)].copy()

    df["publisher_clean"] = df.get("publisher", pd.Series(index=df.index, dtype="object")).fillna("Không rõ")
    df["publisher_clean"] = df["publisher_clean"].replace("", "Không rõ")
    df["author_clean"] = df.get("author", pd.Series(index=df.index, dtype="object")).fillna("Không rõ")
    df["author_clean"] = df["author_clean"].replace("", "Không rõ")

    df["revenue_est"] = df["price"] * df["sold_count"]
    return df

  def with_publisher_group(self, products: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = products.copy()
    publisher_sales = (
      df.groupby("publisher_clean", as_index=False)["sold_count"]
      .sum()
      .sort_values("sold_count", ascending=False)
    )
    top_publishers = publisher_sales["publisher_clean"].head(5).tolist()
    df["publisher_group"] = np.where(df["publisher_clean"].isin(top_publishers), "Top 5 NXB", "Khác")
    return df, top_publishers

  def filter_products(
    self,
    products: pd.DataFrame,
    genres: list[str],
    combo_mode: str,
    price_range: tuple[float, float],
    publisher_mode: str,
  ) -> tuple[pd.DataFrame, list[str]]:
    filtered = products.copy()

    if genres:
      filtered = filtered[filtered["genre"].isin(genres)]

    min_price, max_price = price_range
    filtered = filtered[(filtered["price"] >= min_price) & (filtered["price"] <= max_price)]

    if combo_mode == "Chỉ combo":
      filtered = filtered[filtered["is_combo"]]
    elif combo_mode == "Không combo":
      filtered = filtered[~filtered["is_combo"]]

    filtered, top_publishers = self.with_publisher_group(filtered)

    if publisher_mode == "Top 5 NXB":
      filtered = filtered[filtered["publisher_group"] == "Top 5 NXB"]
    elif publisher_mode == "Khác":
      filtered = filtered[filtered["publisher_group"] == "Khác"]

    return filtered, top_publishers

  def compute_kpis(self, filtered: pd.DataFrame) -> dict[str, float]:
    if filtered.empty:
      return {
        "total_products": 0,
        "total_sold": 0,
        "total_revenue": 0.0,
        "combo_ratio": 0.0,
      }
    return {
      "total_products": int(filtered["product_id"].nunique()),
      "total_sold": float(filtered["sold_count"].sum()),
      "total_revenue": float(filtered["revenue_est"].sum()),
      "combo_ratio": float(filtered["is_combo"].mean() * 100),
    }

  def genre_summary(self, filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
      return pd.DataFrame()
    summary = (
      filtered.groupby("genre", as_index=False)
      .agg(
        product_count=("product_id", "nunique"),
        total_sold=("sold_count", "sum"),
        total_revenue=("revenue_est", "sum"),
        avg_sold=("sold_count", "mean"),
      )
      .sort_values("total_sold", ascending=False)
    )
    return summary

  def publisher_summary(self, filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
      return pd.DataFrame()
    summary = (
      filtered.groupby("publisher_group", as_index=False)
      .agg(
        product_count=("product_id", "nunique"),
        avg_sold=("sold_count", "mean"),
        avg_revenue=("revenue_est", "mean"),
      )
      .sort_values("publisher_group")
    )
    return summary

  def combo_uplift_by_genre(self, filtered: pd.DataFrame, min_products: int = 8) -> pd.DataFrame:
    if filtered.empty:
      return pd.DataFrame()

    grouped = (
      filtered.groupby(["genre", "is_combo"], as_index=False)
      .agg(product_count=("product_id", "nunique"), avg_sold=("sold_count", "mean"))
    )
    enough_data = grouped[grouped["product_count"] >= min_products]
    if enough_data.empty:
      return pd.DataFrame()

    pivot = enough_data.pivot_table(index="genre", columns="is_combo", values="avg_sold")
    pivot.columns = [f"combo_{int(column)}" for column in pivot.columns]
    pivot = pivot.reset_index()

    if "combo_0" not in pivot.columns or "combo_1" not in pivot.columns:
      return pd.DataFrame()

    pivot = pivot[pivot["combo_0"] > 0]
    if pivot.empty:
      return pd.DataFrame()

    pivot["uplift_percent"] = (pivot["combo_1"] - pivot["combo_0"]) / pivot["combo_0"] * 100
    return pivot.sort_values("uplift_percent", ascending=False)

  def combo_pivot_table(self, filtered: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if filtered.empty:
      return pd.DataFrame()
    genre_rank = (
      filtered.groupby("genre", as_index=False)["sold_count"]
      .sum()
      .sort_values("sold_count", ascending=False)
      .head(top_n)
    )
    selected_genres = genre_rank["genre"].tolist()
    narrowed = filtered[filtered["genre"].isin(selected_genres)]
    pivot = narrowed.pivot_table(
      index="genre",
      columns="is_combo",
      values="sold_count",
      aggfunc="mean",
      fill_value=0,
    )
    pivot = pivot.rename(columns={False: "Bán lẻ", True: "Combo"})
    return pivot

  def build_ml_dataset(self, products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    needed = [
      "sold_count",
      "price",
      "discount_percent",
      "rating",
      "review_count",
      "page_count",
      "publish_year",
      "genre",
      "disc_band",
      "publisher_group",
      "is_combo",
      "is_bestseller",
      "is_top100",
      "has_gift",
    ]
    for column in needed:
      if column not in df.columns:
        df[column] = np.nan

    for column in ["genre", "disc_band", "publisher_group"]:
      df[column] = df[column].fillna("Không rõ").astype(str)

    for column in ["is_combo", "is_bestseller", "is_top100", "has_gift"]:
      df[column] = to_bool(df[column]).astype(int)

    numeric_columns = ["sold_count", "price", "discount_percent", "rating", "review_count", "page_count", "publish_year"]
    for column in numeric_columns:
      df[column] = pd.to_numeric(df[column], errors="coerce")

    prepared = df[needed].dropna(subset=["sold_count", "price", "discount_percent", "rating", "review_count"])
    prepared = prepared[prepared["sold_count"] >= 0]
    return prepared

  def smart_snapshot(self, filtered: pd.DataFrame) -> dict[str, float]:
    snapshot = {"publisher_gap_percent": np.nan, "combo_above_20_count": 0, "top_genre": "Không xác định"}
    if filtered.empty:
      return snapshot

    genre_summary = self.genre_summary(filtered)
    if not genre_summary.empty:
      snapshot["top_genre"] = str(genre_summary.iloc[0]["genre"])

    publisher_summary = self.publisher_summary(filtered)
    if {"Top 5 NXB", "Khác"}.issubset(set(publisher_summary["publisher_group"].tolist())):
      top5 = float(publisher_summary.loc[publisher_summary["publisher_group"] == "Top 5 NXB", "avg_sold"].iloc[0])
      other = float(publisher_summary.loc[publisher_summary["publisher_group"] == "Khác", "avg_sold"].iloc[0])
      if other > 0:
        snapshot["publisher_gap_percent"] = (top5 - other) / other * 100

    uplift = self.combo_uplift_by_genre(filtered)
    if not uplift.empty:
      snapshot["combo_above_20_count"] = int((uplift["uplift_percent"] > 20).sum())

    return snapshot