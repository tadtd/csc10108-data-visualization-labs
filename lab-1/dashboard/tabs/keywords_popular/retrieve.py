from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any
import unicodedata

import numpy as np
import pandas as pd
from dashboard.config import PRODUCTS_PATH
from dashboard.utils import load_data


@dataclass
class KeywordsPopularRetriever:
  products_path: Path = PRODUCTS_PATH
  keyword_patterns: dict[str, str] = field(
    default_factory=lambda: {
      "best_seller": r"best\s?seller|bestseller|bán chạy|ban chay",
      "tai_ban": r"tái bản|tai ban|new edition|phiên bản",
      "combo": r"combo|bộ sách|boxset|trọn bộ",
      "tam_ly": r"tâm lý|tam ly|psychology",
      "ky_nang": r"kỹ năng|ky nang|skills?",
    }
  )
  unknown_author_tokens: tuple[str, ...] = (
    "unknown",
    "na",
    "n/a",
    "none",
    "null",
    "khong ro",
    "chua cap nhat",
    "dang cap nhat",
    "various authors",
    "nhieu tac gia",
  )

  def load_products(self) -> tuple[pd.DataFrame, str]:
    source_path = str(self.products_path)
    df = load_data(source_path)
    return df, source_path

  def prepare_dataset(self, products_df: pd.DataFrame) -> pd.DataFrame:
    df = products_df.copy()
    df.columns = self._normalize_columns(df.columns)

    required_columns = ["title", "sold_count"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
      raise ValueError(
        f"Thiếu cột bắt buộc trong dữ liệu: {missing_columns}. "
        "Cần tối thiểu có 'title' và 'sold_count'."
      )

    if "product_id" not in df.columns:
      df["product_id"] = np.arange(1, len(df) + 1)
    if "price" not in df.columns:
      df["price"] = 0
    if "author" not in df.columns:
      df["author"] = "Unknown"
    if "category" not in df.columns:
      df["category"] = "Không phân loại"
    if "review_count" not in df.columns:
      df["review_count"] = 0

    df["title"] = df["title"].astype("string").fillna("")
    df["title_norm"] = (
      df["title"]
      .str.lower()
      .str.replace(r"\s+", " ", regex=True)
      .str.strip()
    )

    df["sold_count"] = pd.to_numeric(df["sold_count"], errors="coerce").fillna(0).clip(lower=0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).clip(lower=0)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).clip(lower=0)
    df["revenue_estimated"] = df["sold_count"] * df["price"]

    self._build_title_features(df)
    self._build_author_features(df)

    df["title_length"] = df["title_norm"].str.len().fillna(0).astype(int)
    df["word_count"] = df["title_norm"].str.split().str.len().fillna(0).astype(int)

    return df

  @staticmethod
  def _normalize_columns(columns: pd.Index) -> pd.Index:
    return (
      pd.Index(columns)
      .astype(str)
      .str.strip()
      .str.lower()
      .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
      .str.replace(r"_+", "_", regex=True)
      .str.strip("_")
    )

  def _build_title_features(self, df: pd.DataFrame) -> None:
    for keyword_key, pattern in self.keyword_patterns.items():
      df[f"kw_{keyword_key}"] = df["title_norm"].str.contains(pattern, regex=True, na=False)

    df["pt_has_number"] = df["title_norm"].str.contains(r"\d", regex=True, na=False)
    df["pt_has_colon"] = df["title_norm"].str.contains(":", regex=False, na=False)
    df["pt_has_dash"] = df["title_norm"].str.contains(r"[-–—]", regex=True, na=False)
    df["pt_has_listing"] = df["title_norm"].str.contains(
      r"\w\s*(?:\+|/|\||,)\s*\w|\b\d+\.\s*\w",
      regex=True,
      na=False,
    )

  def _build_author_features(self, df: pd.DataFrame) -> None:
    author = df["author"].astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    author = author.where(author != "", "Unknown")

    df["author_clean"] = author
    split_pattern = r";|/|\||\s+và\s+|\s+and\s+|\+|\s*&\s*"
    df["author_primary"] = author.str.split(split_pattern, regex=True).str[0].str.strip()
    df["author_primary"] = df["author_primary"].apply(self._normalize_author_name)
    df["author_is_unknown"] = df["author_primary"].eq("Unknown")

  @staticmethod
  def _normalize_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_text.lower().strip()

  def _normalize_author_name(self, author_name: Any) -> str:
    author_str = str(author_name).strip()
    author_str = re.sub(r"\s+", " ", author_str)
    author_str = author_str.strip(" .,-")

    if not author_str:
      return "Unknown"

    author_ascii = self._normalize_ascii(author_str)
    if author_ascii in self.unknown_author_tokens:
      return "Unknown"

    if author_ascii.startswith("nhieu tac gia"):
      return "Unknown"

    return author_str

  def get_title_feature_columns(self) -> list[str]:
    return [
      *(f"kw_{keyword_key}" for keyword_key in self.keyword_patterns),
      "pt_has_number",
      "pt_has_colon",
      "pt_has_dash",
      "pt_has_listing",
    ]

  def get_feature_label_map(self) -> dict[str, str]:
    return {
      "kw_best_seller": "Keyword: Best seller/Bán chạy",
      "kw_tai_ban": "Keyword: Tái bản/Phiên bản",
      "kw_combo": "Keyword: Combo/Trọn bộ",
      "kw_tam_ly": "Keyword: Tâm lý",
      "kw_ky_nang": "Keyword: Kỹ năng",
      "pt_has_number": "Pattern: Có chữ số",
      "pt_has_colon": "Pattern: Có dấu ':'",
      "pt_has_dash": "Pattern: Có dấu '-'",
      "pt_has_listing": "Pattern: Cấu trúc liệt kê (+, /, |, ,)",
      "title_length": "Độ dài tiêu đề",
      "word_count": "Số từ trong tiêu đề",
    }

  def compute_feature_impact(
    self,
    df: pd.DataFrame,
    metric_col: str = "sold_count",
    min_feature_support: int = 30,
  ) -> pd.DataFrame:
    if metric_col not in df.columns:
      raise ValueError(f"Không tìm thấy cột chỉ số '{metric_col}' trong dữ liệu.")

    label_map = self.get_feature_label_map()
    records: list[dict[str, Any]] = []

    for feature_col in self.get_title_feature_columns():
      if feature_col not in df.columns:
        continue

      mask = df[feature_col].fillna(False).astype(bool)
      count_with_feature = int(mask.sum())
      count_without_feature = int((~mask).sum())

      metric_with = float(df.loc[mask, metric_col].mean()) if mask.any() else 0.0
      metric_without = float(df.loc[~mask, metric_col].mean()) if (~mask).any() else 0.0

      sufficient_support = count_with_feature >= min_feature_support and count_without_feature >= min_feature_support

      uplift_pct = np.nan
      if metric_without > 0:
        uplift_pct = ((metric_with - metric_without) / metric_without) * 100

      records.append(
        {
          "feature_col": feature_col,
          "feature_label": label_map.get(feature_col, feature_col),
          "feature_type": "Keyword" if feature_col.startswith("kw_") else "Pattern",
          "count_with_feature": count_with_feature,
          "count_without_feature": count_without_feature,
          "support_ratio": count_with_feature / len(df) if len(df) > 0 else np.nan,
          "sufficient_support": sufficient_support,
          "avg_with_feature": metric_with,
          "avg_without_feature": metric_without,
          "uplift_pct": uplift_pct,
          "target_20pct_met": bool(pd.notna(uplift_pct) and uplift_pct >= 20 and sufficient_support),
          "impact_score": float(uplift_pct * np.log1p(count_with_feature)) if pd.notna(uplift_pct) else np.nan,
        }
      )

    feature_impact_df = pd.DataFrame(records)
    if feature_impact_df.empty:
      return feature_impact_df

    return feature_impact_df.sort_values(
      by=["target_20pct_met", "impact_score", "uplift_pct", "count_with_feature"],
      ascending=[False, False, False, False],
      na_position="last",
    ).reset_index(drop=True)

  def compute_author_popularity(
    self,
    df: pd.DataFrame,
    top_n: int = 10,
    metric_col: str = "sold_count",
  ) -> dict[str, Any]:
    if metric_col not in df.columns:
      raise ValueError(f"Không tìm thấy cột chỉ số '{metric_col}' trong dữ liệu.")

    analysis_df = df[df["author_primary"] != "Unknown"].copy()
    if analysis_df.empty:
      return {
        "author_stats": pd.DataFrame(),
        "group_summary": pd.DataFrame(),
        "top_authors": set(),
        "uplift_pct": np.nan,
        "target_25pct_met": False,
        "analysis_book_count": 0,
      }

    product_count_col = "product_id" if "product_id" in analysis_df.columns else "title"
    author_stats = (
      analysis_df.groupby("author_primary", as_index=False)
      .agg(
        total_books=(product_count_col, "count"),
        total_sales=("sold_count", "sum"),
        avg_sales_per_book=("sold_count", "mean"),
        avg_metric=(metric_col, "mean"),
      )
      .rename(columns={"author_primary": "author_name"})
    )

    author_stats = author_stats[author_stats["author_name"].astype(str).str.strip() != ""]

    if author_stats.empty:
      return {
        "author_stats": author_stats,
        "group_summary": pd.DataFrame(),
        "top_authors": set(),
        "uplift_pct": np.nan,
        "target_25pct_met": False,
      }

    author_stats["popularity_score"] = (
      self._min_max_scale(author_stats["total_books"]) * 0.5
      + self._min_max_scale(author_stats["total_sales"]) * 0.5
    )

    author_stats = author_stats.sort_values(
      by=["popularity_score", "total_sales", "total_books"],
      ascending=[False, False, False],
    ).reset_index(drop=True)
    author_stats["rank"] = np.arange(1, len(author_stats) + 1)

    top_authors = set(author_stats.head(top_n)["author_name"].tolist())

    df_with_group = analysis_df.copy()
    group_name = f"Top {top_n} tác giả phổ biến"
    df_with_group["author_group"] = np.where(
      df_with_group["author_primary"].isin(top_authors),
      group_name,
      "Nhóm tác giả còn lại",
    )

    group_summary = (
      df_with_group.groupby("author_group", as_index=False)
      .agg(
        book_count=(product_count_col, "count"),
        avg_metric=(metric_col, "mean"),
        median_metric=(metric_col, "median"),
      )
      .sort_values(by="avg_metric", ascending=False)
      .reset_index(drop=True)
    )

    top_avg = self._extract_group_average(group_summary, group_name)
    other_avg = self._extract_group_average(group_summary, "Nhóm tác giả còn lại")
    uplift_pct = np.nan
    if pd.notna(other_avg) and other_avg > 0:
      uplift_pct = ((top_avg - other_avg) / other_avg) * 100

    return {
      "author_stats": author_stats,
      "group_summary": group_summary,
      "top_authors": top_authors,
      "uplift_pct": uplift_pct,
      "target_25pct_met": bool(pd.notna(uplift_pct) and uplift_pct >= 25),
      "analysis_book_count": int(len(analysis_df)),
    }

  @staticmethod
  def _extract_group_average(group_summary: pd.DataFrame, group_name: str) -> float:
    row = group_summary[group_summary["author_group"] == group_name]
    if row.empty:
      return np.nan
    return float(row.iloc[0]["avg_metric"])

  @staticmethod
  def _min_max_scale(series: pd.Series) -> pd.Series:
    min_val = float(series.min())
    max_val = float(series.max())
    if np.isclose(max_val, min_val):
      return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)

  def run_ml_feature_importance(
    self,
    df: pd.DataFrame,
    target_col: str = "sold_count",
  ) -> dict[str, Any]:
    try:
      from sklearn.ensemble import RandomForestRegressor
      from sklearn.metrics import mean_absolute_error, r2_score
      from sklearn.model_selection import train_test_split
    except ImportError:
      return {
        "available": False,
        "message": "Không thể chạy ML vì thiếu scikit-learn trong môi trường.",
      }

    if target_col not in df.columns:
      return {
        "available": False,
        "message": f"Không tìm thấy cột mục tiêu '{target_col}' để chạy ML.",
      }

    model_feature_columns = [*self.get_title_feature_columns(), "title_length", "word_count"]
    model_df = df[model_feature_columns + [target_col]].copy()

    for feature_col in self.get_title_feature_columns():
      model_df[feature_col] = model_df[feature_col].astype(int)

    model_df["title_length"] = pd.to_numeric(model_df["title_length"], errors="coerce").fillna(0)
    model_df["word_count"] = pd.to_numeric(model_df["word_count"], errors="coerce").fillna(0)
    model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
    model_df = model_df.dropna(subset=[target_col])

    if len(model_df) < 120:
      return {
        "available": False,
        "message": "Dữ liệu sau lọc còn quá ít (<120 dòng), chưa đủ ổn định để chạy ML.",
      }

    if model_df[target_col].nunique() < 2:
      return {
        "available": False,
        "message": "Biến mục tiêu không đủ biến thiên, không thể huấn luyện mô hình.",
      }

    x_data = model_df[model_feature_columns]
    y_data = model_df[target_col].clip(lower=0)

    x_train, x_test, y_train, y_test = train_test_split(
      x_data,
      y_data,
      test_size=0.2,
      random_state=42,
    )

    y_train_log = np.log1p(y_train)
    model = RandomForestRegressor(
      n_estimators=300,
      max_depth=10,
      min_samples_leaf=3,
      random_state=42,
      n_jobs=-1,
    )
    model.fit(x_train, y_train_log)

    pred_log = model.predict(x_test)
    pred = np.expm1(pred_log)

    mae = mean_absolute_error(y_test, pred)
    r2_log = r2_score(np.log1p(y_test), pred_log)

    label_map = self.get_feature_label_map()
    importance_df = pd.DataFrame(
      {
        "feature_col": model_feature_columns,
        "importance": model.feature_importances_,
      }
    )
    importance_df["feature_label"] = importance_df["feature_col"].map(label_map).fillna(importance_df["feature_col"])
    importance_df = importance_df.sort_values(by="importance", ascending=False).reset_index(drop=True)

    return {
      "available": True,
      "sample_size": int(len(model_df)),
      "mae": float(mae),
      "r2_log": float(r2_log),
      "importance_df": importance_df,
    }