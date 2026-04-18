from __future__ import annotations

from dataclasses import dataclass, field
from html import unescape
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd


def _clean_text(value: Any) -> str:
  if pd.isna(value):
    return ""
  text = unescape(str(value))
  text = re.sub(r"<[^>]+>", " ", text)
  text = re.sub(r"\s+", " ", text)
  return text.strip()


@dataclass
class InfoReviewRetriever:
  product_csv_candidates: tuple[str, ...] = (
    "data/processed/products_clean.csv",
    "data/processed/products_final.csv",
    "data/processed/products.csv",
    "data/raw/products.csv",
  )
  review_csv_candidates: tuple[str, ...] = (
    "data/processed/reviews_clean.csv",
    "data/processed/reviews.csv",
    "data/raw/reviews.csv",
  )
  positive_review_patterns: dict[str, str] = field(
    default_factory=lambda: {
      "noi_dung_hay": r"nội dung hay|hay lắm|hay!|rất hay|cuốn hút|lôi cuốn|hấp dẫn",
      "giao_nhanh": r"giao nhanh|giao hàng nhanh|ship nhanh|nhận nhanh",
      "dep": r"\bđẹp\b|đẹp mắt|xinh|trình bày đẹp|bìa đẹp|sách đẹp",
      "dang_mua": r"đáng mua|nên mua|đáng tiền|rất đáng",
      "sach_moi": r"sách mới|mới tinh|mới nguyên|còn seal|nguyên seal",
      "chat_luong_tot": r"chất lượng tốt|chất lượng ổn|giấy đẹp|in đẹp|đóng gói kỹ|chính hãng",
    }
  )

  def load_data(
    self,
    products_path: str | None = None,
    reviews_path: str | None = None,
  ) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    product_source = self._resolve_source_path(
      provided_path=products_path,
      candidates=self.product_csv_candidates,
      missing_message="Không tìm thấy file dữ liệu sản phẩm.",
    )
    review_source = self._resolve_source_path(
      provided_path=reviews_path,
      candidates=self.review_csv_candidates,
      missing_message="Không tìm thấy file dữ liệu review.",
    )
    return (
      pd.read_csv(product_source),
      pd.read_csv(review_source),
      product_source.as_posix(),
      review_source.as_posix(),
    )

  def _resolve_source_path(
    self,
    provided_path: str | None,
    candidates: tuple[str, ...],
    missing_message: str,
  ) -> Path:
    if provided_path is not None and provided_path.strip():
      custom_path = Path(provided_path)
      if custom_path.exists():
        return custom_path
      raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {provided_path}")

    for candidate in candidates:
      candidate_path = Path(candidate)
      if candidate_path.exists():
        return candidate_path

    raise FileNotFoundError(missing_message)

  def prepare_dataset(
    self,
    products_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
  ) -> pd.DataFrame:
    products = products_df.copy()
    reviews = reviews_df.copy()

    products.columns = self._normalize_columns(products.columns)
    reviews.columns = self._normalize_columns(reviews.columns)

    required_product_cols = ["product_id", "title", "sold_count"]
    missing_product_cols = [col for col in required_product_cols if col not in products.columns]
    if missing_product_cols:
      raise ValueError(
        f"Thiếu cột bắt buộc trong dữ liệu sản phẩm: {missing_product_cols}."
      )

    required_review_cols = ["product_id"]
    missing_review_cols = [col for col in required_review_cols if col not in reviews.columns]
    if missing_review_cols:
      raise ValueError(
        f"Thiếu cột bắt buộc trong dữ liệu review: {missing_review_cols}."
      )

    products = self._prepare_products(products)
    reviews_agg = self._prepare_reviews(reviews)
    merged = products.merge(reviews_agg, on="product_id", how="left")

    numeric_defaults = {
      "review_total": 0,
      "review_text_count": 0,
      "review_avg_rating": np.nan,
      "positive_keyword_review_count": 0,
      "positive_keyword_total_hits": 0,
      "positive_keyword_review_ratio": 0.0,
    }
    for col, default_value in numeric_defaults.items():
      if col not in merged.columns:
        merged[col] = default_value
      merged[col] = merged[col].fillna(default_value)

    for keyword_name in self.positive_review_patterns:
      keyword_col = f"review_kw_{keyword_name}"
      if keyword_col not in merged.columns:
        merged[keyword_col] = 0
      merged[keyword_col] = merged[keyword_col].fillna(0).astype(int)
      merged[f"signal_{keyword_name}"] = merged[keyword_col] > 0

    merged["review_avg_rating_combined"] = (
      merged["review_avg_rating"]
      .where(merged["review_avg_rating"].notna(), merged["rating"])
      .fillna(0)
      .clip(lower=0, upper=5)
    )

    rating_norm = merged["review_avg_rating_combined"] / 5
    keyword_ratio = merged["positive_keyword_review_ratio"].clip(lower=0, upper=1)
    text_coverage = np.where(
      merged["review_total"] > 0,
      merged["review_text_count"] / merged["review_total"].replace(0, np.nan),
      0,
    )
    merged["review_positive_score"] = (
      0.55 * rating_norm.fillna(0)
      + 0.30 * keyword_ratio.fillna(0)
      + 0.15 * pd.Series(text_coverage, index=merged.index).fillna(0).clip(0, 1)
    )
    merged["signal_high_rating"] = merged["review_avg_rating_combined"] >= 4.5

    positive_ratio_non_zero = merged.loc[
      merged["positive_keyword_review_ratio"] > 0,
      "positive_keyword_review_ratio",
    ]
    dynamic_positive_ratio = (
      float(positive_ratio_non_zero.median()) if not positive_ratio_non_zero.empty else 0.25
    )
    merged["signal_positive_content"] = (
      (merged["review_text_count"] > 0)
      & (merged["positive_keyword_review_ratio"] >= max(0.25, dynamic_positive_ratio))
    )

    return merged

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

  def _prepare_products(self, products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()

    default_string_columns = [
      "title",
      "author",
      "publisher",
      "description",
      "category",
    ]
    for col in default_string_columns:
      if col not in df.columns:
        df[col] = ""

    numeric_columns = {
      "product_id": 0,
      "price": 0.0,
      "sold_count": 0.0,
      "rating": np.nan,
      "review_count": 0.0,
      "publish_year": np.nan,
      "page_count": np.nan,
    }
    for col, default_value in numeric_columns.items():
      if col not in df.columns:
        df[col] = default_value
      df[col] = pd.to_numeric(df[col], errors="coerce")

    df["product_id"] = df["product_id"].fillna(0).astype("int64")
    df["price"] = df["price"].fillna(0).clip(lower=0)
    df["sold_count"] = df["sold_count"].fillna(0).clip(lower=0)
    df["rating"] = df["rating"].clip(lower=0, upper=5)
    df["review_count"] = df["review_count"].fillna(0).clip(lower=0)
    df["revenue_estimated"] = df["sold_count"] * df["price"]

    for col in default_string_columns:
      df[col] = df[col].astype("string").fillna("").str.strip()

    df["category"] = df["category"].where(df["category"] != "", "Không phân loại")
    df["description_clean"] = df["description"].map(_clean_text)
    df["title_clean"] = df["title"].map(_clean_text)
    df["combined_text"] = (
      df["title_clean"].str.lower().fillna("")
      + " "
      + df["description_clean"].str.lower().fillna("")
    ).str.strip()

    current_year = pd.Timestamp.now().year + 1
    df["info_has_author"] = df["author"].ne("")
    df["info_has_publisher"] = df["publisher"].ne("")
    df["info_has_publish_year"] = df["publish_year"].between(1900, current_year, inclusive="both")
    df["info_has_page_count"] = df["page_count"].fillna(0) > 0
    df["info_has_description"] = df["description_clean"].str.len().fillna(0) >= 120
    df["info_has_cover_type"] = df["combined_text"].str.contains(
      r"bìa mềm|bìa cứng|hardcover|paperback|hardback",
      regex=True,
      na=False,
    )

    info_cols = self.get_info_feature_columns()
    df["info_completeness_score"] = df[info_cols].sum(axis=1)
    df["info_completeness_ratio"] = df["info_completeness_score"] / len(info_cols)

    return df

  def _prepare_reviews(self, reviews: pd.DataFrame) -> pd.DataFrame:
    df = reviews.copy()
    if "review_id" not in df.columns:
      df["review_id"] = np.arange(1, len(df) + 1)
    if "rating_score" not in df.columns:
      df["rating_score"] = np.nan
    if "content" not in df.columns:
      df["content"] = ""

    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce")
    df = df.dropna(subset=["product_id"]).copy()
    df["product_id"] = df["product_id"].astype("int64")
    df["review_id"] = pd.to_numeric(df["review_id"], errors="coerce")
    df["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce").clip(lower=0, upper=5)
    df["content_clean"] = df["content"].map(_clean_text).astype("string")
    df["content_norm"] = df["content_clean"].str.lower().fillna("")
    df["has_text_content"] = df["content_norm"].str.len() > 0

    keyword_cols: list[str] = []
    for keyword_name, pattern in self.positive_review_patterns.items():
      keyword_col = f"review_kw_{keyword_name}"
      keyword_cols.append(keyword_col)
      df[keyword_col] = df["content_norm"].str.contains(pattern, regex=True, na=False)

    df["positive_keyword_review"] = df[keyword_cols].any(axis=1)
    df["positive_keyword_hit_count"] = df[keyword_cols].sum(axis=1)

    agg_dict: dict[str, Any] = {
      "review_id": "count",
      "rating_score": "mean",
      "has_text_content": "sum",
      "positive_keyword_review": "sum",
      "positive_keyword_hit_count": "sum",
    }
    for keyword_col in keyword_cols:
      agg_dict[keyword_col] = "sum"

    reviews_agg = (
      df.groupby("product_id", as_index=False)
      .agg(agg_dict)
      .rename(columns={
        "review_id": "review_total",
        "rating_score": "review_avg_rating",
        "has_text_content": "review_text_count",
        "positive_keyword_review": "positive_keyword_review_count",
        "positive_keyword_hit_count": "positive_keyword_total_hits",
      })
    )

    reviews_agg["positive_keyword_review_ratio"] = np.where(
      reviews_agg["review_text_count"] > 0,
      reviews_agg["positive_keyword_review_count"] / reviews_agg["review_text_count"],
      0,
    )

    return reviews_agg

  def get_info_feature_columns(self) -> list[str]:
    return [
      "info_has_description",
      "info_has_author",
      "info_has_publisher",
      "info_has_publish_year",
      "info_has_page_count",
      "info_has_cover_type",
    ]

  def get_info_feature_label_map(self) -> dict[str, str]:
    return {
      "info_has_description": "Mô tả nội dung đủ dài",
      "info_has_author": "Có thông tin tác giả",
      "info_has_publisher": "Có thông tin nhà xuất bản",
      "info_has_publish_year": "Có năm xuất bản hợp lệ",
      "info_has_page_count": "Có số trang",
      "info_has_cover_type": "Có thông tin hình thức bìa",
    }

  def get_review_signal_columns(self) -> list[str]:
    return [
      "signal_high_rating",
      "signal_positive_content",
      "signal_noi_dung_hay",
      "signal_dep",
      "signal_chat_luong_tot",
      "signal_giao_nhanh",
      "signal_dang_mua",
      "signal_sach_moi",
    ]

  def get_review_signal_label_map(self) -> dict[str, str]:
    return {
      "signal_high_rating": "Điểm rating trung bình >= 4.5",
      "signal_positive_content": "Tỷ lệ review tích cực cao",
      "signal_noi_dung_hay": "Có nhắc 'nội dung hay' / hấp dẫn",
      "signal_dep": "Có nhắc sách đẹp / trình bày đẹp",
      "signal_chat_luong_tot": "Có nhắc chất lượng tốt / đóng gói tốt",
      "signal_giao_nhanh": "Có nhắc giao nhanh",
      "signal_dang_mua": "Có nhắc đáng mua / đáng tiền",
      "signal_sach_moi": "Có nhắc sách mới / còn seal",
    }

  def compute_information_completeness_impact(
    self,
    df: pd.DataFrame,
    metric_col: str = "sold_count",
    score_threshold: int = 4,
    min_feature_support: int = 20,
  ) -> dict[str, Any]:
    if metric_col not in df.columns:
      raise ValueError(f"Không tìm thấy cột chỉ số '{metric_col}'.")

    info_cols = self.get_info_feature_columns()
    labels = self.get_info_feature_label_map()

    mask_high = df["info_completeness_score"] >= score_threshold
    mask_low = ~mask_high

    avg_high = float(df.loc[mask_high, metric_col].mean()) if mask_high.any() else np.nan
    avg_low = float(df.loc[mask_low, metric_col].mean()) if mask_low.any() else np.nan
    median_high = float(df.loc[mask_high, metric_col].median()) if mask_high.any() else np.nan
    median_low = float(df.loc[mask_low, metric_col].median()) if mask_low.any() else np.nan
    uplift_pct = ((avg_high - avg_low) / avg_low * 100) if pd.notna(avg_low) and avg_low > 0 else np.nan

    overall_summary = pd.DataFrame([
      {
        "group": f"Thông tin đầy đủ hơn (score >= {score_threshold})",
        "avg_metric": avg_high,
        "median_metric": median_high,
        "count": int(mask_high.sum()),
      },
      {
        "group": f"Nhóm còn lại (score < {score_threshold})",
        "avg_metric": avg_low,
        "median_metric": median_low,
        "count": int(mask_low.sum()),
      },
    ])

    score_summary = (
      df.groupby("info_completeness_score", as_index=False)
      .agg(
        product_count=("product_id", "count"),
        avg_metric=(metric_col, "mean"),
        median_metric=(metric_col, "median"),
      )
      .sort_values("info_completeness_score")
      .reset_index(drop=True)
    )

    feature_records: list[dict[str, Any]] = []
    for col in info_cols:
      mask_feature = df[col].astype(bool)
      mask_without = ~mask_feature
      count_with = int(mask_feature.sum())
      count_without = int(mask_without.sum())
      avg_with = float(df.loc[mask_feature, metric_col].mean()) if mask_feature.any() else np.nan
      avg_without = float(df.loc[mask_without, metric_col].mean()) if mask_without.any() else np.nan
      uplift = (
        ((avg_with - avg_without) / avg_without) * 100
        if pd.notna(avg_without) and avg_without > 0
        else np.nan
      )
      feature_records.append({
        "feature_col": col,
        "feature_label": labels[col],
        "count_with_feature": count_with,
        "count_without_feature": count_without,
        "support_ratio": float(mask_feature.mean()) if len(df) else 0.0,
        "avg_with_feature": avg_with,
        "avg_without_feature": avg_without,
        "uplift_pct": uplift,
        "is_informative": bool(count_with > 0 and count_without > 0),
        "sufficient_support": bool(
          count_with >= min_feature_support
          and count_without >= min_feature_support
        ),
        "target_20pct_met": bool(pd.notna(uplift) and uplift >= 20),
      })

    feature_impact = pd.DataFrame(feature_records).sort_values(
      ["uplift_pct", "count_with_feature"],
      ascending=[False, False],
      na_position="last",
    ).reset_index(drop=True)

    top_components = feature_impact[
      feature_impact["is_informative"]
      & feature_impact["sufficient_support"]
      & feature_impact["uplift_pct"].notna()
    ].head(3).copy()

    feature_impact = feature_impact[
      feature_impact["is_informative"]
    ].reset_index(drop=True)

    return {
      "overall_summary": overall_summary,
      "score_summary": score_summary,
      "feature_impact": feature_impact,
      "top_components": top_components,
      "uplift_pct": uplift_pct,
      "target_20pct_met": bool(pd.notna(uplift_pct) and uplift_pct >= 20),
      "high_info_count": int(mask_high.sum()),
      "low_info_count": int(mask_low.sum()),
      "score_threshold": score_threshold,
    }

  def compute_review_impact(
    self,
    df: pd.DataFrame,
    metric_col: str = "sold_count",
    min_feature_support: int = 20,
  ) -> dict[str, Any]:
    if metric_col not in df.columns:
      raise ValueError(f"Không tìm thấy cột chỉ số '{metric_col}'.")

    signal_cols = self.get_review_signal_columns()
    label_map = self.get_review_signal_label_map()

    eligible_review_mask = (df["review_total"] > 0) | (df["review_count"] > 0)
    review_scores = df.loc[eligible_review_mask, "review_positive_score"]
    positivity_threshold = float(review_scores.median()) if not review_scores.empty else 0.65
    positive_mask = eligible_review_mask & (df["review_positive_score"] >= positivity_threshold)
    other_mask = ~positive_mask

    avg_positive = float(df.loc[positive_mask, metric_col].mean()) if positive_mask.any() else np.nan
    avg_other = float(df.loc[other_mask, metric_col].mean()) if other_mask.any() else np.nan
    median_positive = float(df.loc[positive_mask, metric_col].median()) if positive_mask.any() else np.nan
    median_other = float(df.loc[other_mask, metric_col].median()) if other_mask.any() else np.nan
    uplift_pct = (
      ((avg_positive - avg_other) / avg_other) * 100
      if pd.notna(avg_other) and avg_other > 0
      else np.nan
    )

    overall_summary = pd.DataFrame([
      {
        "group": "Review tích cực hơn",
        "avg_metric": avg_positive,
        "median_metric": median_positive,
        "count": int(positive_mask.sum()),
      },
      {
        "group": "Nhóm còn lại",
        "avg_metric": avg_other,
        "median_metric": median_other,
        "count": int(other_mask.sum()),
      },
    ])

    score_summary = (
      df.assign(
        review_score_bucket=pd.cut(
          df["review_positive_score"],
          bins=[-0.001, 0.4, 0.55, 0.7, 1.0],
          labels=["Rất thấp", "Trung bình", "Tích cực", "Rất tích cực"],
        )
      )
      .groupby("review_score_bucket", observed=False, as_index=False)
      .agg(
        product_count=("product_id", "count"),
        avg_metric=(metric_col, "mean"),
      )
    )

    signal_records: list[dict[str, Any]] = []
    for col in signal_cols:
      if col not in df.columns:
        continue
      mask_signal = df[col].astype(bool)
      mask_without = ~mask_signal
      count_with = int(mask_signal.sum())
      count_without = int(mask_without.sum())
      avg_with = float(df.loc[mask_signal, metric_col].mean()) if mask_signal.any() else np.nan
      avg_without = float(df.loc[mask_without, metric_col].mean()) if mask_without.any() else np.nan
      uplift = (
        ((avg_with - avg_without) / avg_without) * 100
        if pd.notna(avg_without) and avg_without > 0
        else np.nan
      )
      signal_records.append({
        "signal_col": col,
        "signal_label": label_map[col],
        "count_with_signal": count_with,
        "count_without_signal": count_without,
        "support_ratio": float(mask_signal.mean()) if len(df) else 0.0,
        "avg_with_signal": avg_with,
        "avg_without_signal": avg_without,
        "uplift_pct": uplift,
        "is_informative": bool(count_with > 0 and count_without > 0),
        "sufficient_support": bool(
          count_with >= min_feature_support
          and count_without >= min_feature_support
        ),
        "target_15pct_met": bool(pd.notna(uplift) and uplift >= 15),
      })

    signal_impact = pd.DataFrame(signal_records).sort_values(
      ["uplift_pct", "count_with_signal"],
      ascending=[False, False],
      na_position="last",
    ).reset_index(drop=True)

    top_signals = signal_impact[
      signal_impact["is_informative"]
      & signal_impact["sufficient_support"]
      & signal_impact["uplift_pct"].notna()
    ].head(2).copy()

    signal_impact = signal_impact[
      signal_impact["is_informative"]
    ].reset_index(drop=True)

    return {
      "overall_summary": overall_summary,
      "score_summary": score_summary,
      "signal_impact": signal_impact,
      "top_signals": top_signals,
      "uplift_pct": uplift_pct,
      "target_15pct_met": bool(pd.notna(uplift_pct) and uplift_pct >= 15),
      "positive_count": int(positive_mask.sum()),
      "other_count": int(other_mask.sum()),
      "positivity_threshold": positivity_threshold,
    }
