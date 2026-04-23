from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dashboard.config import PRODUCTS_PATH
from dashboard.utils import load_data


@dataclass
class FomoGiftRetriever:
  products_path: Path = PRODUCTS_PATH
  gift_keyword_patterns: dict[str, str] = field(
    default_factory=lambda: {
      "ban_dac_biet": r"bản đặc biệt|phiên bản đặc biệt|special edition|limited edition|bản giới hạn",
      "tang_kem_bookmark": r"tặng kèm|tặng bookmark|kèm bookmark|kèm túi|kèm bao|tang kem",
      "postcard": r"postcard|post card|thiệp|card nghệ thuật",
      "kem_cd": r"kèm cd|kèm dvd|kèm usb|kem cd|kem dvd|kèm đĩa",
      "hop_qua": r"hộp quà|gift set|gift box|túi quà|bộ quà|hop qua",
      "bia_cung": r"bìa cứng|hardcover|bìa gỗ|hardback",
    }
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
    if "category" not in df.columns:
      df["category"] = "Không phân loại"
    if "review_count" not in df.columns:
      df["review_count"] = 0
    if "is_top100" not in df.columns:
      df["is_top100"] = False
    if "is_bestseller" not in df.columns:
      df["is_bestseller"] = False
    if "has_gift" not in df.columns:
      df["has_gift"] = False
    if "description" not in df.columns:
      df["description"] = ""

    df["title"] = df["title"].astype("string").fillna("")
    df["description"] = df["description"].astype("string").fillna("")
    df["title_norm"] = df["title"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df["desc_norm"] = df["description"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    df["text_combined"] = df["title_norm"] + " " + df["desc_norm"]

    df["sold_count"] = pd.to_numeric(df["sold_count"], errors="coerce").fillna(0).clip(lower=0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).clip(lower=0)
    df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).clip(lower=0)
    df["revenue_estimated"] = df["sold_count"] * df["price"]

    df["is_top100"] = pd.to_numeric(df["is_top100"], errors="coerce").fillna(0).astype(bool)
    df["is_bestseller"] = pd.to_numeric(df["is_bestseller"], errors="coerce").fillna(0).astype(bool)
    df["has_gift"] = pd.to_numeric(df["has_gift"], errors="coerce").fillna(0).astype(bool)

    self._build_gift_features(df)

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

  def _build_gift_features(self, df: pd.DataFrame) -> None:
    for key, pattern in self.gift_keyword_patterns.items():
      df[f"gift_{key}"] = df["text_combined"].str.contains(
        pattern, regex=True, na=False, case=False
      )
    kw_cols = [f"gift_{k}" for k in self.gift_keyword_patterns]
    df["has_any_gift_keyword"] = df[kw_cols].any(axis=1)
    df["has_gift_combined"] = df["has_gift"] | df["has_any_gift_keyword"]

  def get_gift_feature_columns(self) -> list[str]:
    return [f"gift_{k}" for k in self.gift_keyword_patterns]

  def get_gift_feature_label_map(self) -> dict[str, str]:
    return {
      "gift_ban_dac_biet": "Bản đặc biệt / Special Edition",
      "gift_tang_kem_bookmark": "Tặng kèm / Bookmark",
      "gift_postcard": "Postcard / Thiệp",
      "gift_kem_cd": "Kèm CD / DVD / USB",
      "gift_hop_qua": "Hộp quà / Gift Set",
      "gift_bia_cung": "Bìa cứng / Hardcover",
      "has_gift": "Trường has_gift (dữ liệu crawl)",
      "has_gift_combined": "Kết hợp: has_gift + keyword",
    }

  def compute_top100_impact(
    self,
    df: pd.DataFrame,
    metric_col: str = "sold_count",
    badge_col: str = "is_top100",
    min_category_support: int = 10,
  ) -> dict[str, Any]:
    if metric_col not in df.columns:
      raise ValueError(f"Không tìm thấy cột chỉ số '{metric_col}'.")

    if badge_col not in df.columns or df[badge_col].sum() == 0:
      return {
        "overall_summary": pd.DataFrame(),
        "category_summary": pd.DataFrame(),
        "uplift_pct": np.nan,
        "target_50pct_met": False,
        "top_count": 0,
        "non_top_count": int(len(df)),
        "avg_top": np.nan,
        "avg_non_top": float(df[metric_col].mean()) if not df.empty else 0.0,
      }

    mask_top = df[badge_col].astype(bool)
    top_count = int(mask_top.sum())
    non_top_count = int((~mask_top).sum())

    avg_top = float(df.loc[mask_top, metric_col].mean()) if mask_top.any() else 0.0
    avg_non_top = float(df.loc[~mask_top, metric_col].mean()) if (~mask_top).any() else 0.0
    median_top = float(df.loc[mask_top, metric_col].median()) if mask_top.any() else 0.0
    median_non_top = float(df.loc[~mask_top, metric_col].median()) if (~mask_top).any() else 0.0

    uplift_pct = np.nan
    if avg_non_top > 0:
      uplift_pct = ((avg_top - avg_non_top) / avg_non_top) * 100

    overall_summary = pd.DataFrame([
      {
        "group": f"Có huy hiệu Top (N={top_count:,})",
        "avg_metric": avg_top,
        "median_metric": median_top,
        "count": top_count,
      },
      {
        "group": f"Không có huy hiệu Top (N={non_top_count:,})",
        "avg_metric": avg_non_top,
        "median_metric": median_non_top,
        "count": non_top_count,
      },
    ])

    generic_categories = {
      "root", "sách tiếng việt", "sach tieng viet",
      "sách", "sach", "books", "book", "không phân loại",
    }

    cat_records: list[dict[str, Any]] = []
    for cat, cat_df in df.groupby("category"):
      if str(cat).strip().lower() in generic_categories:
        continue
      m_top = cat_df[badge_col].astype(bool)
      cnt_top = int(m_top.sum())
      cnt_non = int((~m_top).sum())
      if cnt_top < min_category_support or cnt_non < min_category_support:
        continue
      a_top = float(cat_df.loc[m_top, metric_col].mean())
      a_non = float(cat_df.loc[~m_top, metric_col].mean())
      up = ((a_top - a_non) / a_non * 100) if a_non > 0 else np.nan
      cat_records.append({
        "category": str(cat),
        "avg_top": a_top,
        "avg_non_top": a_non,
        "uplift_pct": up,
        "count_top": cnt_top,
        "count_non_top": cnt_non,
        "target_50pct_met": bool(pd.notna(up) and up >= 50),
      })

    category_summary = pd.DataFrame(cat_records)
    if not category_summary.empty:
      category_summary = category_summary.sort_values(
        "uplift_pct", ascending=False
      ).reset_index(drop=True)

    return {
      "overall_summary": overall_summary,
      "category_summary": category_summary,
      "uplift_pct": uplift_pct,
      "target_50pct_met": bool(pd.notna(uplift_pct) and uplift_pct >= 50),
      "top_count": top_count,
      "non_top_count": non_top_count,
      "avg_top": avg_top,
      "avg_non_top": avg_non_top,
    }

  def compute_gift_impact(
    self,
    df: pd.DataFrame,
    metric_col: str = "sold_count",
    min_feature_support: int = 20,
  ) -> pd.DataFrame:
    if metric_col not in df.columns:
      raise ValueError(f"Không tìm thấy cột chỉ số '{metric_col}'.")

    label_map = self.get_gift_feature_label_map()
    feature_cols = self.get_gift_feature_columns() + ["has_gift", "has_gift_combined"]
    records: list[dict[str, Any]] = []

    for feature_col in feature_cols:
      if feature_col not in df.columns:
        continue

      mask = df[feature_col].fillna(False).astype(bool)
      count_with = int(mask.sum())
      count_without = int((~mask).sum())

      avg_with = float(df.loc[mask, metric_col].mean()) if mask.any() else 0.0
      avg_without = float(df.loc[~mask, metric_col].mean()) if (~mask).any() else 0.0

      sufficient = count_with >= min_feature_support and count_without >= min_feature_support

      uplift_pct = np.nan
      if avg_without > 0:
        uplift_pct = ((avg_with - avg_without) / avg_without) * 100

      records.append({
        "feature_col": feature_col,
        "feature_label": label_map.get(feature_col, feature_col),
        "count_with_feature": count_with,
        "count_without_feature": count_without,
        "support_ratio": count_with / len(df) if len(df) > 0 else np.nan,
        "sufficient_support": sufficient,
        "avg_with_feature": avg_with,
        "avg_without_feature": avg_without,
        "uplift_pct": uplift_pct,
        "target_20pct_met": bool(pd.notna(uplift_pct) and uplift_pct >= 20 and sufficient),
        "impact_score": (
          float(uplift_pct * np.log1p(count_with)) if pd.notna(uplift_pct) else np.nan
        ),
      })

    result_df = pd.DataFrame(records)
    if result_df.empty:
      return result_df

    return result_df.sort_values(
      by=["target_20pct_met", "impact_score", "uplift_pct"],
      ascending=[False, False, False],
      na_position="last",
    ).reset_index(drop=True)
