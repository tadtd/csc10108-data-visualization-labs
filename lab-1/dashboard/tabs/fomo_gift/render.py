from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .chart import (
  plot_gift_average_comparison,
  plot_gift_uplift,
  plot_top100_by_category,
  plot_top100_overall_comparison,
  plot_top100_uplift_by_category,
)
from .retrieve import FomoGiftRetriever


@st.cache_data(show_spinner=False)
def _load_prepared_data(csv_path: str) -> tuple[pd.DataFrame, str]:
  retriever = FomoGiftRetriever()
  products_df, source_path = retriever.load_products(csv_path if csv_path.strip() else None)
  prepared_df = retriever.prepare_dataset(products_df)
  return prepared_df, source_path


def _format_number(value: float, decimals: int = 2) -> str:
  if pd.isna(value):
    return "N/A"
  return f"{float(value):,.{decimals}f}"


def render():
  st.subheader("Mục tiêu phân tích: Hiệu ứng FOMO từ huy hiệu Top Bán Chạy và Giá trị cộng thêm")
  st.caption(
    "Tab này tập trung vào 2 mục tiêu SMART: "
    "(1) kiểm chứng huy hiệu 'Top Bán Chạy' giúp doanh số vượt trội ≥50% so với sản phẩm không lọt Top; "
    "(2) đánh giá nhóm sách có quà tặng/bản đặc biệt có doanh số cao hơn ≥20% so với bản thường."
  )

  retriever = FomoGiftRetriever()

  with st.expander("Nguồn dữ liệu", expanded=True):
    csv_path = st.text_input(
      "Đường dẫn file products.csv",
      value="data/processed/products_clean.csv",
      key="fomo_gift_csv_path",
    )
    st.caption("Có thể để trống để tab tự tìm file theo thứ tự ưu tiên trong thư mục data.")

  try:
    prepared_df, source_path = _load_prepared_data(csv_path)
  except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    return

  if prepared_df.empty:
    st.warning("Dữ liệu rỗng, chưa thể thực hiện phân tích.")
    return

  st.info(f"Đang sử dụng dữ liệu: {source_path} | Tổng số sản phẩm: {len(prepared_df):,}")

  metric_options = {
    "Số lượng bán": "sold_count",
    "Doanh thu ước tính (sold_count x price)": "revenue_estimated",
  }

  category_series = prepared_df["category"].astype("string").fillna("Không phân loại").str.strip()
  category_series = category_series.where(category_series != "", "Không phân loại")
  available_categories = sorted(category_series.unique().tolist())

  with st.expander("Bộ lọc & cấu hình phân tích", expanded=True):
    filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns(4)
    with filter_col_1:
      metric_name = st.selectbox("Chỉ số đánh giá", list(metric_options.keys()), key="fomo_metric")
      metric_col = metric_options[metric_name]
    with filter_col_2:
      badge_col = st.selectbox(
        "Cột huy hiệu (SMART 1)",
        ["is_top100", "is_bestseller"],
        key="fomo_badge_col",
        help="is_top100: đứng trong Top 100 danh mục con; is_bestseller: huy hiệu bán chạy.",
      )
    with filter_col_3:
      min_sold_count = st.number_input("Ngưỡng sold_count tối thiểu", min_value=0, value=0, step=10)
    with filter_col_4:
      min_feature_support = st.slider(
        "Số mẫu tối thiểu/đặc trưng",
        min_value=5,
        max_value=200,
        value=20,
        step=5,
        help="Đặc trưng có số mẫu thấp hơn ngưỡng này sẽ không dùng để kết luận SMART 2.",
      )

    selected_categories = st.multiselect(
      "Lọc theo category",
      options=available_categories,
      default=[],
      help="Bỏ trống để giữ toàn bộ category.",
    )

  filtered_df = prepared_df.copy()
  filtered_df["category"] = category_series
  filtered_df = filtered_df[filtered_df["sold_count"] >= float(min_sold_count)]

  if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

  if filtered_df.empty:
    st.warning("Không còn dữ liệu sau khi lọc. Hãy nới lỏng điều kiện lọc để tiếp tục.")
    return

  top_count = int(filtered_df[badge_col].astype(bool).sum()) if badge_col in filtered_df.columns else 0
  gift_count = int(filtered_df["has_gift_combined"].astype(bool).sum()) if "has_gift_combined" in filtered_df.columns else 0

  ov_col_1, ov_col_2, ov_col_3, ov_col_4 = st.columns(4)
  with ov_col_1:
    st.metric("Số sản phẩm đang phân tích", f"{len(filtered_df):,}")
  with ov_col_2:
    st.metric(f"Sản phẩm có huy hiệu Top", f"{top_count:,}")
  with ov_col_3:
    st.metric("Sản phẩm có quà tặng/đặc biệt", f"{gift_count:,}")
  with ov_col_4:
    st.metric(f"{metric_name} trung bình", _format_number(filtered_df[metric_col].mean()))

  # ── SMART 1 ────────────────────────────────────────────────────────────────
  st.markdown("### Mục tiêu 1: Hiệu ứng FOMO từ huy hiệu 'Top Bán Chạy'")
  st.caption(
    f"Kiểm chứng liệu sản phẩm có huy hiệu **{badge_col}** có doanh số trung bình "
    "cao hơn ít nhất **50%** so với sản phẩm không lọt Top trong cùng thể loại hay không."
  )

  top100_result = retriever.compute_top100_impact(
    filtered_df,
    metric_col=metric_col,
    badge_col=badge_col,
    min_category_support=10,
  )

  overall_summary = top100_result["overall_summary"]
  category_summary = top100_result["category_summary"]
  uplift_pct = top100_result["uplift_pct"]

  if overall_summary.empty:
    st.warning(
      f"Cột '{badge_col}' không tồn tại hoặc không có sản phẩm nào mang huy hiệu trong bộ lọc hiện tại."
    )
  else:
    smart1_col_1, smart1_col_2, smart1_col_3, smart1_col_4 = st.columns(4)
    with smart1_col_1:
      st.metric(f"{metric_name} TB (Có Top)", _format_number(top100_result["avg_top"]))
    with smart1_col_2:
      st.metric(f"{metric_name} TB (Không Top)", _format_number(top100_result["avg_non_top"]))
    with smart1_col_3:
      st.metric("Mức chênh lệch %", _format_number(uplift_pct) + "%")
    with smart1_col_4:
      st.metric("Số SP có huy hiệu Top", f"{top100_result['top_count']:,}")

    if top100_result["target_50pct_met"]:
      st.success(
        f"Đạt mục tiêu SMART 1: Sản phẩm có huy hiệu Top vượt trội hơn "
        f"{_format_number(uplift_pct)}% — đạt ngưỡng ≥50%."
      )
    elif pd.notna(uplift_pct) and uplift_pct > 0:
      st.warning(
        f"Có chênh lệch tích cực ({_format_number(uplift_pct)}%) nhưng chưa đạt ngưỡng 50% "
        "của mục tiêu SMART 1 với bộ lọc hiện tại."
      )
    else:
      st.error(
        "Chưa thấy lợi thế rõ ràng từ huy hiệu Top với bộ lọc hiện tại. "
        "Thử mở rộng bộ lọc hoặc chọn cột huy hiệu khác."
      )

    st.plotly_chart(
      plot_top100_overall_comparison(overall_summary, metric_name),
      use_container_width=True,
    )

    if not category_summary.empty:
      st.markdown("#### Phân tích theo danh mục")
      cat_hit = int(category_summary["target_50pct_met"].sum())
      cat_total = len(category_summary)
      st.caption(
        f"Có **{cat_hit}/{cat_total}** danh mục đạt ngưỡng chênh lệch ≥50% "
        f"(ngưỡng tối thiểu mỗi nhóm: 10 sản phẩm)."
      )

      chart_s1_cat_1, chart_s1_cat_2 = st.columns(2)
      with chart_s1_cat_1:
        st.plotly_chart(
          plot_top100_by_category(category_summary, top_n=10),
          use_container_width=True,
        )
      with chart_s1_cat_2:
        st.plotly_chart(
          plot_top100_uplift_by_category(category_summary, top_n=12),
          use_container_width=True,
        )

      cat_table = category_summary.rename(columns={
        "category": "Danh mục",
        "avg_top": f"TB {metric_name} (Có Top)",
        "avg_non_top": f"TB {metric_name} (Không Top)",
        "uplift_pct": "Chênh lệch %",
        "count_top": "Số SP có Top",
        "count_non_top": "Số SP không Top",
        "target_50pct_met": "Đạt ngưỡng 50%",
      })
      cat_table["Đạt ngưỡng 50%"] = np.where(cat_table["Đạt ngưỡng 50%"], "Đạt", "Chưa đạt")
      st.dataframe(cat_table, use_container_width=True)
    else:
      st.info(
        "Không có danh mục nào đủ mẫu (≥10 sản phẩm mỗi nhóm) để phân tích theo category."
      )

  # ── SMART 2 ────────────────────────────────────────────────────────────────
  st.markdown("### Mục tiêu 2: Sức hút của 'Giá trị cộng thêm' (Bản đặc biệt, Quà tặng kèm)")
  st.caption(
    "Phát hiện từ khóa quà tặng/bản đặc biệt trong tiêu đề và mô tả sản phẩm, "
    "sau đó kiểm chứng nhóm này có doanh số trung bình cao hơn ≥**20%** so với bản thường."
  )

  gift_impact_df = retriever.compute_gift_impact(
    filtered_df,
    metric_col=metric_col,
    min_feature_support=min_feature_support,
  )

  if gift_impact_df.empty:
    st.warning("Không đủ dữ liệu để phân tích tác động quà tặng/giá trị cộng thêm.")
  else:
    eligible_gift_df = gift_impact_df[
      gift_impact_df["sufficient_support"] & gift_impact_df["uplift_pct"].notna()
    ].copy()

    smart2_col_1, smart2_col_2, smart2_col_3 = st.columns(3)
    with smart2_col_1:
      st.metric("Đặc trưng đủ mẫu", f"{len(eligible_gift_df)}/{len(gift_impact_df)}")
    with smart2_col_2:
      st.metric("Đặc trưng đạt ≥20%", f"{int(eligible_gift_df['target_20pct_met'].sum()):,}")
    with smart2_col_3:
      combined_row = gift_impact_df[gift_impact_df["feature_col"] == "has_gift_combined"]
      combined_uplift = float(combined_row.iloc[0]["uplift_pct"]) if not combined_row.empty else np.nan
      st.metric("Chênh lệch (Kết hợp tất cả quà)", _format_number(combined_uplift) + "%")

    if not eligible_gift_df.empty and eligible_gift_df["target_20pct_met"].any():
      best_gift = eligible_gift_df[eligible_gift_df["target_20pct_met"]].iloc[0]
      st.success(
        f"Đạt mục tiêu SMART 2: Đặc trưng **{best_gift['feature_label']}** "
        f"có mức chênh lệch {_format_number(best_gift['uplift_pct'])}% — đạt ngưỡng ≥20%."
      )
    elif not eligible_gift_df.empty:
      best_gift = eligible_gift_df.iloc[0]
      st.warning(
        f"Chưa đạt ngưỡng 20% cho mục tiêu SMART 2. "
        f"Đặc trưng tốt nhất hiện tại: **{best_gift['feature_label']}** "
        f"({_format_number(best_gift['uplift_pct'])}%). "
        "Thử giảm ngưỡng mẫu tối thiểu hoặc mở rộng bộ lọc."
      )
    else:
      st.warning("Không có đặc trưng nào đủ mẫu để kết luận SMART 2.")

    chart_s2_col_1, chart_s2_col_2 = st.columns(2)
    with chart_s2_col_1:
      st.plotly_chart(plot_gift_uplift(eligible_gift_df if not eligible_gift_df.empty else gift_impact_df), use_container_width=True)
    with chart_s2_col_2:
      st.plotly_chart(plot_gift_average_comparison(eligible_gift_df if not eligible_gift_df.empty else gift_impact_df), use_container_width=True)

    sort_map = {
      "Chênh lệch % giảm dần": ("uplift_pct", False),
      "Điểm tác động giảm dần": ("impact_score", False),
      "Số mẫu có đặc trưng": ("count_with_feature", False),
      "Tỉ lệ xuất hiện giảm dần": ("support_ratio", False),
      "Tên đặc trưng": ("feature_label", True),
    }
    selected_sort = st.selectbox("Sắp xếp bảng đặc trưng", list(sort_map.keys()), key="fomo_gift_sort")
    sort_col, ascending = sort_map[selected_sort]
    gift_table_df = gift_impact_df.sort_values(by=sort_col, ascending=ascending).copy()

    gift_table_df = gift_table_df[[
      "feature_label",
      "count_with_feature",
      "support_ratio",
      "sufficient_support",
      "avg_with_feature",
      "avg_without_feature",
      "uplift_pct",
      "impact_score",
      "target_20pct_met",
    ]].rename(columns={
      "feature_label": "Loại giá trị cộng thêm",
      "count_with_feature": "Số mẫu có đặc trưng",
      "support_ratio": "Tỉ lệ xuất hiện (%)",
      "sufficient_support": "Đủ mẫu tối thiểu",
      "avg_with_feature": "TB có quà tặng/đặc biệt",
      "avg_without_feature": "TB không có quà tặng/đặc biệt",
      "uplift_pct": "Chênh lệch %",
      "impact_score": "Điểm tác động",
      "target_20pct_met": "Đạt ngưỡng 20%",
    })
    gift_table_df["Tỉ lệ xuất hiện (%)"] = gift_table_df["Tỉ lệ xuất hiện (%)"] * 100
    gift_table_df["Đủ mẫu tối thiểu"] = np.where(gift_table_df["Đủ mẫu tối thiểu"], "Đủ", "Thiếu")
    gift_table_df["Đạt ngưỡng 20%"] = np.where(gift_table_df["Đạt ngưỡng 20%"], "Đạt", "Chưa đạt")
    st.dataframe(gift_table_df, use_container_width=True)

  # ── Kết luận ───────────────────────────────────────────────────────────────
  st.markdown("### Kết luận nhanh theo dữ liệu đang lọc")
  conclusion_lines: list[str] = []

  if not overall_summary.empty and pd.notna(uplift_pct):
    conclusion_lines.append(
      f"- Huy hiệu **{badge_col}**: nhóm Có Top có {metric_name} trung bình cao hơn nhóm còn lại "
      f"**{_format_number(uplift_pct)}%** — "
      + ("đạt ngưỡng SMART 50%." if top100_result["target_50pct_met"] else "chưa đạt ngưỡng SMART 50%.")
    )

  if not gift_impact_df.empty:
    combined_row = gift_impact_df[gift_impact_df["feature_col"] == "has_gift_combined"]
    if not combined_row.empty:
      cu = float(combined_row.iloc[0]["uplift_pct"])
      conclusion_lines.append(
        f"- Giá trị cộng thêm (kết hợp): nhóm có quà tặng/bản đặc biệt cao hơn "
        f"**{_format_number(cu)}%** — "
        + ("đạt ngưỡng SMART 20%." if cu >= 20 else "chưa đạt ngưỡng SMART 20%.")
      )
    if not gift_impact_df.empty:
      best = gift_impact_df.iloc[0]
      conclusion_lines.append(
        f"- Đặc trưng quà tặng nổi bật nhất: **{best['feature_label']}** "
        f"(chênh lệch {_format_number(best['uplift_pct'])}%)."
      )

  if not conclusion_lines:
    st.write("Chưa đủ dữ liệu để kết luận.")
  else:
    for line in conclusion_lines:
      st.markdown(line)
