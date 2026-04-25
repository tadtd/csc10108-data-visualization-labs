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
from dashboard.utils import apply_common_style, get_palette, render_chart_with_insight


@st.cache_data(show_spinner=False)
def _load_prepared_data() -> tuple[pd.DataFrame, str]:
  retriever = FomoGiftRetriever()
  products_df, source_path = retriever.load_products()
  prepared_df = retriever.prepare_dataset(products_df)
  return prepared_df, source_path


def _format_number(value: float, decimals: int = 2) -> str:
  if pd.isna(value):
    return "N/A"
  return f"{float(value):,.{decimals}f}"


def render():
  apply_common_style()
  # st.subheader("Huy hiệu bán chạy và giá trị cộng thêm")
  # st.caption(
  #   "Phân tích tác động của huy hiệu bán chạy và các yếu tố quà tặng/bản đặc biệt "
  #   "đến hiệu quả bán hàng theo từng nhóm sản phẩm."
  # )

  retriever = FomoGiftRetriever()
  color_mode = st.session_state.get("color_mode", "Mặc định")
  palette = get_palette(color_mode)

  try:
    prepared_df, source_path = _load_prepared_data()
  except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    return

  if prepared_df.empty:
    st.warning("Dữ liệu rỗng, chưa thể thực hiện phân tích.")
    return

  # st.info(f"Đang sử dụng cleaned data: {source_path} | Tổng số sản phẩm: {len(prepared_df):,}")

  metric_options = {
    "Số lượng bán": "sold_count",
    "Doanh thu ước tính (sold_count x price)": "revenue_estimated",
  }

  category_series = prepared_df["category"].astype("string").fillna("Không phân loại").str.strip()
  category_series = category_series.where(category_series != "", "Không phân loại")
  available_categories = sorted(category_series.unique().tolist())

  with st.sidebar:
    st.markdown("### Bộ lọc phân tích")
    metric_name = st.selectbox("Chỉ số đánh giá", list(metric_options.keys()), key="fomo_metric")
    metric_col = metric_options[metric_name]
    badge_col = st.selectbox(
      "Cột huy hiệu",
      ["is_top100", "is_bestseller"],
      key="fomo_badge_col",
      help="is_top100: đứng trong Top 100 danh mục con; is_bestseller: huy hiệu bán chạy.",
    )
    min_sold_count = st.number_input(
      "Ngưỡng sold_count tối thiểu",
      min_value=0,
      value=0,
      step=10,
      key="fomo_min_sold_count",
    )
    min_feature_support = st.slider(
      "Số mẫu tối thiểu/đặc trưng",
      min_value=5,
      max_value=200,
      value=20,
      step=5,
      key="fomo_min_feature_support",
      help="Đặc trưng có số mẫu thấp hơn ngưỡng này chỉ nên dùng để tham khảo.",
    )
    selected_categories = st.multiselect(
      "Lọc theo category",
      options=available_categories,
      default=[],
      key="fomo_selected_categories",
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

  st.markdown("### Tổng quan")
  kpi_col_1, kpi_col_2, kpi_col_3 = st.columns(3)
  with kpi_col_1:
    st.metric("Số mẫu", f"{len(filtered_df):,}".replace(",", "."))
  with kpi_col_2:
    st.metric("Lượt bán TB", _format_number(filtered_df["sold_count"].mean(), 1))
  with kpi_col_3:
    st.metric("Rating TB", _format_number(filtered_df["rating"].mean(), 2))

  st.markdown("### 1) Tác động của huy hiệu bán chạy")
  st.caption(
    f"So sánh chênh lệch {metric_name.lower()} giữa nhóm có huy hiệu **{badge_col}** "
    "và nhóm còn lại trong cùng tập lọc."
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

    insight_1 = "Hiệu ứng FOMO thể hiện sức mạnh áp đảo, huy hiệu bán chạy giúp duy trì mức doanh số vượt trội hơn hẳn so với mặt bằng chung." if pd.notna(uplift_pct) and uplift_pct >= 50 else "Huy hiệu bán chạy có mang lại lợi thế doanh số, nhưng chưa đạt mức chênh lệch đột phá để tạo thành hiệu ứng FOMO rõ rệt."
    render_chart_with_insight(
      plot_top100_overall_comparison(overall_summary, metric_name),
      toggle_key="fomo_chart_top100_overall",
      insight_text=insight_1,
      palette=palette,
    )

    if not category_summary.empty:
      st.markdown("#### Phân tích theo danh mục")
      cat_hit = int(category_summary["target_50pct_met"].sum())
      cat_total = len(category_summary)

      chart_s1_cat_1, chart_s1_cat_2 = st.columns(2)
      with chart_s1_cat_1:
        insight_cat_1 = "Nhiều danh mục sách tận dụng rất tốt huy hiệu bán chạy để kéo doanh số, chứng tỏ hiệu ứng FOMO phát huy tác dụng cực kỳ hiệu quả ở các ngách nhất định." if cat_total > 0 and cat_hit >= cat_total / 3 else "Chỉ một số ít danh mục có sự bứt phá về doanh số nhờ huy hiệu, cho thấy hiệu ứng FOMO phân bổ không đồng đều trên thị trường."
        render_chart_with_insight(
          plot_top100_by_category(category_summary, top_n=10),
          toggle_key="fomo_chart_top100_by_category",
          insight_text=insight_cat_1,
          palette=palette,
        )
      with chart_s1_cat_2:
        render_chart_with_insight(
          plot_top100_uplift_by_category(category_summary, top_n=12),
          toggle_key="fomo_chart_top100_uplift_category",
          insight_text="Những cột cao nhất chính là các danh mục được hưởng lợi mạnh nhất từ huy hiệu, gợi ý đây là các nhóm ưu tiên hàng đầu để chạy chiến dịch tranh Top.",
          palette=palette,
        )
    else:
      st.info(
        "Không có danh mục nào đủ mẫu (≥10 sản phẩm mỗi nhóm) để phân tích theo category."
      )

  st.markdown("### 2) Tác động của quà tặng và phiên bản đặc biệt")
  st.caption(
    "Phát hiện từ khóa quà tặng/bản đặc biệt trong tiêu đề và mô tả sản phẩm, "
    "sau đó so sánh hiệu quả bán giữa nhóm có/không có giá trị cộng thêm."
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

    gift_target_met = int((gift_impact_df["target_20pct_met"]).sum()) if not gift_impact_df.empty else 0
    insight_gift_1 = "Chiến lược 'Giá trị cộng thêm' (bản đặc biệt, quà tặng) đã thành công rực rỡ trong việc kích thích tâm lý sưu tầm, kéo theo doanh thu tăng vọt." if gift_target_met >= 1 else "Giá trị cộng thêm mang lại sự khác biệt tích cực, nhưng mức chênh lệch doanh số vẫn chưa thực sự tạo cú hích đủ lớn như kỳ vọng ban đầu."
    render_chart_with_insight(
      plot_gift_uplift(eligible_gift_df if not eligible_gift_df.empty else gift_impact_df),
      toggle_key="fomo_chart_gift_uplift",
      insight_text=insight_gift_1,
      palette=palette,
    )
    render_chart_with_insight(
      plot_gift_average_comparison(eligible_gift_df if not eligible_gift_df.empty else gift_impact_df),
      toggle_key="fomo_chart_gift_avg_compare",
      insight_text="Sự khác biệt về giá trị trung bình tái khẳng định sức hút của sách có bản đặc biệt hoặc quà tặng so với phiên bản tiêu chuẩn trên thị trường.",
      palette=palette,
    )
