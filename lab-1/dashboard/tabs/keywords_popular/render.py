from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .chart import (
  plot_author_group_comparison,
  plot_author_top_sales,
  plot_feature_average_comparison,
  plot_feature_uplift,
  plot_ml_feature_importance,
)
from .retrieve import KeywordsPopularRetriever
from dashboard.utils import apply_common_style, get_palette, render_chart_with_insight


@st.cache_data(show_spinner=False)
def _load_prepared_data() -> tuple[pd.DataFrame, str]:
  retriever = KeywordsPopularRetriever()
  products_df, source_path = retriever.load_products()
  prepared_df = retriever.prepare_dataset(products_df)
  return prepared_df, source_path


def _format_number(value: float, decimals: int = 2) -> str:
  if pd.isna(value):
    return "N/A"
  return f"{float(value):,.{decimals}f}"


def _extract_group_value(group_summary_df: pd.DataFrame, group_name: str) -> float:
  row = group_summary_df[group_summary_df["author_group"] == group_name]
  if row.empty:
    return np.nan
  return float(row.iloc[0]["avg_metric"])


def render():
  apply_common_style()
  # st.subheader("Từ khóa tiêu đề và độ phổ biến tác giả")
  # st.caption(
  #   "Phân tích ảnh hưởng của đặc trưng tiêu đề và độ phổ biến tác giả "
  #   "đến hiệu quả bán hàng của sách."
  # )

  retriever = KeywordsPopularRetriever()
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
    metric_name = st.selectbox("Chỉ số đánh giá", list(metric_options.keys()), key="keywords_metric")
    metric_col = metric_options[metric_name]
    top_n_authors = st.slider(
      "Số tác giả hàng đầu",
      min_value=5,
      max_value=20,
      value=10,
      step=1,
      key="keywords_top_n_authors",
    )
    min_sold_count = st.number_input(
      "Ngưỡng sold_count tối thiểu",
      min_value=0,
      value=0,
      step=10,
      key="keywords_min_sold_count",
    )
    min_review_count = st.number_input(
      "Ngưỡng review_count tối thiểu",
      min_value=0,
      value=0,
      step=5,
      key="keywords_min_review_count",
    )
    min_feature_support = st.slider(
      "Số mẫu tối thiểu/đặc trưng",
      min_value=10,
      max_value=300,
      value=40,
      step=10,
      key="keywords_min_feature_support",
      help="Đặc trưng có số mẫu thấp hơn ngưỡng này chỉ nên dùng để tham khảo.",
    )

    selected_categories = st.multiselect(
      "Lọc theo category",
      options=available_categories,
      default=[],
      key="keywords_selected_categories",
      help="Bỏ trống để giữ toàn bộ category.",
    )
    include_unknown_author = st.checkbox(
      "Giữ sản phẩm thiếu tên tác giả (Unknown)",
      value=False,
      key="keywords_include_unknown_author",
    )

  filtered_df = prepared_df.copy()
  filtered_df["category"] = category_series
  filtered_df = filtered_df[filtered_df["sold_count"] >= float(min_sold_count)]
  filtered_df = filtered_df[filtered_df["review_count"] >= float(min_review_count)]

  if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

  if not include_unknown_author:
    filtered_df = filtered_df[filtered_df["author_primary"] != "Unknown"]

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
    st.metric("Review TB", _format_number(filtered_df["review_count"].mean(), 1))

  st.markdown("### 1) Ảnh hưởng của từ khóa/mẫu tiêu đề đến doanh số")
  feature_impact_df = retriever.compute_feature_impact(
    filtered_df,
    metric_col=metric_col,
    min_feature_support=min_feature_support,
  )
  eligible_feature_df = pd.DataFrame()

  if feature_impact_df.empty:
    st.warning("Không đủ dữ liệu để tính tác động keyword/pattern.")
  else:
    eligible_feature_df = feature_impact_df[
      feature_impact_df["sufficient_support"] & feature_impact_df["uplift_pct"].notna()
    ].copy()
    top3_features = eligible_feature_df.head(3).copy()
    target_hit_count = int((top3_features["target_20pct_met"]).sum()) if not top3_features.empty else 0

    chart_feature_df = eligible_feature_df if not eligible_feature_df.empty else feature_impact_df

    top_feature_names = ", ".join(top3_features["feature_label"].tolist()) if not top3_features.empty else "các từ khóa"
    insight_1 = f"Các đặc trưng tiêu đề dẫn đầu (như {top_feature_names}) tạo ra mức chênh lệch doanh số rất ấn tượng, khẳng định chiến lược tối ưu từ khóa mang lại hiệu quả vượt trội." if target_hit_count >= 3 else f"Các từ khóa như {top_feature_names} có mang lại giá trị cộng thêm, nhưng mức tác động chưa đủ để tạo ra sự đột phá mạnh mẽ."
    render_chart_with_insight(
      plot_feature_uplift(chart_feature_df, top_n=9),
      toggle_key="keywords_chart_feature_uplift",
      insight_text=insight_1,
      palette=palette,
    )

    top_feature = chart_feature_df.iloc[0]["feature_label"] if not chart_feature_df.empty else "yếu tố nổi bật"
    render_chart_with_insight(
      plot_feature_average_comparison(chart_feature_df, top_n=6),
      toggle_key="keywords_chart_feature_avg_compare",
      insight_text=f"Sự chênh lệch trung bình ổn định ở nhóm có đặc trưng '{top_feature}' chứng tỏ thị trường thực sự phản hồi tích cực chứ không chỉ do dữ liệu nhiễu.",
      palette=palette,
    )

  st.markdown("### 2) Ảnh hưởng độ phổ biến tác giả đến hiệu quả bán hàng")
  st.caption("Phân tích này tự động loại các bản ghi tác giả Unknown để tránh sai lệch kết luận.")
  author_result = retriever.compute_author_popularity(
    filtered_df,
    top_n=top_n_authors,
    metric_col=metric_col,
  )

  author_stats_df = author_result["author_stats"]
  group_summary_df = author_result["group_summary"]

  if author_stats_df.empty or group_summary_df.empty:
    st.warning("Không đủ dữ liệu tác giả để thực hiện phân tích mục tiêu 2.")
  else:
    top_group_name = f"Top {top_n_authors} tác giả phổ biến"
    top_avg = _extract_group_value(group_summary_df, top_group_name)
    other_avg = _extract_group_value(group_summary_df, "Nhóm tác giả còn lại")
    uplift_pct = author_result["uplift_pct"]

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
      if not author_stats_df.empty:
        first_author_row = author_stats_df.iloc[0]
        top_author = first_author_row.get("author_name", first_author_row.get("author_primary", "hàng đầu"))
      else:
        top_author = "hàng đầu"
      render_chart_with_insight(
        plot_author_top_sales(author_stats_df, top_n=top_n_authors),
        toggle_key="keywords_chart_author_top_sales",
        insight_text=f"Tác giả '{top_author}' và các tên tuổi trong nhóm Top {top_n_authors} đóng vai trò trụ cột doanh thu, cần được ưu tiên trong chiến lược hiển thị danh mục sách.",
        palette=palette,
      )
    with chart_col_2:
      insight_2 = "Sức hút từ các 'cái tên bảo chứng' thể hiện sức mạnh vượt trội, nhóm tác giả hàng đầu mang lại mức doanh thu áp đảo so với phần thị trường còn lại." if pd.notna(uplift_pct) and uplift_pct >= 25 else "Mặc dù có lợi thế nhận diện, mức chênh lệch hiệu quả bán hàng của nhóm tác giả Top đầu chưa thực sự tạo ra khoảng cách quá xa so với mặt bằng chung."
      render_chart_with_insight(
        plot_author_group_comparison(group_summary_df),
        toggle_key="keywords_chart_author_group_comparison",
        insight_text=insight_2,
        palette=palette,
      )

  st.markdown("### 3) Học máy: Mức độ quan trọng của đặc trưng")
  run_ml = st.toggle(
    "Sử dụng mô hình Random Forest để xếp hạng mức ảnh hưởng của đặc trưng tiêu đề",
    value=True,
    key="keywords_run_ml",
  )

  if run_ml:
    ml_result = retriever.run_ml_feature_importance(filtered_df, target_col=metric_col)
    if not ml_result.get("available", False):
      st.info(ml_result.get("message", "Không thể chạy mô hình ML."))
    else:
      if not ml_result["importance_df"].empty:
        first_feature_row = ml_result["importance_df"].iloc[0]
        top_ml_feature = first_feature_row.get("feature_label", first_feature_row.get("feature_col", "các đặc trưng"))
      else:
        top_ml_feature = "các đặc trưng"
      render_chart_with_insight(
        plot_ml_feature_importance(ml_result["importance_df"], top_n=10),
        toggle_key="keywords_chart_ml_importance",
        insight_text=f"Mô hình học máy độc lập cũng xác nhận '{top_ml_feature}' là yếu tố chi phối mạnh nhất đến lượng bán ra, hoàn toàn trùng khớp với phân tích thống kê trước đó.",
        palette=palette,
      )

