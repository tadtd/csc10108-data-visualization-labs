from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .chart import (
  plot_group_comparison,
  plot_score_distribution,
  plot_uplift_bars,
)
from .retrieve import InfoReviewRetriever
from dashboard.utils import apply_common_style, get_palette, render_chart_with_insight


@st.cache_data(show_spinner=False)
def _load_prepared_data() -> tuple[pd.DataFrame, str, str]:
  retriever = InfoReviewRetriever()
  products_df, reviews_df, product_source, review_source = retriever.load_data()
  prepared_df = retriever.prepare_dataset(products_df, reviews_df)
  return prepared_df, product_source, review_source


def _format_number(value: float, decimals: int = 2) -> str:
  if pd.isna(value):
    return "N/A"
  return f"{float(value):,.{decimals}f}"


def _format_list(df: pd.DataFrame, label_col: str, value_col: str) -> str:
  if df.empty:
    return "Chưa đủ dữ liệu"
  return "; ".join(
    f"{row[label_col]} ({_format_number(row[value_col])}%)"
    for _, row in df.iterrows()
  )


def _build_reliability_note(total_count: int, support_count: int, threshold: int) -> str:
  if support_count == 0:
    return "Chưa có đặc trưng nào đủ mẫu để kết luận chắc chắn."
  return (
    f"Có {support_count}/{total_count} đặc trưng đủ mẫu (mỗi phía >= {threshold} sản phẩm). "
    "Nên ưu tiên đọc kết quả của các đặc trưng này."
  )


def render():
  apply_common_style()
  st.subheader("Thông tin sách và phản hồi người mua")
  st.caption(
    "Phân tích chất lượng thông tin sản phẩm và nội dung review để rút ra insight "
    "về hành vi mua hàng và hiệu quả bán."
  )

  retriever = InfoReviewRetriever()
  color_mode = st.session_state.get("color_mode", "Mặc định")
  palette = get_palette(color_mode)

  try:
    prepared_df, product_source, review_source = _load_prepared_data()
  except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    return

  if prepared_df.empty:
    st.warning("Dữ liệu rỗng, chưa thể thực hiện phân tích.")
    return

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
      metric_name = st.selectbox(
        "Chỉ số đánh giá",
        list(metric_options.keys()),
        key="info_review_metric",
      )
      metric_col = metric_options[metric_name]
    with filter_col_2:
      min_sold_count = st.number_input(
        "Ngưỡng sold_count tối thiểu",
        min_value=0,
        value=0,
        step=10,
        key="info_review_min_sold_count",
      )
    with filter_col_3:
      info_threshold = st.slider(
        "Ngưỡng điểm thông tin đầy đủ",
        min_value=1,
        max_value=len(retriever.get_info_feature_columns()),
        value=4,
        step=1,
        key="info_review_info_threshold",
        help="Sản phẩm có score >= ngưỡng sẽ được xem là nhóm thông tin đầy đủ hơn.",
      )
    with filter_col_4:
      min_feature_support = st.slider(
        "Số mẫu tối thiểu/đặc trưng",
        min_value=5,
        max_value=200,
        value=50,
        step=5,
        key="info_review_min_feature_support",
      )
    show_detail_insights = st.toggle(
      "Hiển thị insight chi tiết",
      value=True,
      key="info_review_show_detail_insights",
      help="Bật để xem thêm các nhận xét diễn giải chi tiết.",
    )

    selected_categories = st.multiselect(
      "Lọc theo category",
      options=available_categories,
      default=[],
      key="info_review_selected_categories",
      help="Bỏ trống để giữ toàn bộ category.",
    )

  filtered_df = prepared_df.copy()
  filtered_df["category"] = category_series
  filtered_df = filtered_df[filtered_df["sold_count"] >= float(min_sold_count)]
  if selected_categories:
    filtered_df = filtered_df[filtered_df["category"].isin(selected_categories)]

  if filtered_df.empty:
    st.warning("Không còn dữ liệu sau khi lọc. Hãy nới điều kiện lọc để tiếp tục.")
    return

  overview_col_1, overview_col_2, overview_col_3, overview_col_4 = st.columns(4)
  with overview_col_1:
    st.metric("Số sản phẩm đang phân tích", f"{len(filtered_df):,}")
  with overview_col_2:
    st.metric(
      "Điểm thông tin TB",
      _format_number(filtered_df["info_completeness_score"].mean()),
    )
  with overview_col_3:
    st.metric(
      "Rating review TB",
      _format_number(filtered_df["review_avg_rating_combined"].replace(0, np.nan).mean()),
    )
  with overview_col_4:
    st.metric(
      f"{metric_name} trung bình",
      _format_number(filtered_df[metric_col].mean()),
    )

  st.markdown("### 1) Tác động của mức độ hoàn thiện thông tin sách")
  st.caption(
    "So sánh hiệu quả bán giữa nhóm có thông tin đầy đủ hơn và nhóm còn lại, "
    f"đồng thời xác định các thành phần thông tin ảnh hưởng mạnh đến {metric_name.lower()}."
  )

  info_result = retriever.compute_information_completeness_impact(
    filtered_df,
    metric_col=metric_col,
    score_threshold=info_threshold,
    min_feature_support=min_feature_support,
  )

  info_col_1, info_col_2, info_col_3, info_col_4 = st.columns(4)
  with info_col_1:
    st.metric(
      f"{metric_name} TB (Thông tin đầy đủ hơn)",
      _format_number(info_result["overall_summary"].iloc[0]["avg_metric"]),
    )
  with info_col_2:
    st.metric(
      f"{metric_name} TB (Nhóm còn lại)",
      _format_number(info_result["overall_summary"].iloc[1]["avg_metric"]),
    )
  with info_col_3:
    st.metric("Chênh lệch %", _format_number(info_result["uplift_pct"]) + "%")
  with info_col_4:
    st.metric("Top 3 thành phần nổi bật", f"{len(info_result['top_components']):,}")

  if show_detail_insights:
    if pd.notna(info_result["uplift_pct"]) and info_result["uplift_pct"] > 0:
      st.success(
        f"Nhóm thông tin đầy đủ hơn đang cao hơn {_format_number(info_result['uplift_pct'])}% so với nhóm còn lại."
      )
    elif pd.notna(info_result["uplift_pct"]):
      st.warning(
        f"Nhóm thông tin đầy đủ hơn đang thấp hơn {abs(float(info_result['uplift_pct'])):.2f}% so với nhóm còn lại."
      )
    else:
      st.info("Chưa đủ dữ liệu để lượng hóa chênh lệch ở bộ lọc hiện tại.")

  if show_detail_insights:
    st.caption(
      "3 thành phần thông tin đang có tín hiệu mạnh nhất: "
      + _format_list(info_result["top_components"], "feature_label", "uplift_pct")
    )
    st.caption(_build_reliability_note(
      total_count=len(info_result["feature_impact"]),
      support_count=int(info_result["feature_impact"]["sufficient_support"].sum()),
      threshold=min_feature_support,
    ))
    if (info_result["overall_summary"]["median_metric"] == 0).any():
      st.caption(
        "Lưu ý: median của ít nhất một nhóm đang bằng 0, cho thấy dữ liệu bán hàng lệch mạnh; "
        "hãy đọc kết quả uplift cùng với support và median."
      )

  info_chart_col_1, info_chart_col_2 = st.columns(2)
  with info_chart_col_1:
    info_target_note = "Đạt mục tiêu >=20%" if pd.notna(info_result["uplift_pct"]) and info_result["uplift_pct"] >= 20 else "Chưa đạt mục tiêu >=20%"
    render_chart_with_insight(
      plot_group_comparison(
        info_result["overall_summary"],
        metric_name,
        "So sánh nhóm thông tin đầy đủ hơn vs nhóm còn lại",
      ),
      toggle_key="info_chart_group_comparison",
      insight_text=f"Khoảng cách hai nhóm phản ánh tác động của mức độ đầy đủ thông tin đến hiệu quả bán. {info_target_note}.",
      palette=palette,
    )
  with info_chart_col_2:
    render_chart_with_insight(
      plot_score_distribution(
        info_result["score_summary"],
        "info_completeness_score",
        "Điểm đầy đủ thông tin",
        metric_name,
        "Phân bố điểm thông tin và biến động hiệu quả bán hàng",
      ),
      toggle_key="info_chart_score_distribution",
      insight_text="Phân bố score giúp xác định ngưỡng thông tin đầy đủ đang tạo cải thiện doanh số rõ nhất cho từng nhóm sản phẩm.",
      palette=palette,
    )

  render_chart_with_insight(
    plot_uplift_bars(
      info_result["feature_impact"],
      "feature_label",
      "Mức chênh lệch theo từng thành phần thông tin",
      threshold_pct=20,
    ),
    toggle_key="info_chart_feature_uplift",
    insight_text="Các thành phần vượt ngưỡng uplift >=20% là ứng viên ưu tiên để tối ưu mô tả/metadata sản phẩm.",
    palette=palette,
  )

  info_table = info_result["feature_impact"][[
    "feature_label",
    "count_with_feature",
    "count_without_feature",
    "support_ratio",
    "avg_with_feature",
    "avg_without_feature",
    "uplift_pct",
    "sufficient_support",
    "target_20pct_met",
  ]].rename(columns={
    "feature_label": "Thành phần thông tin",
    "count_with_feature": "Số SP có thành phần",
    "count_without_feature": "Số SP không có thành phần",
    "support_ratio": "Tỷ lệ xuất hiện",
    "avg_with_feature": f"TB {metric_name} (Có thành phần)",
    "avg_without_feature": f"TB {metric_name} (Không có thành phần)",
    "uplift_pct": "Chênh lệch %",
    "sufficient_support": "Đủ mẫu",
    "target_20pct_met": "Uplift >= 20%",
  })
  info_table["Đủ mẫu"] = np.where(info_table["Đủ mẫu"], "Đủ", "Thiếu")
  info_table["Uplift >= 20%"] = np.where(info_table["Uplift >= 20%"], "Có", "Không")
  st.dataframe(info_table, width='stretch')

  st.markdown("### 2) Tác động của cảm nhận người mua qua review")
  st.caption(
    "Kết hợp rating trung bình, tỷ lệ review tích cực và từ khóa cảm nhận "
    f"để so sánh khác biệt {metric_name.lower()} giữa các nhóm review."
  )

  review_result = retriever.compute_review_impact(
    filtered_df,
    metric_col=metric_col,
    min_feature_support=min_feature_support,
  )

  review_col_1, review_col_2, review_col_3, review_col_4 = st.columns(4)
  with review_col_1:
    st.metric(
      f"{metric_name} TB (Review tích cực hơn)",
      _format_number(review_result["overall_summary"].iloc[0]["avg_metric"]),
    )
  with review_col_2:
    st.metric(
      f"{metric_name} TB (Nhóm còn lại)",
      _format_number(review_result["overall_summary"].iloc[1]["avg_metric"]),
    )
  with review_col_3:
    st.metric("Chênh lệch %", _format_number(review_result["uplift_pct"]) + "%")
  with review_col_4:
    st.metric("Ngưỡng review tích cực", _format_number(review_result["positivity_threshold"]))

  if show_detail_insights:
    if pd.notna(review_result["uplift_pct"]) and review_result["uplift_pct"] > 0:
      st.success(
        f"Nhóm review tích cực hơn đang cao hơn {_format_number(review_result['uplift_pct'])}% so với nhóm còn lại."
      )
    elif pd.notna(review_result["uplift_pct"]):
      st.warning(
        f"Nhóm review tích cực hơn đang thấp hơn {abs(float(review_result['uplift_pct'])):.2f}% so với nhóm còn lại."
      )
    else:
      st.info("Chưa đủ dữ liệu để lượng hóa chênh lệch theo review.")

  if show_detail_insights:
    st.caption(
      "2 tín hiệu review mạnh nhất hiện tại: "
      + _format_list(review_result["top_signals"], "signal_label", "uplift_pct")
    )
    st.caption(_build_reliability_note(
      total_count=len(review_result["signal_impact"]),
      support_count=int(review_result["signal_impact"]["sufficient_support"].sum()),
      threshold=min_feature_support,
    ))
    if (review_result["overall_summary"]["median_metric"] == 0).any():
      st.caption(
        "Lưu ý: median của một nhóm review đang bằng 0, nên cần đọc kết quả cùng độ lệch dữ liệu "
        "và số mẫu của từng tín hiệu."
      )

  review_chart_col_1, review_chart_col_2 = st.columns(2)
  with review_chart_col_1:
    review_target_note = "Đạt mục tiêu >=15%" if pd.notna(review_result["uplift_pct"]) and review_result["uplift_pct"] >= 15 else "Chưa đạt mục tiêu >=15%"
    render_chart_with_insight(
      plot_group_comparison(
        review_result["overall_summary"],
        metric_name,
        "So sánh nhóm review tích cực hơn vs nhóm còn lại",
      ),
      toggle_key="info_chart_review_group",
      insight_text=f"So sánh hai nhóm review cho thấy tác động tổng quan của cảm nhận người mua tới doanh số. {review_target_note}.",
      palette=palette,
    )
  with review_chart_col_2:
    render_chart_with_insight(
      plot_score_distribution(
        review_result["score_summary"],
        "review_score_bucket",
        "Mức độ tích cực của review",
        metric_name,
        "Phân bố mức tích cực của review và hiệu quả bán hàng",
      ),
      toggle_key="info_chart_review_score_distribution",
      insight_text="Biểu đồ giúp xác định bucket review tích cực nào đang gắn với mức hiệu suất bán nổi trội hơn.",
      palette=palette,
    )

  render_chart_with_insight(
    plot_uplift_bars(
      review_result["signal_impact"],
      "signal_label",
      "Mức chênh lệch theo từng tín hiệu review",
      threshold_pct=15,
    ),
    toggle_key="info_chart_review_signal_uplift",
    insight_text="Các tín hiệu review vượt ngưỡng uplift >=15% nên được ưu tiên trong chiến lược cải thiện trải nghiệm khách hàng.",
    palette=palette,
  )

  review_table = review_result["signal_impact"][[
    "signal_label",
    "count_with_signal",
    "count_without_signal",
    "support_ratio",
    "avg_with_signal",
    "avg_without_signal",
    "uplift_pct",
    "sufficient_support",
    "target_15pct_met",
  ]].rename(columns={
    "signal_label": "Tín hiệu review",
    "count_with_signal": "Số SP có tín hiệu",
    "count_without_signal": "Số SP không có tín hiệu",
    "support_ratio": "Tỷ lệ xuất hiện",
    "avg_with_signal": f"TB {metric_name} (Có tín hiệu)",
    "avg_without_signal": f"TB {metric_name} (Không có tín hiệu)",
    "uplift_pct": "Chênh lệch %",
    "sufficient_support": "Đủ mẫu",
    "target_15pct_met": "Uplift >= 15%",
  })
  review_table["Đủ mẫu"] = np.where(review_table["Đủ mẫu"], "Đủ", "Thiếu")
  review_table["Uplift >= 15%"] = np.where(review_table["Uplift >= 15%"], "Có", "Không")
  st.dataframe(review_table, width='stretch')
