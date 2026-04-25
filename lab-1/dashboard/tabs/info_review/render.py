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
  # st.subheader("Thông tin sách và phản hồi người mua")
  # st.caption(
  #   "Phân tích chất lượng thông tin sản phẩm và nội dung review để rút ra insight "
  #   "về hành vi mua hàng và hiệu quả bán."
  # )

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

  with st.sidebar:
    st.markdown("### Bộ lọc phân tích")
    metric_name = st.selectbox(
      "Chỉ số đánh giá",
      list(metric_options.keys()),
      key="info_review_metric",
    )
    metric_col = metric_options[metric_name]
    min_sold_count = st.number_input(
      "Ngưỡng sold_count tối thiểu",
      min_value=0,
      value=0,
      step=10,
      key="info_review_min_sold_count",
    )
    info_threshold = st.slider(
      "Ngưỡng điểm thông tin đầy đủ",
      min_value=1,
      max_value=len(retriever.get_info_feature_columns()),
      value=4,
      step=1,
      key="info_review_info_threshold",
      help="Sản phẩm có score >= ngưỡng sẽ được xem là nhóm thông tin đầy đủ hơn.",
    )
    min_feature_support = st.slider(
      "Số mẫu tối thiểu/đặc trưng",
      min_value=5,
      max_value=200,
      value=50,
      step=5,
      key="info_review_min_feature_support",
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

  st.markdown("### Tổng quan")
  kpi_col_1, kpi_col_2, kpi_col_3 = st.columns(3)
  with kpi_col_1:
    st.metric("Số mẫu", f"{len(filtered_df):,}".replace(",", "."))
  with kpi_col_2:
    st.metric("Lượt bán TB", _format_number(filtered_df["sold_count"].mean(), 1))
  with kpi_col_3:
    st.metric("Review TB", _format_number(filtered_df["review_count"].mean(), 1))

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

  info_chart_col_1, info_chart_col_2 = st.columns(2)
  with info_chart_col_1:
    info_uplift = info_result.get("uplift_pct", np.nan)
    insight_text_1 = "Sách có độ hoàn thiện thông tin cao giúp người mua tin tưởng hơn, từ đó cải thiện doanh số đáng kể, khẳng định vai trò thiết yếu của mô tả sản phẩm chi tiết." if pd.notna(info_uplift) and info_uplift >= 20 else "Thông tin đầy đủ có hỗ trợ việc bán hàng, nhưng mức độ chênh lệch doanh thu chưa thực sự tạo đột phá lớn như kỳ vọng đối với tập dữ liệu hiện tại."
    render_chart_with_insight(
      plot_group_comparison(
        info_result["overall_summary"],
        metric_name,
        "So sánh nhóm thông tin đầy đủ hơn vs nhóm còn lại",
      ),
      toggle_key="info_chart_group_comparison",
      insight_text=insight_text_1,
      palette=palette,
    )
  with info_chart_col_2:
    top_score_bucket = info_result["score_summary"].sort_values("avg_metric", ascending=False).iloc[0]["info_completeness_score"] if not info_result["score_summary"].empty else "cao"
    insight_text_2 = f"Sản phẩm có độ hoàn thiện thông tin đạt điểm {top_score_bucket} đang là ngưỡng lý tưởng mang lại lượng bán trung bình cao nhất, nên lấy đây làm tiêu chuẩn chung cho việc trình bày sản phẩm."
    render_chart_with_insight(
      plot_score_distribution(
        info_result["score_summary"],
        "info_completeness_score",
        "Điểm đầy đủ thông tin",
        metric_name,
        "Phân bố điểm thông tin và biến động hiệu quả bán hàng",
      ),
      toggle_key="info_chart_score_distribution",
      insight_text=insight_text_2,
      palette=palette,
    )

  info_features_df = info_result["feature_impact"]
  target_met_count = int((info_features_df["target_20pct_met"]).sum()) if not info_features_df.empty else 0
  top_info_features = ", ".join(info_features_df.head(2)["feature_label"].tolist()) if not info_features_df.empty else "các thành phần"
  insight_text_3 = f"Đầu tư vào các thông tin như '{top_info_features}' mang lại lượt bán tăng vọt, đây chính là các trọng điểm cần ưu tiên bổ sung ngay khi cập nhật danh mục sách." if target_met_count >= 3 else f"Tuy các thành phần thông tin (như {top_info_features}) có tác động tích cực, nhưng mức chênh lệch chưa đủ để tạo thành cú hích doanh số mạnh mẽ."
  render_chart_with_insight(
    plot_uplift_bars(
      info_result["feature_impact"],
      "feature_label",
      "Mức chênh lệch theo từng thành phần thông tin",
      threshold_pct=20,
    ),
    toggle_key="info_chart_feature_uplift",
    insight_text=insight_text_3,
    palette=palette,
  )

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

  review_chart_col_1, review_chart_col_2 = st.columns(2)
  with review_chart_col_1:
    review_uplift = review_result.get("uplift_pct", np.nan)
    insight_text_review_1 = "Trải nghiệm và đánh giá tích cực từ người mua trước là thỏi nam châm thu hút khách hàng, kéo theo sự tăng trưởng doanh số ấn tượng." if pd.notna(review_uplift) and review_uplift >= 15 else "Mặc dù review tích cực giúp xây dựng hình ảnh tốt, nhưng sự chênh lệch về hiệu quả bán hàng giữa hai nhóm chưa đạt mức vượt trội."
    render_chart_with_insight(
      plot_group_comparison(
        review_result["overall_summary"],
        metric_name,
        "So sánh nhóm review tích cực hơn vs nhóm còn lại",
      ),
      toggle_key="info_chart_review_group",
      insight_text=insight_text_review_1,
      palette=palette,
    )
  with review_chart_col_2:
    top_review_bucket = review_result["score_summary"].sort_values("avg_metric", ascending=False).iloc[0]["review_score_bucket"] if not review_result["score_summary"].empty else "tích cực"
    insight_text_review_2 = f"Các sản phẩm thuộc nhóm điểm đánh giá '{top_review_bucket}' sở hữu mức hiệu suất bán nổi trội nhất, minh chứng cho sức mạnh của sự truyền miệng tích cực từ cộng đồng."
    render_chart_with_insight(
      plot_score_distribution(
        review_result["score_summary"],
        "review_score_bucket",
        "Mức độ tích cực của review",
        metric_name,
        "Phân bố mức tích cực của review và hiệu quả bán hàng",
      ),
      toggle_key="info_chart_review_score_distribution",
      insight_text=insight_text_review_2,
      palette=palette,
    )

  review_features_df = review_result["signal_impact"]
  review_target_met_count = int((review_features_df["target_15pct_met"]).sum()) if not review_features_df.empty else 0
  top_review_features = ", ".join(review_features_df.head(2)["signal_label"].tolist()) if not review_features_df.empty else "các tín hiệu"
  insight_text_review_3 = f"Những phản hồi thực tế từ khách hàng về '{top_review_features}' đã thành công trong việc kích thích người dùng mới chốt đơn hiệu quả hơn hẳn." if review_target_met_count >= 2 else f"Các từ khóa review (như {top_review_features}) đem lại cảm nhận tốt cho gian hàng, nhưng sức bật doanh thu cụ thể của từng yếu tố chưa quá bùng nổ."
  render_chart_with_insight(
    plot_uplift_bars(
      review_result["signal_impact"],
      "signal_label",
      "Mức chênh lệch theo từng tín hiệu review",
      threshold_pct=15,
    ),
    toggle_key="info_chart_review_signal_uplift",
    insight_text=insight_text_review_3,
    palette=palette,
  )
