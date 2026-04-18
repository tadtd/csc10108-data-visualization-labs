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


@st.cache_data(show_spinner=False)
def _load_prepared_data(products_path: str, reviews_path: str) -> tuple[pd.DataFrame, str, str]:
  retriever = InfoReviewRetriever()
  products_df, reviews_df, product_source, review_source = retriever.load_data(
    products_path if products_path.strip() else None,
    reviews_path if reviews_path.strip() else None,
  )
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
  st.subheader("Mục tiêu phân tích: Thông tin sách và cảm nhận người mua qua review")
  st.caption(
    "Tab này triển khai 2 mục tiêu SMART của Tâm: "
    "(1) đo tác động của độ hoàn thiện thông tin sách đến doanh số/doanh thu; "
    "(2) đo tác động của review tích cực và xác định các tín hiệu review quan trọng nhất."
  )

  retriever = InfoReviewRetriever()

  with st.expander("Nguồn dữ liệu", expanded=True):
    products_path = st.text_input(
      "Đường dẫn file products.csv",
      value="data/processed/products_clean.csv",
      key="info_review_products_path",
    )
    reviews_path = st.text_input(
      "Đường dẫn file reviews.csv",
      value="data/processed/reviews_clean.csv",
      key="info_review_reviews_path",
    )
    st.caption("Có thể để trống để tab tự tìm file phù hợp trong thư mục data.")

  try:
    prepared_df, product_source, review_source = _load_prepared_data(products_path, reviews_path)
  except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    return

  if prepared_df.empty:
    st.warning("Dữ liệu rỗng, chưa thể thực hiện phân tích.")
    return

  st.info(
    f"Products: {product_source} | Reviews: {review_source} | Tổng số sản phẩm: {len(prepared_df):,}"
  )

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
      )
    with filter_col_3:
      info_threshold = st.slider(
        "Ngưỡng điểm thông tin đầy đủ",
        min_value=1,
        max_value=len(retriever.get_info_feature_columns()),
        value=4,
        step=1,
        help="Sản phẩm có score >= ngưỡng sẽ được xem là nhóm thông tin đầy đủ hơn.",
      )
    with filter_col_4:
      min_feature_support = st.slider(
        "Số mẫu tối thiểu/đặc trưng",
        min_value=5,
        max_value=200,
        value=50,
        step=5,
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

  st.markdown("### Mục tiêu 1: Ảnh hưởng của mức độ hoàn thiện thông tin sách")
  st.caption(
    "Kiểm chứng liệu nhóm sách có thông tin đầy đủ hơn có "
    f"{metric_name.lower()} trung bình cao hơn ít nhất **20%** và xác định 3 thành phần thông tin quan trọng nhất."
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

  if info_result["target_20pct_met"]:
    st.success(
      f"Trong dữ liệu hiện tại, nhóm thông tin đầy đủ hơn có {metric_name.lower()} trung bình "
      f"cao hơn {_format_number(info_result['uplift_pct'])}% so với nhóm còn lại."
    )
  elif pd.notna(info_result["uplift_pct"]) and info_result["uplift_pct"] > 0:
    st.warning(
      f"Có chênh lệch tích cực {_format_number(info_result['uplift_pct'])}% "
      "nhưng chưa đạt ngưỡng 20% với bộ lọc hiện tại."
    )
  else:
    st.error("Chưa thấy lợi thế rõ ràng từ mức độ hoàn thiện thông tin với bộ lọc hiện tại.")

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
    st.plotly_chart(
      plot_group_comparison(
        info_result["overall_summary"],
        metric_name,
        "So sánh nhóm thông tin đầy đủ hơn vs nhóm còn lại",
      ),
      use_container_width=True,
    )
  with info_chart_col_2:
    st.plotly_chart(
      plot_score_distribution(
        info_result["score_summary"],
        "info_completeness_score",
        "Điểm đầy đủ thông tin",
        metric_name,
        "Phân bố điểm thông tin và biến động hiệu quả bán hàng",
      ),
      use_container_width=True,
    )

  st.plotly_chart(
    plot_uplift_bars(
      info_result["feature_impact"],
      "feature_label",
      "Mức chênh lệch theo từng thành phần thông tin",
      threshold_pct=20,
    ),
    use_container_width=True,
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
    "target_20pct_met": "Đạt ngưỡng 20%",
  })
  info_table["Đủ mẫu"] = np.where(info_table["Đủ mẫu"], "Đủ", "Thiếu")
  info_table["Đạt ngưỡng 20%"] = np.where(info_table["Đạt ngưỡng 20%"], "Đạt", "Chưa đạt")
  st.dataframe(info_table, use_container_width=True)

  st.markdown("### Mục tiêu 2: Ảnh hưởng của cảm nhận người mua qua review")
  st.caption(
    "Kết hợp rating trung bình, tỷ lệ review tích cực và từ khóa cảm nhận để kiểm chứng "
    f"liệu nhóm sách có review tích cực hơn có {metric_name.lower()} trung bình cao hơn ít nhất **15%**."
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

  if review_result["target_15pct_met"]:
    st.success(
      f"Trong dữ liệu hiện tại, nhóm review tích cực hơn có {metric_name.lower()} trung bình "
      f"cao hơn {_format_number(review_result['uplift_pct'])}% so với nhóm còn lại."
    )
  elif pd.notna(review_result["uplift_pct"]) and review_result["uplift_pct"] > 0:
    st.warning(
      f"Có chênh lệch tích cực {_format_number(review_result['uplift_pct'])}% "
      "nhưng chưa đạt ngưỡng 15% với bộ lọc hiện tại."
    )
  else:
    st.error("Chưa thấy lợi thế rõ ràng từ review tích cực với bộ lọc hiện tại.")

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
    st.plotly_chart(
      plot_group_comparison(
        review_result["overall_summary"],
        metric_name,
        "So sánh nhóm review tích cực hơn vs nhóm còn lại",
      ),
      use_container_width=True,
    )
  with review_chart_col_2:
    st.plotly_chart(
      plot_score_distribution(
        review_result["score_summary"],
        "review_score_bucket",
        "Mức độ tích cực của review",
        metric_name,
        "Phân bố mức tích cực của review và hiệu quả bán hàng",
      ),
      use_container_width=True,
    )

  st.plotly_chart(
    plot_uplift_bars(
      review_result["signal_impact"],
      "signal_label",
      "Mức chênh lệch theo từng tín hiệu review",
      threshold_pct=15,
    ),
    use_container_width=True,
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
    "target_15pct_met": "Đạt ngưỡng 15%",
  })
  review_table["Đủ mẫu"] = np.where(review_table["Đủ mẫu"], "Đủ", "Thiếu")
  review_table["Đạt ngưỡng 15%"] = np.where(review_table["Đạt ngưỡng 15%"], "Đạt", "Chưa đạt")
  st.dataframe(review_table, use_container_width=True)
