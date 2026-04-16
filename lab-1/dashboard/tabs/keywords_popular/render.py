from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .chart import (
  plot_author_group_comparison,
  plot_author_popularity_scatter,
  plot_author_top_sales,
  plot_feature_average_comparison,
  plot_feature_uplift,
  plot_ml_feature_importance,
)
from .retrieve import KeywordsPopularRetriever


@st.cache_data(show_spinner=False)
def _load_prepared_data(csv_path: str) -> tuple[pd.DataFrame, str]:
  retriever = KeywordsPopularRetriever()
  products_df, source_path = retriever.load_products(csv_path if csv_path.strip() else None)
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
  st.subheader("Mục tiêu Tuấn: Keyword/Pattern tiêu đề và độ phổ biến tác giả")
  st.caption(
    "Tab này tập trung vào 2 mục tiêu SMART của thành viên Tuấn: "
    "(1) xác định top đặc trưng tiêu đề giúp cải thiện doanh số; "
    "(2) kiểm chứng nhóm Top tác giả phổ biến có hiệu quả bán tốt hơn nhóm còn lại."
  )

  retriever = KeywordsPopularRetriever()

  with st.expander("Nguồn dữ liệu", expanded=True):
    csv_path = st.text_input(
      "Đường dẫn file products.csv",
      value="data/raw/products.csv",
      key="keywords_popular_csv_path",
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
      metric_name = st.selectbox("Chỉ số đánh giá", list(metric_options.keys()), key="keywords_metric")
      metric_col = metric_options[metric_name]
    with filter_col_2:
      top_n_authors = st.slider("Số tác giả Top", min_value=5, max_value=20, value=10, step=1)
    with filter_col_3:
      min_sold_count = st.number_input("Ngưỡng sold_count tối thiểu", min_value=0, value=0, step=10)
    with filter_col_4:
      min_review_count = st.number_input("Ngưỡng review_count tối thiểu", min_value=0, value=0, step=5)

    selected_categories = st.multiselect(
      "Lọc theo category",
      options=available_categories,
      default=[],
      help="Bỏ trống để giữ toàn bộ category.",
    )
    include_unknown_author = st.checkbox(
      "Giữ sản phẩm thiếu tên tác giả (Unknown)",
      value=True,
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

  metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
  with metric_col_1:
    st.metric("Số sản phẩm đang phân tích", f"{len(filtered_df):,}")
  with metric_col_2:
    st.metric("Số tác giả duy nhất", f"{filtered_df['author_primary'].nunique():,}")
  with metric_col_3:
    st.metric(f"{metric_name} trung bình", _format_number(filtered_df[metric_col].mean()))
  with metric_col_4:
    st.metric("Tỉ lệ sách có tác giả Unknown", _format_number((filtered_df["author_primary"] == "Unknown").mean() * 100) + "%")

  st.markdown("### Mục tiêu 1: Ảnh hưởng của keyword/pattern tiêu đề đến doanh số")
  feature_impact_df = retriever.compute_feature_impact(filtered_df, metric_col=metric_col)

  if feature_impact_df.empty:
    st.warning("Không đủ dữ liệu để tính tác động keyword/pattern.")
  else:
    top3_features = (
      feature_impact_df.dropna(subset=["uplift_pct"])
      .sort_values(by="uplift_pct", ascending=False)
      .head(3)
      .copy()
    )
    target_hit_count = int((top3_features["uplift_pct"] >= 20).sum()) if not top3_features.empty else 0

    if target_hit_count >= 3:
      st.success(
        "Đạt mục tiêu SMART 1: Top 3 đặc trưng quan trọng đều có mức chênh lệch từ 20% trở lên."
      )
    else:
      st.warning(
        "Chưa đạt hoàn toàn SMART 1 với bộ lọc hiện tại. "
        f"Số đặc trưng đạt ngưỡng 20% trong Top 3: {target_hit_count}/3."
      )

    viz_col_1, viz_col_2 = st.columns(2)
    with viz_col_1:
      st.plotly_chart(plot_feature_uplift(feature_impact_df, top_n=9), use_container_width=True)
    with viz_col_2:
      st.plotly_chart(plot_feature_average_comparison(feature_impact_df, top_n=6), use_container_width=True)

    sort_map = {
      "Chênh lệch % giảm dần": ("uplift_pct", False),
      "Giá trị trung bình nhóm có đặc trưng": ("avg_with_feature", False),
      "Số lượng mẫu nhóm có đặc trưng": ("count_with_feature", False),
      "Tên đặc trưng": ("feature_label", True),
    }
    selected_sort = st.selectbox("Sắp xếp bảng đặc trưng", list(sort_map.keys()), key="keywords_feature_sort")
    sort_col, ascending = sort_map[selected_sort]
    feature_table_df = feature_impact_df.sort_values(by=sort_col, ascending=ascending).copy()

    feature_table_df = feature_table_df[
      [
        "feature_type",
        "feature_label",
        "count_with_feature",
        "avg_with_feature",
        "avg_without_feature",
        "uplift_pct",
        "target_20pct_met",
      ]
    ].rename(
      columns={
        "feature_type": "Loại",
        "feature_label": "Đặc trưng",
        "count_with_feature": "Số mẫu có đặc trưng",
        "avg_with_feature": "Trung bình nhóm có đặc trưng",
        "avg_without_feature": "Trung bình nhóm còn lại",
        "uplift_pct": "Chênh lệch %",
        "target_20pct_met": "Đạt ngưỡng 20%",
      }
    )
    feature_table_df["Đạt ngưỡng 20%"] = np.where(feature_table_df["Đạt ngưỡng 20%"], "Đạt", "Chưa đạt")
    st.dataframe(feature_table_df, use_container_width=True)

  st.markdown("### Mục tiêu 2: Ảnh hưởng độ phổ biến tác giả đến hiệu quả bán hàng")
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

    metric_a, metric_b, metric_c = st.columns(3)
    with metric_a:
      st.metric(f"{metric_name} TB nhóm Top", _format_number(top_avg))
    with metric_b:
      st.metric(f"{metric_name} TB nhóm còn lại", _format_number(other_avg))
    with metric_c:
      st.metric("Mức chênh lệch %", _format_number(uplift_pct) + "%")

    if author_result["target_25pct_met"]:
      st.success("Đạt mục tiêu SMART 2: nhóm sách thuộc Top tác giả phổ biến cao hơn ít nhất 25%.")
    else:
      st.warning("Chưa đạt ngưỡng 25% với bộ lọc hiện tại cho mục tiêu SMART 2.")

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
      st.plotly_chart(plot_author_top_sales(author_stats_df, top_n=top_n_authors), use_container_width=True)
    with chart_col_2:
      st.plotly_chart(plot_author_group_comparison(group_summary_df), use_container_width=True)

    st.plotly_chart(plot_author_popularity_scatter(author_stats_df, top_n=top_n_authors), use_container_width=True)

    author_table_df = author_stats_df[
      [
        "rank",
        "author_name",
        "total_books",
        "total_sales",
        "avg_sales_per_book",
        "popularity_score",
      ]
    ].rename(
      columns={
        "rank": "Hạng",
        "author_name": "Tác giả",
        "total_books": "Số đầu sách",
        "total_sales": "Tổng số bán",
        "avg_sales_per_book": "TB số bán/cuốn",
        "popularity_score": "Điểm phổ biến",
      }
    )
    st.dataframe(author_table_df.head(30), use_container_width=True)

  st.markdown("### Bonus: Machine Learning (Feature Importance)")
  run_ml = st.toggle(
    "Bật mô hình RandomForest để xếp hạng mức ảnh hưởng của đặc trưng tiêu đề",
    value=True,
    key="keywords_run_ml",
  )

  if run_ml:
    ml_result = retriever.run_ml_feature_importance(filtered_df, target_col=metric_col)
    if not ml_result.get("available", False):
      st.info(ml_result.get("message", "Không thể chạy mô hình ML."))
    else:
      ml_metric_1, ml_metric_2, ml_metric_3 = st.columns(3)
      with ml_metric_1:
        st.metric("Số mẫu train/test", f"{ml_result['sample_size']:,}")
      with ml_metric_2:
        st.metric("MAE", _format_number(ml_result["mae"]))
      with ml_metric_3:
        st.metric("R2 (log-scale)", _format_number(ml_result["r2_log"], decimals=4))

      st.plotly_chart(
        plot_ml_feature_importance(ml_result["importance_df"], top_n=10),
        use_container_width=True,
      )

      ml_top3_df = ml_result["importance_df"].head(3).copy()
      ml_top3_df = ml_top3_df[["feature_label", "importance"]].rename(
        columns={
          "feature_label": "Top đặc trưng theo ML",
          "importance": "Mức độ quan trọng",
        }
      )
      st.dataframe(ml_top3_df, use_container_width=True)

  st.markdown("### Kết luận nhanh theo dữ liệu đang lọc")
  conclusion_lines: list[str] = []

  if not feature_impact_df.empty:
    strongest_feature = feature_impact_df.iloc[0]
    conclusion_lines.append(
      "- Đặc trưng tiêu đề nổi bật nhất hiện tại: "
      f"**{strongest_feature['feature_label']}** (chênh lệch {strongest_feature['uplift_pct']:.2f}%)."
    )

  if not group_summary_df.empty:
    conclusion_lines.append(
      "- Chênh lệch giữa nhóm Top tác giả và nhóm còn lại: "
      f"**{_format_number(author_result['uplift_pct'])}%** theo chỉ số **{metric_name}**."
    )

  if not conclusion_lines:
    st.write("Chưa đủ dữ liệu để kết luận.")
  else:
    for line in conclusion_lines:
      st.markdown(line)