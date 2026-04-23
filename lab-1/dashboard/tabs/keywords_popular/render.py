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
  st.subheader("Từ khóa tiêu đề và độ phổ biến tác giả")
  st.caption(
    "Phân tích ảnh hưởng của đặc trưng tiêu đề và độ phổ biến tác giả "
    "đến hiệu quả bán hàng của sách."
  )

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

  with st.expander("Bộ lọc & cấu hình phân tích", expanded=True):
    filter_col_1, filter_col_2, filter_col_3, filter_col_4, filter_col_5 = st.columns(5)
    with filter_col_1:
      metric_name = st.selectbox("Chỉ số đánh giá", list(metric_options.keys()), key="keywords_metric")
      metric_col = metric_options[metric_name]
    with filter_col_2:
      top_n_authors = st.slider(
        "Số tác giả hàng đầu",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        key="keywords_top_n_authors",
      )
    with filter_col_3:
      min_sold_count = st.number_input(
        "Ngưỡng sold_count tối thiểu",
        min_value=0,
        value=0,
        step=10,
        key="keywords_min_sold_count",
      )
    with filter_col_4:
      min_review_count = st.number_input(
        "Ngưỡng review_count tối thiểu",
        min_value=0,
        value=0,
        step=5,
        key="keywords_min_review_count",
      )
    with filter_col_5:
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
    show_detail_insights = st.toggle(
      "Hiển thị insight chi tiết",
      value=True,
      key="keywords_show_detail_insights",
      help="Bật để hiển thị thêm diễn giải chi tiết theo từng biểu đồ.",
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

    stat_1, stat_2, stat_3 = st.columns(3)
    with stat_1:
      st.metric("Đặc trưng đủ mẫu", f"{len(eligible_feature_df)}/{len(feature_impact_df)}")
    with stat_2:
      st.metric("Đặc trưng uplift dương", f"{int((eligible_feature_df['uplift_pct'] > 0).sum()):,}")
    with stat_3:
      st.metric("Top 3 có uplift >=20%", f"{target_hit_count}/3")

    if show_detail_insights:
      if len(top3_features) < 3:
        st.warning(
          "Chưa đủ 3 đặc trưng có số mẫu mạnh để kết luận chắc chắn. "
          "Bạn có thể giảm ngưỡng hỗ trợ hoặc mở rộng bộ lọc dữ liệu."
        )
      elif target_hit_count >= 3:
        st.success(
          "Nhóm top đặc trưng tiêu đề đang cho chênh lệch tích cực rõ rệt ở tập dữ liệu hiện tại."
        )
      else:
        st.warning(
          f"Số đặc trưng có uplift >=20% trong Top 3 hiện tại: {target_hit_count}/3."
        )

    chart_feature_df = eligible_feature_df if not eligible_feature_df.empty else feature_impact_df

    viz_col_1, viz_col_2 = st.columns(2)
    with viz_col_1:
      render_chart_with_insight(
        plot_feature_uplift(chart_feature_df, top_n=9),
        toggle_key="keywords_chart_feature_uplift",
        insight_text=f"Những đặc trưng nằm trên cùng là ứng viên Top ảnh hưởng tiêu đề; hiện Top 3 có {target_hit_count}/3 đặc trưng đạt ngưỡng uplift >=20%.",
        palette=palette,
      )
    with viz_col_2:
      render_chart_with_insight(
        plot_feature_average_comparison(chart_feature_df, top_n=6),
        toggle_key="keywords_chart_feature_avg_compare",
        insight_text="So sánh trung bình giữa nhóm có/không có đặc trưng giúp xác nhận mức uplift tiêu đề có ổn định hay chỉ do nhiễu mẫu.",
        palette=palette,
      )

    sort_map = {
      "Chênh lệch % giảm dần": ("uplift_pct", False),
      "Điểm tác động giảm dần": ("impact_score", False),
      "Giá trị trung bình nhóm có đặc trưng": ("avg_with_feature", False),
      "Số lượng mẫu nhóm có đặc trưng": ("count_with_feature", False),
      "Tỉ lệ xuất hiện giảm dần": ("support_ratio", False),
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
        "support_ratio",
        "sufficient_support",
        "avg_with_feature",
        "avg_without_feature",
        "uplift_pct",
        "impact_score",
        "target_20pct_met",
      ]
    ].rename(
      columns={
        "feature_type": "Loại",
        "feature_label": "Đặc trưng",
        "count_with_feature": "Số mẫu có đặc trưng",
        "support_ratio": "Tỉ lệ xuất hiện (%)",
        "sufficient_support": "Đủ mẫu tối thiểu",
        "avg_with_feature": "Trung bình nhóm có đặc trưng",
        "avg_without_feature": "Trung bình nhóm còn lại",
        "uplift_pct": "Chênh lệch %",
        "impact_score": "Điểm tác động",
        "target_20pct_met": "Uplift >= 20%",
      }
    )
    feature_table_df["Tỉ lệ xuất hiện (%)"] = feature_table_df["Tỉ lệ xuất hiện (%)"] * 100
    feature_table_df["Đủ mẫu tối thiểu"] = np.where(feature_table_df["Đủ mẫu tối thiểu"], "Đủ", "Thiếu")
    feature_table_df["Uplift >= 20%"] = np.where(feature_table_df["Uplift >= 20%"], "Có", "Không")
    st.dataframe(feature_table_df, width='stretch')

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

    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    with metric_a:
      st.metric(f"{metric_name} TB nhóm Top", _format_number(top_avg))
    with metric_b:
      st.metric(f"{metric_name} TB nhóm còn lại", _format_number(other_avg))
    with metric_c:
      st.metric("Mức chênh lệch %", _format_number(uplift_pct) + "%")
    with metric_d:
      st.metric("Số sách có tác giả hợp lệ", f"{author_result.get('analysis_book_count', 0):,}")

    if show_detail_insights:
      if pd.notna(uplift_pct) and uplift_pct > 0:
        st.success(
          f"Nhóm tác giả phổ biến đang có lợi thế {_format_number(uplift_pct)}% so với nhóm còn lại."
        )
      elif pd.notna(uplift_pct):
        st.warning(
          f"Nhóm tác giả phổ biến đang thấp hơn {abs(float(uplift_pct)):.2f}% so với nhóm còn lại."
        )
      else:
        st.info("Chưa đủ dữ liệu để lượng hóa chênh lệch theo độ phổ biến tác giả.")

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
      render_chart_with_insight(
        plot_author_top_sales(author_stats_df, top_n=top_n_authors),
        toggle_key="keywords_chart_author_top_sales",
        insight_text=f"Biểu đồ xếp hạng doanh số cho thấy nhóm Top {top_n_authors} tác giả nên được ưu tiên trong chiến lược danh mục.",
        palette=palette,
      )
    with chart_col_2:
      author_target_note = "Đạt mục tiêu >=25%" if pd.notna(uplift_pct) and uplift_pct >= 25 else "Chưa đạt mục tiêu >=25%"
      render_chart_with_insight(
        plot_author_group_comparison(group_summary_df),
        toggle_key="keywords_chart_author_group_comparison",
        insight_text=f"Khoảng cách giữa nhóm tác giả hàng đầu và nhóm còn lại phản ánh hiệu ứng thương hiệu tác giả. {author_target_note}.",
        palette=palette,
      )

    render_chart_with_insight(
      plot_author_popularity_scatter(author_stats_df, top_n=top_n_authors),
      toggle_key="keywords_chart_author_popularity",
      insight_text="Biểu đồ tương quan giúp nhận diện tác giả outlier (ít đầu sách nhưng bán tốt, hoặc ngược lại) để điều chỉnh chiến lược hợp tác.",
      palette=palette,
    )

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
    st.dataframe(author_table_df.head(30), width='stretch')

  st.markdown("### 3) Học máy: Mức độ quan trọng của đặc trưng")
  run_ml = st.toggle(
    "Bật mô hình Rừng ngẫu nhiên để xếp hạng mức ảnh hưởng của đặc trưng tiêu đề",
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

      render_chart_with_insight(
        plot_ml_feature_importance(ml_result["importance_df"], top_n=10),
        toggle_key="keywords_chart_ml_importance",
        insight_text="Các đặc trưng đứng đầu theo mô hình học máy giúp xác nhận nhóm yếu tố tiêu đề ảnh hưởng mạnh nhất đến dự báo doanh số.",
        palette=palette,
      )

      ml_top3_df = ml_result["importance_df"].head(3).copy()
      ml_top3_df = ml_top3_df[["feature_label", "importance"]].rename(
        columns={
          "feature_label": "Đặc trưng hàng đầu theo học máy",
          "importance": "Mức độ quan trọng",
        }
      )
      st.dataframe(ml_top3_df, width='stretch')

  if show_detail_insights:
    st.markdown("### Kết luận nhanh theo dữ liệu đang lọc")
    conclusion_lines: list[str] = []

    if not eligible_feature_df.empty:
      strongest_feature = eligible_feature_df.iloc[0]
      conclusion_lines.append(
        "- Đặc trưng tiêu đề nổi bật nhất hiện tại: "
        f"**{strongest_feature['feature_label']}** (chênh lệch {strongest_feature['uplift_pct']:.2f}%)."
      )

    if not group_summary_df.empty:
      conclusion_lines.append(
        "- Chênh lệch giữa nhóm tác giả hàng đầu và nhóm còn lại: "
        f"**{_format_number(author_result['uplift_pct'])}%** theo chỉ số **{metric_name}**."
      )

    if not conclusion_lines:
      st.write("Chưa đủ dữ liệu để kết luận.")
    else:
      for line in conclusion_lines:
        st.markdown(line)