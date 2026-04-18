from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _empty_figure(message: str) -> go.Figure:
  fig = go.Figure()
  fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
  fig.update_layout(height=420)
  return fig


def plot_feature_uplift(feature_impact_df: pd.DataFrame, top_n: int = 8) -> go.Figure:
  if feature_impact_df.empty:
    return _empty_figure("Không có dữ liệu đặc trưng để trực quan hóa.")

  chart_df = feature_impact_df.copy().head(top_n)
  if "support_ratio" not in chart_df.columns:
    chart_df["support_ratio"] = np.nan
  if "count_with_feature" not in chart_df.columns:
    chart_df["count_with_feature"] = 0
  if "sufficient_support" not in chart_df.columns:
    chart_df["sufficient_support"] = False

  chart_df = chart_df.sort_values(by="uplift_pct", ascending=True)
  chart_df["group"] = np.where(chart_df["uplift_pct"] >= 0, "Tích cực", "Tiêu cực")

  fig = px.bar(
    chart_df,
    x="uplift_pct",
    y="feature_label",
    orientation="h",
    color="group",
    color_discrete_map={"Tích cực": "#2E8B57", "Tiêu cực": "#B22222"},
    labels={
      "uplift_pct": "Chênh lệch % của nhóm có đặc trưng so với nhóm còn lại",
      "feature_label": "Đặc trưng",
      "group": "Xu hướng",
    },
    hover_data={
      "count_with_feature": True,
      "support_ratio": ":.2%",
      "sufficient_support": True,
      "group": False,
    },
    title="Top đặc trưng tiêu đề theo mức chênh lệch doanh số",
  )
  fig.add_vline(x=20, line_dash="dash", line_color="#1E90FF")
  fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10), height=460)
  return fig


def plot_feature_average_comparison(feature_impact_df: pd.DataFrame, top_n: int = 6) -> go.Figure:
  if feature_impact_df.empty:
    return _empty_figure("Không có dữ liệu để so sánh trung bình giữa các nhóm.")

  chart_df = feature_impact_df.copy().head(top_n)
  chart_df = chart_df.sort_values(by="avg_with_feature", ascending=True)

  fig = go.Figure()
  fig.add_trace(
    go.Bar(
      y=chart_df["feature_label"],
      x=chart_df["avg_with_feature"],
      orientation="h",
      name="Nhóm có đặc trưng",
      marker_color="#0068C9",
    )
  )
  fig.add_trace(
    go.Bar(
      y=chart_df["feature_label"],
      x=chart_df["avg_without_feature"],
      orientation="h",
      name="Nhóm không có đặc trưng",
      marker_color="#8BA3C7",
    )
  )

  fig.update_layout(
    barmode="group",
    title="So sánh chỉ số trung bình giữa 2 nhóm sản phẩm",
    xaxis_title="Giá trị trung bình",
    yaxis_title="Đặc trưng",
    margin=dict(l=10, r=10, t=60, b=10),
    height=460,
    legend_title_text="",
  )
  return fig


def plot_author_top_sales(author_stats_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
  if author_stats_df.empty:
    return _empty_figure("Không có dữ liệu tác giả để hiển thị.")

  chart_df = author_stats_df.head(top_n).copy()
  chart_df = chart_df.sort_values(by="total_sales", ascending=True)

  fig = px.bar(
    chart_df,
    x="total_sales",
    y="author_name",
    orientation="h",
    labels={"total_sales": "Tổng số lượng bán", "author_name": "Tác giả"},
    title=f"Top {top_n} tác giả phổ biến theo tổng số bán",
    color="rank",
    color_continuous_scale="Blues",
  )
  fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=60, b=10), height=500)
  return fig


def plot_author_popularity_scatter(author_stats_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
  if author_stats_df.empty:
    return _empty_figure("Không có dữ liệu tác giả để vẽ scatter.")

  chart_df = author_stats_df.copy()
  chart_df["group"] = np.where(chart_df["rank"] <= top_n, f"Top {top_n}", "Khác")

  fig = px.scatter(
    chart_df,
    x="total_books",
    y="total_sales",
    size="avg_sales_per_book",
    color="group",
    hover_name="author_name",
    color_discrete_map={f"Top {top_n}": "#0068C9", "Khác": "#9AA5B1"},
    labels={
      "total_books": "Số đầu sách",
      "total_sales": "Tổng số lượng bán",
      "avg_sales_per_book": "TB số bán/cuốn",
      "group": "Nhóm",
    },
    title="Quan hệ giữa số đầu sách và tổng số bán theo tác giả",
  )
  fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=60, b=10), height=500)
  return fig


def plot_author_group_comparison(group_summary_df: pd.DataFrame) -> go.Figure:
  if group_summary_df.empty:
    return _empty_figure("Không có dữ liệu nhóm tác giả để so sánh.")

  fig = px.bar(
    group_summary_df,
    x="author_group",
    y="avg_metric",
    text="avg_metric",
    color="author_group",
    color_discrete_sequence=["#0068C9", "#8BA3C7"],
    labels={"author_group": "Nhóm tác giả", "avg_metric": "Giá trị trung bình"},
    title="So sánh nhóm Top tác giả phổ biến và nhóm còn lại",
  )
  fig.update_traces(texttemplate="%{text:,.2f}", textposition="outside")
  fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=60, b=10), height=420)
  return fig


def plot_ml_feature_importance(importance_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
  if importance_df.empty:
    return _empty_figure("Không có kết quả feature importance từ mô hình ML.")

  chart_df = importance_df.head(top_n).copy()
  chart_df = chart_df.sort_values(by="importance", ascending=True)

  fig = px.bar(
    chart_df,
    x="importance",
    y="feature_label",
    orientation="h",
    title="Feature importance (RandomForest)",
    labels={"importance": "Mức độ quan trọng", "feature_label": "Đặc trưng"},
    color="importance",
    color_continuous_scale="Teal",
  )
  fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=60, b=10), height=500)
  return fig
