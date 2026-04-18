from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_COLOR_PRIMARY = "#D95F02"
_COLOR_SECONDARY = "#4C78A8"
_COLOR_POSITIVE = "#2A9D8F"
_COLOR_NEGATIVE = "#D62828"
_LABEL_MARGIN_LEFT = 220
_PLOT_BG = "#111318"
_PAPER_BG = "#111318"
_GRID = "#2A2F3A"
_TEXT = "#E8ECF3"
_SUBTLE = "#9AA4B2"


def _empty_figure(message: str) -> go.Figure:
  fig = go.Figure()
  fig.add_annotation(
    text=message,
    x=0.5,
    y=0.5,
    showarrow=False,
    xref="paper",
    yref="paper",
    font=dict(color=_TEXT),
  )
  fig.update_layout(
    height=420,
    plot_bgcolor=_PLOT_BG,
    paper_bgcolor=_PAPER_BG,
    font=dict(color=_TEXT),
  )
  return fig


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
  fig.update_layout(
    plot_bgcolor=_PLOT_BG,
    paper_bgcolor=_PAPER_BG,
    font=dict(color=_TEXT),
    title_font=dict(color=_TEXT),
    legend_font=dict(color=_TEXT),
    hoverlabel=dict(bgcolor="#1B1F2A", font_color=_TEXT),
  )
  fig.update_xaxes(
    gridcolor=_GRID,
    zerolinecolor=_GRID,
    tickfont=dict(color=_TEXT),
    title_font=dict(color=_TEXT),
  )
  fig.update_yaxes(
    gridcolor=_GRID,
    zerolinecolor=_GRID,
    tickfont=dict(color=_TEXT),
    title_font=dict(color=_TEXT),
  )
  for annotation in fig.layout.annotations:
    annotation.font = dict(color=_TEXT, size=annotation.font.size if annotation.font else 12)
  return fig


def plot_group_comparison(
  overall_summary_df: pd.DataFrame,
  metric_name: str,
  chart_title: str,
) -> go.Figure:
  if overall_summary_df.empty or len(overall_summary_df) < 2:
    return _empty_figure("Không đủ dữ liệu để so sánh 2 nhóm.")

  left = overall_summary_df.iloc[0]
  right = overall_summary_df.iloc[1]
  uplift_mean = (
    ((left["avg_metric"] - right["avg_metric"]) / right["avg_metric"]) * 100
    if pd.notna(right["avg_metric"]) and float(right["avg_metric"]) > 0
    else np.nan
  )
  uplift_median = (
    ((left["median_metric"] - right["median_metric"]) / right["median_metric"]) * 100
    if pd.notna(right["median_metric"]) and float(right["median_metric"]) > 0
    else np.nan
  )

  fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
      f"Mean: {uplift_mean:+.1f}%" if pd.notna(uplift_mean) else "Mean",
      f"Median: {uplift_median:+.1f}%" if pd.notna(uplift_median) else "Median",
    ),
    horizontal_spacing=0.12,
  )

  labels = [
    "Nhóm nổi bật hơn",
    "Nhóm còn lại",
  ]

  for value, label, color in [
    (left["avg_metric"], labels[0], _COLOR_PRIMARY),
    (right["avg_metric"], labels[1], _COLOR_SECONDARY),
  ]:
    fig.add_trace(go.Bar(
      x=[label],
      y=[value],
      marker_color=color,
      text=[f"{float(value):,.1f}" if pd.notna(value) else "N/A"],
      textposition="outside",
      cliponaxis=False,
      showlegend=False,
      customdata=[[int(left["count"])] if color == _COLOR_PRIMARY else [int(right["count"])]],
      hovertemplate="<b>%{x}</b><br>Giá trị: %{y:,.1f}<br>Số sản phẩm: %{customdata[0]:,}<extra></extra>",
    ), row=1, col=1)

  for value, label, color in [
    (left["median_metric"], labels[0], _COLOR_PRIMARY),
    (right["median_metric"], labels[1], _COLOR_SECONDARY),
  ]:
    fig.add_trace(go.Bar(
      x=[label],
      y=[value],
      marker_color=color,
      text=[f"{float(value):,.1f}" if pd.notna(value) else "N/A"],
      textposition="outside",
      cliponaxis=False,
      showlegend=False,
      customdata=[[int(left["count"])] if color == _COLOR_PRIMARY else [int(right["count"])]],
      hovertemplate="<b>%{x}</b><br>Giá trị: %{y:,.1f}<br>Số sản phẩm: %{customdata[0]:,}<extra></extra>",
    ), row=1, col=2)

  fig.update_layout(
    title=dict(text=chart_title, x=0.5, xanchor="center"),
    height=450,
    margin=dict(l=20, r=20, t=90, b=20),
    yaxis=dict(gridcolor=_GRID, zeroline=False, title=metric_name),
    yaxis2=dict(gridcolor=_GRID, zeroline=False, title=metric_name),
  )
  return _apply_dark_theme(fig)


def plot_score_distribution(
  score_summary_df: pd.DataFrame,
  score_col: str,
  score_label: str,
  metric_name: str,
  chart_title: str,
) -> go.Figure:
  if score_summary_df.empty:
    return _empty_figure("Không có dữ liệu phân bố điểm.")

  chart_df = score_summary_df.copy()
  chart_df[score_col] = chart_df[score_col].astype(str)

  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(go.Bar(
    x=chart_df[score_col],
    y=chart_df["product_count"],
    name="Số sản phẩm",
    marker_color=_COLOR_SECONDARY,
    hovertemplate=f"{score_label}: %{{x}}<br>Số sản phẩm: %{{y:,}}<extra></extra>",
  ), secondary_y=False)
  fig.add_trace(go.Scatter(
    x=chart_df[score_col],
    y=chart_df["avg_metric"],
    name=f"{metric_name} trung bình",
    mode="lines+markers",
    line=dict(color=_COLOR_PRIMARY, width=3),
    marker=dict(size=9),
    hovertemplate=f"{score_label}: %{{x}}<br>{metric_name} TB: %{{y:,.1f}}<extra></extra>",
  ), secondary_y=True)

  fig.update_layout(
    title=dict(text=chart_title, x=0.5, xanchor="center"),
    height=420,
    margin=dict(l=20, r=20, t=70, b=20),
    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
  )
  fig.update_xaxes(title_text=score_label, showgrid=False)
  fig.update_yaxes(title_text="Số sản phẩm", secondary_y=False, gridcolor=_GRID)
  fig.update_yaxes(title_text=metric_name, secondary_y=True, showgrid=False)
  return _apply_dark_theme(fig)


def plot_uplift_bars(
  impact_df: pd.DataFrame,
  label_col: str,
  title: str,
  threshold_pct: float,
) -> go.Figure:
  if impact_df.empty:
    return _empty_figure("Không có dữ liệu để xếp hạng tác động.")

  chart_df = impact_df.copy().sort_values("uplift_pct", ascending=True)
  colors = np.where(chart_df["uplift_pct"].fillna(-999) >= 0, _COLOR_POSITIVE, _COLOR_NEGATIVE)
  count_col = "count_with_feature" if "count_with_feature" in chart_df.columns else "count_with_signal"

  fig = go.Figure()
  fig.add_trace(go.Bar(
    y=chart_df[label_col],
    x=chart_df["uplift_pct"],
    orientation="h",
    marker_color=colors,
    text=chart_df["uplift_pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "N/A"),
    textposition="outside",
    cliponaxis=False,
    customdata=np.stack([
      chart_df["support_ratio"].fillna(0),
      chart_df[count_col].fillna(0),
    ], axis=1),
    hovertemplate=(
      "<b>%{y}</b><br>"
      "Chênh lệch: %{x:+.1f}%<br>"
      "Tỷ lệ xuất hiện: %{customdata[0]:.1%}<br>"
      "Số sản phẩm có tín hiệu: %{customdata[1]:,.0f}<extra></extra>"
    ),
    showlegend=False,
  ))
  fig.add_vline(
    x=threshold_pct,
    line_dash="dot",
    line_color="#1D6996",
    annotation_text=f"Ngưỡng {threshold_pct:.0f}%",
    annotation_position="top right",
  )
  fig.add_vline(x=0, line_color="#AAAAAA", line_width=1)
  fig.update_layout(
    title=dict(text=title, x=0.5, xanchor="center"),
    xaxis_title="Chênh lệch %",
    yaxis_title="",
    yaxis_automargin=True,
    height=max(420, len(chart_df) * 48),
    margin=dict(l=_LABEL_MARGIN_LEFT, r=50, t=80, b=20),
    xaxis=dict(gridcolor=_GRID, zeroline=False),
    yaxis=dict(showgrid=False),
  )
  return _apply_dark_theme(fig)
