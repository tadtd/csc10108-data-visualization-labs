from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_LABEL_MARGIN_LEFT = 220
_COLOR_TOP = "#FF6B35"
_COLOR_NON_TOP = "#4A90D9"
_COLOR_POSITIVE = "#2E8B57"
_COLOR_NEGATIVE = "#B22222"
_COLOR_GIFT = "#FF6B35"
_COLOR_NO_GIFT = "#8BA3C7"


def _empty_figure(message: str) -> go.Figure:
  fig = go.Figure()
  fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
  fig.update_layout(height=420)
  return fig


def _shorten_group_label(label: str) -> str:
  """Bỏ phần '(N=xxx)' để nhãn trục X không quá dài."""
  if "(" in label:
    return label[: label.index("(")].strip()
  return label


def plot_top100_overall_comparison(
  overall_summary_df: pd.DataFrame,
  metric_name: str = "Số lượng bán",
) -> go.Figure:
  """2 subplot độc lập: trái = Mean, phải = Median — mỗi subplot có trục Y riêng."""
  if overall_summary_df.empty:
    return _empty_figure("Không có dữ liệu so sánh huy hiệu Top.")

  row_top = overall_summary_df[overall_summary_df["group"].str.startswith("Có")]
  row_non = overall_summary_df[overall_summary_df["group"].str.startswith("Không")]
  if row_top.empty or row_non.empty:
    return _empty_figure("Không đủ dữ liệu 2 nhóm để so sánh.")

  avg_top = float(row_top.iloc[0]["avg_metric"])
  avg_non = float(row_non.iloc[0]["avg_metric"])
  med_top = float(row_top.iloc[0]["median_metric"])
  med_non = float(row_non.iloc[0]["median_metric"])
  cnt_top = int(row_top.iloc[0]["count"])
  cnt_non = int(row_non.iloc[0]["count"])

  uplift_mean = ((avg_top - avg_non) / avg_non * 100) if avg_non > 0 else 0.0
  uplift_med = ((med_top - med_non) / med_non * 100) if med_non > 0 else 0.0

  labels = [f"Có Top\n(N={cnt_top:,})", f"Không Top\n(N={cnt_non:,})"]

  fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
      f"Trung bình (Mean) — chênh lệch: <b>{uplift_mean:+.1f}%</b>",
      f"Trung vị (Median) — chênh lệch: <b>{uplift_med:+.1f}%</b>",
    ),
    horizontal_spacing=0.12,
  )

  # Subplot trái: Mean
  for val, label, color in [
    (avg_top, labels[0], _COLOR_TOP),
    (avg_non, labels[1], _COLOR_NON_TOP),
  ]:
    fig.add_trace(go.Bar(
      x=[label],
      y=[val],
      marker_color=color,
      text=[f"{val:,.1f}"],
      textposition="outside",
      cliponaxis=False,
      showlegend=False,
      hovertemplate=f"<b>{label}</b><br>{metric_name} TB: {val:,.2f}<extra></extra>",
    ), row=1, col=1)

  # Subplot phải: Median
  for val, label, color in [
    (med_top, labels[0], _COLOR_TOP),
    (med_non, labels[1], _COLOR_NON_TOP),
  ]:
    fig.add_trace(go.Bar(
      x=[label],
      y=[val],
      marker_color=color,
      text=[f"{val:,.1f}"],
      textposition="outside",
      cliponaxis=False,
      showlegend=False,
      hovertemplate=f"<b>{label}</b><br>{metric_name} Median: {val:,.2f}<extra></extra>",
    ), row=1, col=2)

  # Màu title subplot theo chiều hướng chênh lệch
  title_color_mean = _COLOR_POSITIVE if uplift_mean >= 0 else _COLOR_NEGATIVE
  title_color_med = _COLOR_POSITIVE if uplift_med >= 0 else _COLOR_NEGATIVE

  fig.update_layout(
    title=dict(
      text=f"So sánh {metric_name}: Có vs Không có huy hiệu Top Bán Chạy",
      x=0.5, xanchor="center",
    ),
    bargap=0.4,
    margin=dict(l=20, r=20, t=100, b=20),
    height=460,
    plot_bgcolor="white",
    yaxis=dict(gridcolor="#ebebeb", zeroline=False, title=metric_name),
    yaxis2=dict(gridcolor="#ebebeb", zeroline=False, title=metric_name),
  )

  # Tô màu subtitle theo chiều hướng
  fig.layout.annotations[0].font = dict(size=13, color=title_color_mean)
  fig.layout.annotations[1].font = dict(size=13, color=title_color_med)

  return fig


def plot_top100_median_comparison(
  overall_summary_df: pd.DataFrame,
  metric_name: str = "Số lượng bán",
) -> go.Figure:
  """Alias — trả về empty figure vì đã gộp vào plot_top100_overall_comparison."""
  return _empty_figure("Đã gộp vào biểu đồ Mean + Median phía trên.")


def plot_top100_by_category(
  category_summary_df: pd.DataFrame,
  top_n: int = 10,
) -> go.Figure:
  """Grouped horizontal bar — so sánh TB doanh số có/không Top theo danh mục."""
  if category_summary_df.empty:
    return _empty_figure("Không đủ dữ liệu theo danh mục để so sánh.")

  # Lấy top_n danh mục có uplift cao nhất, sắp xếp theo avg_top tăng dần để bar dài nhất trên cùng
  chart_df = category_summary_df.head(top_n).copy()
  chart_df = chart_df.sort_values("avg_top", ascending=True)

  fig = go.Figure()
  fig.add_trace(go.Bar(
    y=chart_df["category"],
    x=chart_df["avg_non_top"],
    orientation="h",
    name="Không có huy hiệu Top",
    marker=dict(color=_COLOR_NON_TOP, line=dict(color="white", width=1)),
    hovertemplate="<b>%{y}</b><br>Không Top: %{x:,.1f}<extra></extra>",
  ))
  fig.add_trace(go.Bar(
    y=chart_df["category"],
    x=chart_df["avg_top"],
    orientation="h",
    name="Có huy hiệu Top",
    marker=dict(color=_COLOR_TOP, line=dict(color="white", width=1)),
    hovertemplate="<b>%{y}</b><br>Có Top: %{x:,.1f}<extra></extra>",
  ))

  fig.update_layout(
    barmode="group",
    bargap=0.25,
    bargroupgap=0.05,
    title=dict(text=f"Doanh số TB theo danh mục — Top {top_n} chênh lệch cao nhất", x=0.5, xanchor="center"),
    xaxis_title="Số lượng bán trung bình",
    xaxis_tickformat=",",
    yaxis_title="",
    yaxis_automargin=True,
    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
    margin=dict(l=_LABEL_MARGIN_LEFT, r=20, t=55, b=70),
    height=max(420, top_n * 52),
    plot_bgcolor="white",
    xaxis=dict(gridcolor="#ebebeb", showgrid=True, zeroline=False),
    yaxis=dict(showgrid=False),
  )
  return fig


def plot_top100_uplift_by_category(
  category_summary_df: pd.DataFrame,
  top_n: int = 12,
) -> go.Figure:
  """Horizontal bar — mức chênh lệch % Top vs Không Top theo danh mục."""
  if category_summary_df.empty:
    return _empty_figure("Không đủ dữ liệu category.")

  chart_df = category_summary_df.head(top_n).copy()
  chart_df = chart_df.sort_values("uplift_pct", ascending=True)
  colors = [_COLOR_POSITIVE if v >= 0 else _COLOR_NEGATIVE for v in chart_df["uplift_pct"]]

  fig = go.Figure()
  fig.add_trace(go.Bar(
    y=chart_df["category"],
    x=chart_df["uplift_pct"],
    orientation="h",
    marker_color=colors,
    customdata=np.stack([chart_df["count_top"], chart_df["count_non_top"]], axis=1),
    hovertemplate=(
      "<b>%{y}</b><br>"
      "Chênh lệch: %{x:+.1f}%<br>"
      "Có Top: %{customdata[0]:,} SP | Không Top: %{customdata[1]:,} SP"
      "<extra></extra>"
    ),
    name="",
  ))

  fig.add_vline(x=50, line_dash="dot", line_color="#1E90FF", line_width=2,
                annotation_text="Ngưỡng 50%", annotation_position="top right",
                annotation_font_color="#1E90FF", annotation_font_size=11)
  fig.add_vline(x=0, line_color="#aaaaaa", line_width=1)

  fig.update_layout(
    title="Chênh lệch doanh số (%) — Có Top vs Không Top theo danh mục",
    xaxis_title="Chênh lệch %",
    yaxis_title="",
    yaxis_automargin=True,
    showlegend=False,
    margin=dict(l=_LABEL_MARGIN_LEFT, r=40, t=80, b=20),
    height=max(400, top_n * 46),
    plot_bgcolor="white",
    xaxis=dict(gridcolor="#ebebeb", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#ebebeb"),
  )
  return fig


def plot_gift_uplift(gift_impact_df: pd.DataFrame) -> go.Figure:
  if gift_impact_df.empty:
    return _empty_figure("Không có dữ liệu quà tặng/giá trị cộng thêm.")

  chart_df = gift_impact_df.copy()
  chart_df = chart_df.sort_values("uplift_pct", ascending=True)
  chart_df["color"] = np.where(chart_df["uplift_pct"] >= 0, "Tích cực", "Tiêu cực")
  chart_df["uplift_label"] = chart_df["uplift_pct"].map("{:+.1f}%".format)

  fig = px.bar(
    chart_df,
    x="uplift_pct",
    y="feature_label",
    orientation="h",
    color="color",
    color_discrete_map={"Tích cực": _COLOR_POSITIVE, "Tiêu cực": _COLOR_NEGATIVE},
    text="uplift_label",
    labels={
      "uplift_pct": "Chênh lệch % (Có quà vs Không có quà)",
      "feature_label": "",
    },
    custom_data=["count_with_feature", "support_ratio", "sufficient_support"],
    title="Mức chênh lệch doanh số theo từng loại quà tặng/giá trị cộng thêm",
  )
  fig.update_traces(
    textposition="outside",
    cliponaxis=False,
    hovertemplate="<b>%{y}</b><br>Chênh lệch: %{x:+.1f}%<br>"
    "Số mẫu có đặc trưng: %{customdata[0]:,}<br>"
    "Tỉ lệ xuất hiện: %{customdata[1]:.1%}<br>"
    "Đủ mẫu: %{customdata[2]}<extra></extra>",
  )
  fig.add_vline(
    x=20,
    line_dash="dash",
    line_color="#1E90FF",
    annotation_text="Ngưỡng 20%",
    annotation_position="top right",
    annotation_font_size=11,
  )
  fig.update_layout(
    legend_title_text="",
    yaxis_automargin=True,
    margin=dict(l=_LABEL_MARGIN_LEFT, r=60, t=70, b=20),
    height=max(420, len(chart_df) * 52),
  )
  return fig


def plot_gift_average_comparison(
  gift_impact_df: pd.DataFrame,
  top_n: int = 6,
) -> go.Figure:
  """Overlay horizontal bar — so sánh TB có/không có quà tặng."""
  if gift_impact_df.empty:
    return _empty_figure("Không có dữ liệu để so sánh trung bình.")

  chart_df = gift_impact_df.head(top_n).copy()
  chart_df = chart_df.sort_values("avg_with_feature", ascending=True)

  fig = go.Figure()
  fig.add_trace(go.Bar(
    y=chart_df["feature_label"],
    x=chart_df["avg_without_feature"],
    orientation="h",
    name="Không có quà tặng/bản đặc biệt",
    marker_color=_COLOR_NO_GIFT,
    opacity=0.9,
    hovertemplate="<b>%{y}</b><br>Không có: %{x:,.1f}<extra></extra>",
  ))
  fig.add_trace(go.Bar(
    y=chart_df["feature_label"],
    x=chart_df["avg_with_feature"],
    orientation="h",
    name="Có quà tặng/bản đặc biệt",
    marker_color=_COLOR_GIFT,
    opacity=0.85,
    hovertemplate="<b>%{y}</b><br>Có quà: %{x:,.1f}<extra></extra>",
  ))

  fig.update_layout(
    barmode="overlay",
    title="So sánh doanh số TB: Có vs Không có giá trị cộng thêm",
    xaxis_title="Số lượng bán trung bình",
    xaxis_tickformat=",",
    yaxis_title="",
    yaxis_automargin=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=_LABEL_MARGIN_LEFT, r=20, t=80, b=20),
    height=max(400, top_n * 58),
    plot_bgcolor="white",
    xaxis=dict(gridcolor="#ebebeb", showgrid=True),
    yaxis=dict(gridcolor="#ebebeb"),
  )
  return fig
