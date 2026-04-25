import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def draw_genre_chart(top_genres: pd.DataFrame, palette: list[str]):
  figure = px.bar(
    top_genres,
    x="total_sold",
    y="genre",
    orientation="h",
    color_discrete_sequence=[palette[0]],
    labels={"total_sold": "Tổng lượt bán", "genre": "Thể loại"},
    title="Top thể loại theo tổng lượt bán",
  )
  figure.update_layout(yaxis={"categoryorder": "total ascending"})
  return figure


def draw_publisher_chart(publisher_summary: pd.DataFrame, palette: list[str]):
  return px.bar(
    publisher_summary,
    x="publisher_group",
    y="avg_sold",
    color="publisher_group",
    color_discrete_sequence=[palette[1], palette[2]],
    labels={"publisher_group": "Nhóm nhà xuất bản", "avg_sold": "Lượt bán trung bình/sản phẩm"},
    title="Lượt bán trung bình theo nhóm nhà xuất bản",
  )


def draw_combo_uplift_chart(show_uplift: pd.DataFrame, palette: list[str]):
  chart_df = show_uplift.copy()
  chart_df = chart_df[chart_df["uplift_percent"].notna()].sort_values("uplift_percent", ascending=False)
  figure = px.bar(
    chart_df,
    x="genre",
    y="uplift_percent",
    color_discrete_sequence=[palette[3]],
    labels={"genre": "Thể loại", "uplift_percent": "Mức tăng lượt bán khi bán combo (%)"},
    title="Mức chênh lệch doanh số khi bán combo theo thể loại",
  )
  figure.add_hline(y=20, line_dash="dash", line_color="red")
  figure.update_traces(
    text=chart_df["uplift_percent"].map(lambda x: f"{x:.1f}%"),
    textposition="outside",
    cliponaxis=False,
  )
  figure.update_layout(xaxis={"categoryorder": "total descending"})
  return figure


def draw_combo_compare_chart_single(genre_row: pd.Series, palette: list[str]):
  genre_name = str(genre_row["genre"])
  non_combo = float(genre_row["combo_0"])
  combo = float(genre_row["combo_1"])
  uplift = float(genre_row["uplift_percent"])

  figure = go.Figure()
  figure.add_trace(
    go.Bar(
      x=["Bán lẻ", "Combo"],
      y=[non_combo, combo],
      marker_color=[palette[1], palette[3]],
      text=[f"{non_combo:.1f}", f"{combo:.1f}"],
      textposition="outside",
      cliponaxis=False,
    )
  )
  figure.update_layout(
    title=f"So sánh trực tiếp Bán lẻ và Combo ({genre_name})",
    yaxis_title="Lượt bán trung bình",
    xaxis_title="Hình thức bán",
    showlegend=False,
  )
  figure.add_annotation(
    x=0.5,
    y=max(non_combo, combo) * 1.08 if max(non_combo, combo) > 0 else 0,
    xref="x domain",
    yref="y",
    text=f"Mức chênh lệch: {uplift:.1f}%",
    showarrow=False,
    font={"size": 13},
  )
  return figure


def draw_pivot_chart(pivot_table: pd.DataFrame):
  return px.imshow(
    pivot_table,
    text_auto=".1f",
    color_continuous_scale="Blues",
    labels={"color": "Lượt bán trung bình"},
    aspect="auto",
    title="Bảng chéo thể loại: lượt bán trung bình theo hình thức bán",
  )


def draw_ml_coef_chart(coef_view: pd.DataFrame):
  return px.bar(
    coef_view,
    x="coef",
    y="feature",
    orientation="h",
    color="coef",
    color_continuous_scale="RdBu",
    labels={"coef": "Hệ số tác động", "feature": "Biến"},
    title="Top biến ảnh hưởng theo mô hình Ridge Regression",
  )


def draw_ml_feature_validation_chart(validation_df: pd.DataFrame, feature_label: str, palette: list[str]):
  figure = px.line(
    validation_df,
    x="Nhóm giá trị",
    y="Lượt bán trung vị",
    markers=True,
    color_discrete_sequence=[palette[0]],
    labels={
      "Nhóm giá trị": f"Nhóm giá trị của {feature_label}",
      "Lượt bán trung vị": "Lượt bán trung vị (thực tế)",
    },
    title=f"Đối chiếu thực tế: {feature_label} và lượt bán",
  )
  figure.update_traces(mode="lines+markers+text", text=validation_df["Lượt bán trung vị"].map(lambda x: f"{x:,.1f}"), textposition="top center")
  return figure


def draw_ml_feature_validation_compact_chart(validation_df: pd.DataFrame, feature_label: str, palette: list[str]):
  figure = px.bar(
    validation_df,
    x="Nhóm giá trị",
    y="Lượt bán trung vị",
    color_discrete_sequence=[palette[1]],
    labels={
      "Nhóm giá trị": f"Nhóm giá trị của {feature_label}",
      "Lượt bán trung vị": "Lượt bán trung vị (thực tế)",
    },
    title=f"Đối chiếu nhanh thực tế: {feature_label} và lượt bán",
  )
  figure.update_traces(
    text=validation_df["Lượt bán trung vị"].map(lambda x: f"{x:,.1f}"),
    textposition="outside",
    cliponaxis=False,
  )
  return figure


def draw_ml_error_bucket_chart(error_df: pd.DataFrame, palette: list[str]):
  figure = px.bar(
    error_df,
    x="Nhóm dự báo",
    y="Sai lệch tuyệt đối (%)",
    color_discrete_sequence=[palette[2]],
    title="Mức sai lệch giữa dự báo và thực tế theo nhóm lượt bán",
    labels={
      "Nhóm dự báo": "Nhóm lượt bán dự báo",
      "Sai lệch tuyệt đối (%)": "Sai lệch tuyệt đối (%)",
    },
  )
  # Vẽ đường ngưỡng chạy toàn bộ chiều ngang biểu đồ.
  figure.add_shape(
    type="line",
    xref="paper",
    x0=0,
    x1=1,
    yref="y",
    y0=20,
    y1=20,
    line={"color": "red", "dash": "dash", "width": 2},
  )
  # Trace dummy để hiện legend cho đường ngưỡng.
  figure.add_trace(
    go.Scatter(
      x=[None],
      y=[None],
      mode="lines",
      line={"color": "red", "dash": "dash", "width": 2},
      name="Ngưỡng tham chiếu (20%)",
      hoverinfo="skip",
      showlegend=True,
    )
  )
  figure.update_traces(
    text=error_df["Sai lệch tuyệt đối (%)"].map(lambda x: f"{x:.1f}%"),
    textposition="outside",
    cliponaxis=False,
    selector={"type": "bar"},
  )
  return figure
