import pandas as pd
import plotly.express as px


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
  figure = px.bar(
    show_uplift,
    x="genre",
    y="uplift_percent",
    color_discrete_sequence=[palette[3]],
    labels={"genre": "Thể loại", "uplift_percent": "Mức tăng lượt bán khi bán combo (%)"},
    title="Combo uplift theo thể loại",
  )
  figure.add_hline(y=20, line_dash="dash", line_color="red")
  return figure


def draw_pivot_chart(pivot_table: pd.DataFrame):
  return px.imshow(
    pivot_table,
    text_auto=".1f",
    color_continuous_scale="Blues",
    labels={"color": "Lượt bán trung bình"},
    aspect="auto",
    title="Pivot: lượt bán trung bình theo thể loại và hình thức bán",
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
    title="Top biến ảnh hưởng theo mô hình Ridge",
  )


def draw_ml_bucket_chart(bucket_long: pd.DataFrame, palette: list[str]):
  return px.bar(
    bucket_long,
    x="Nhóm dự báo",
    y="Lượt bán",
    color="Loại giá trị",
    barmode="group",
    color_discrete_sequence=[palette[0], palette[1]],
    title="So sánh thực tế và dự báo theo nhóm sản phẩm",
  )
