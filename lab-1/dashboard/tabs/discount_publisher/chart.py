import pandas as pd
import plotly.express as px


def draw_price_band_chart(summary: pd.DataFrame, palette: list[str]):
  return px.bar(
    summary,
    x="price_band",
    y="avg_sold",
    color_discrete_sequence=[palette[0]],
    labels={"price_band": "Nhóm giá", "avg_sold": "Lượt bán trung bình"},
    title="Lượt bán trung bình theo nhóm giá",
  )


def draw_discount_band_chart(summary: pd.DataFrame, palette: list[str]):
  return px.bar(
    summary,
    x="disc_band",
    y="avg_sold",
    color_discrete_sequence=[palette[1]],
    labels={"disc_band": "Nhóm giảm giá", "avg_sold": "Lượt bán trung bình"},
    title="Lượt bán trung bình theo mức giảm giá",
  )


def draw_rating_group_chart(summary: pd.DataFrame, palette: list[str]):
  return px.bar(
    summary,
    x="rating_group",
    y="avg_sold",
    color="rating_group",
    color_discrete_sequence=[palette[2], palette[0]],
    labels={"rating_group": "Nhóm đánh giá", "avg_sold": "Lượt bán trung bình"},
    title="So sánh nhóm đánh giá >=4.5 và <4.5",
  )


def draw_publisher_group_chart(summary: pd.DataFrame, palette: list[str]):
  return px.bar(
    summary,
    x="publisher_group",
    y="avg_sold",
    color="publisher_group",
    color_discrete_sequence=[palette[3], palette[1]],
    labels={"publisher_group": "Nhóm nhà xuất bản", "avg_sold": "Lượt bán trung bình"},
    title="So sánh Top 5 NXB và nhóm còn lại",
  )


def draw_corr_heatmap(corr: pd.DataFrame):
  return px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="Blues",
    aspect="auto",
    title="Ma trận tương quan giữa biến giá/đánh giá/số review và lượt bán",
  )


def draw_ml_importance(importance: pd.DataFrame):
  return px.bar(
    importance,
    x="importance",
    y="feature",
    orientation="h",
    color_discrete_sequence=["#2ca02c"],
    labels={"importance": "Mức quan trọng", "feature": "Biến"},
    title="Nhóm biến quan trọng theo mô hình Random forrest",
  )


def draw_ml_bucket(bucket_long: pd.DataFrame, palette: list[str]):
  return px.bar(
    bucket_long,
    x="Nhóm dự báo",
    y="Lượt bán",
    color="Loại giá trị",
    barmode="group",
    color_discrete_sequence=[palette[0], palette[1]],
    title="So sánh thực tế và dự báo theo nhóm sản phẩm",
  )
