import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dashboard.utils import apply_common_style, get_palette, vnd_format
from .chart import (
  draw_corr_heatmap,
  draw_discount_band_chart,
  draw_ml_bucket,
  draw_ml_importance,
  draw_price_band_chart,
  draw_publisher_group_chart,
  draw_rating_group_chart,
)
from .retrieve import DiscountPublisherRetriever


def _format_number(value: float) -> str:
  return f"{value:,.0f}".replace(",", ".")


@st.cache_data(show_spinner=False)
def _load_products() -> pd.DataFrame:
  return DiscountPublisherRetriever().load_products()


@st.cache_data(show_spinner=False)
def _run_ml(df: pd.DataFrame) -> dict[str, pd.DataFrame | float]:
  model_df = df.copy()
  model_df = model_df.dropna(subset=["sold_count", "price", "discount_percent", "rating", "review_count"])
  if len(model_df) < 200:
    return {"ok": False}

  features = ["price", "discount_percent", "rating", "review_count", "genre", "publisher_group"]
  x = model_df[features]
  y = model_df["sold_count"]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  prep = ColumnTransformer(
    transformers=[
      ("num", "passthrough", ["price", "discount_percent", "rating", "review_count"]),
      ("cat", OneHotEncoder(handle_unknown="ignore"), ["genre", "publisher_group"]),
    ]
  )
  model = Pipeline(
    steps=[
      ("prep", prep),
      ("rf", RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
    ]
  )
  model.fit(x_train, y_train)
  pred = model.predict(x_test)
  metrics = {"r2": float(r2_score(y_test, pred)), "mae": float(mean_absolute_error(y_test, pred))}

  feature_names = model.named_steps["prep"].get_feature_names_out()
  importances = model.named_steps["rf"].feature_importances_
  importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
    "importance", ascending=False
  )
  importance_df["feature"] = importance_df["feature"].str.replace("num__", "", regex=False)
  importance_df["feature"] = importance_df["feature"].str.replace("cat__", "", regex=False)

  compare_df = pd.DataFrame({"Thực tế": y_test, "Dự báo": pred})
  compare_df["Nhóm dự báo"] = pd.qcut(compare_df["Dự báo"], q=6, duplicates="drop")
  bucket = compare_df.groupby("Nhóm dự báo", as_index=False)[["Thực tế", "Dự báo"]].mean()
  bucket["Nhóm dự báo"] = bucket["Nhóm dự báo"].astype(str)
  return {"ok": True, "metrics": metrics, "importance": importance_df, "bucket": bucket}


def render():
  apply_common_style()
  st.subheader("Giá, giảm giá, đánh giá và nhà xuất bản")

  retriever = DiscountPublisherRetriever()
  color_mode = st.session_state.get("color_mode", "Mặc định")
  palette = get_palette(color_mode)

  try:
    products_df = _load_products()
  except FileNotFoundError as error:
    st.error(f"Không tìm thấy dữ liệu clean: {error}")
    return

  if products_df.empty:
    st.warning("Dữ liệu sản phẩm đang rỗng.")
    return

  st.sidebar.markdown("### Bộ lọc tab Giá & NXB")
  genres = [genre for genre in products_df["genre"].value_counts().index.tolist() if str(genre).strip().casefold() not in {"root", "other"}]
  selected_genres = st.sidebar.multiselect("Thể loại (tab Đàm Đạt)", options=genres, default=genres[:8], key="damdat_genres")

  min_price = int(products_df["price"].min())
  max_price = int(products_df["price"].max())
  price_step = max((max_price - min_price) // 120, 1)
  price_options = list(range(min_price, max_price + price_step, price_step))
  if price_options[-1] != max_price:
    price_options.append(max_price)
  price_range = st.sidebar.select_slider(
    "Khoảng giá (VND) - tab Đàm Đạt",
    options=price_options,
    value=(min_price, max_price),
    format_func=lambda x: f"{vnd_format(x)} đ",
    key="damdat_price_range",
  )
  rating_range = st.sidebar.slider("Khoảng rating", 0.0, 5.0, (0.0, 5.0), 0.1, key="damdat_rating_range")
  min_reviews = st.sidebar.slider("Review tối thiểu", 0, 50, 1, key="damdat_min_reviews")

  filtered_df, top_publishers = retriever.filter_products(
    products_df,
    genres=selected_genres,
    price_range=price_range,
    rating_range=rating_range,
    min_reviews=min_reviews,
  )
  if filtered_df.empty:
    st.warning("Không có dữ liệu phù hợp bộ lọc hiện tại.")
    return

  kpis = retriever.compute_kpis(filtered_df)
  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Số sản phẩm", _format_number(kpis["total_products"]))
  c2.metric("Tổng lượt bán", _format_number(kpis["total_sold"]))
  c3.metric("Doanh thu ước tính", f"{vnd_format(kpis['total_revenue'])} đ")
  c4.metric("Giảm giá trung bình", f"{kpis['avg_discount']:.1f}%")
  if top_publishers:
    st.caption("Top 5 nhà xuất bản theo dữ liệu đã lọc: " + ", ".join(top_publishers))

  st.markdown("### 1) Ảnh hưởng của giá và giảm giá")
  price_summary = retriever.price_band_summary(filtered_df)
  disc_summary = retriever.discount_band_summary(filtered_df)
  col_a, col_b = st.columns(2)
  col_a.plotly_chart(draw_price_band_chart(price_summary, palette), width="stretch")
  col_b.plotly_chart(draw_discount_band_chart(disc_summary, palette), width="stretch")

  st.markdown("### 2) Kiểm tra ngưỡng SMART: rating >= 4.5 cao hơn >=20%?")
  rating_summary = retriever.rating_group_summary(filtered_df)
  st.plotly_chart(draw_rating_group_chart(rating_summary, palette), width="stretch")
  rating_note = "Chưa đủ dữ liệu để so sánh."
  if {"< 4.5", ">= 4.5"}.issubset(set(rating_summary["rating_group"].tolist())):
    high = float(rating_summary.loc[rating_summary["rating_group"] == ">= 4.5", "avg_sold"].iloc[0])
    low = float(rating_summary.loc[rating_summary["rating_group"] == "< 4.5", "avg_sold"].iloc[0])
    if low > 0:
      uplift = (high - low) / low * 100
      status = "đạt" if uplift >= 20 else "chưa đạt"
      rating_note = f"Nhóm rating >= 4.5 cao hơn {uplift:.1f}% so với nhóm còn lại, {status} ngưỡng SMART 20%."
  st.markdown(f"<div class='insight-box'>{rating_note}</div>", unsafe_allow_html=True)

  st.markdown("### 3) Kiểm tra ngưỡng SMART: Top 5 NXB cao hơn >=30%?")
  publisher_summary = retriever.publisher_group_summary(filtered_df)
  st.plotly_chart(draw_publisher_group_chart(publisher_summary, palette), width="stretch")
  publisher_note = "Chưa đủ dữ liệu để so sánh."
  if {"Top 5 NXB", "Khác"}.issubset(set(publisher_summary["publisher_group"].tolist())):
    top5 = float(publisher_summary.loc[publisher_summary["publisher_group"] == "Top 5 NXB", "avg_sold"].iloc[0])
    other = float(publisher_summary.loc[publisher_summary["publisher_group"] == "Khác", "avg_sold"].iloc[0])
    if other > 0:
      gap = (top5 - other) / other * 100
      status = "đạt" if gap >= 30 else "chưa đạt"
      publisher_note = f"Top 5 NXB cao hơn {gap:.1f}% so với nhóm còn lại, {status} ngưỡng SMART 30%."
  st.markdown(f"<div class='insight-box'>{publisher_note}</div>", unsafe_allow_html=True)

  st.markdown("### 4) Top yếu tố ảnh hưởng (thống kê + ML)")
  corr = retriever.numeric_correlation(filtered_df)
  if corr.empty:
    st.info("Dữ liệu chưa đủ để tính tương quan.")
  else:
    st.plotly_chart(draw_corr_heatmap(corr), width="stretch")
    corr_target = corr["sold_count"].drop(labels=["sold_count"]).abs().sort_values(ascending=False).head(3)
    top_features = ", ".join([f"{feature} ({score:.2f})" for feature, score in corr_target.items()])
    st.markdown(f"<div class='insight-box'>Top 3 biến tương quan mạnh với lượt bán: <b>{top_features}</b>.</div>", unsafe_allow_html=True)

  ml_result = _run_ml(filtered_df)
  if ml_result["ok"]:
    st.markdown("#### ML insight (RandomForest Regressor)")
    m1, m2 = st.columns(2)
    m1.metric("R²", f"{ml_result['metrics']['r2']:.3f}")
    m2.metric("MAE", _format_number(ml_result["metrics"]["mae"]))
    importance = ml_result["importance"].head(10).sort_values("importance")
    st.plotly_chart(draw_ml_importance(importance), width="stretch")

    bucket = ml_result["bucket"]
    bucket_long = bucket.melt(id_vars="Nhóm dự báo", var_name="Loại giá trị", value_name="Lượt bán")
    st.plotly_chart(draw_ml_bucket(bucket_long, palette), width="stretch")
    st.caption("ML dùng để diễn giải xu hướng ảnh hưởng biến, không khẳng định quan hệ nhân quả.")
  else:
    st.info("Số mẫu chưa đủ để huấn luyện mô hình ML ổn định.")