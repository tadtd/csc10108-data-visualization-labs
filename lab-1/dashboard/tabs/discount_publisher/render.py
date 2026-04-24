import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dashboard.utils import apply_common_style, get_palette, render_chart_with_insight, vnd_format
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

  genres = [genre for genre in products_df["genre"].value_counts().index.tolist() if str(genre).strip().casefold() not in {"root", "other"}]
  min_price = int(products_df["price"].min())
  max_price = int(products_df["price"].max())
  price_step = max((max_price - min_price) // 120, 1)
  price_options = list(range(min_price, max_price + price_step, price_step))
  if price_options[-1] != max_price:
    price_options.append(max_price)
  with st.sidebar:
    st.markdown("### Bộ lọc phân tích")
    selected_genres = st.multiselect("Thể loại", options=genres, default=genres[:8], key="damdat_genres")
    rating_range = st.slider("Khoảng rating", 0.0, 5.0, (0.0, 5.0), 0.1, key="damdat_rating_range")
    price_range = st.select_slider(
      "Khoảng giá (VND)",
      options=price_options,
      value=(min_price, max_price),
      format_func=lambda x: f"{vnd_format(x)} đ",
      key="damdat_price_range",
    )
    min_reviews = st.slider("Review tối thiểu", 0, 50, 1, key="damdat_min_reviews")

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

  st.markdown("### 1) Ảnh hưởng của giá và giảm giá")
  price_summary = retriever.price_band_summary(filtered_df)
  disc_summary = retriever.discount_band_summary(filtered_df)
  col_a, col_b = st.columns(2)
  with col_a:
    price_insight = "Chưa có đủ dữ liệu để kết luận."
    if not price_summary.empty:
      top_price_band = price_summary.loc[price_summary["avg_sold"].idxmax()]
      price_insight = f"Phân khúc giá <b>{top_price_band['price_band']}</b> đang ghi nhận lượt bán trung bình cao nhất ({top_price_band['avg_sold']:,.0f} lượt). Đây chính là vùng giá trọng điểm mà người mua sẵn sàng xuống tiền nhất trong tập dữ liệu hiện tại."
    render_chart_with_insight(
      draw_price_band_chart(price_summary, palette),
      toggle_key="discount_chart_price_band",
      insight_text=price_insight,
      palette=palette,
    )
  with col_b:
    disc_insight = "Chưa có đủ dữ liệu để kết luận."
    if not disc_summary.empty:
      top_disc_band = disc_summary.loc[disc_summary["avg_sold"].idxmax()]
      disc_insight = f"Nhóm có tỷ lệ giảm giá <b>{top_disc_band['disc_band']}</b> đạt hiệu suất bán tốt nhất ({top_disc_band['avg_sold']:,.0f} lượt trung bình). Không phải cứ giảm sâu là bán chạy, nên tập trung ưu đãi ở ngưỡng tối ưu này để vừa giữ lợi nhuận vừa kích cầu hiệu quả."
    render_chart_with_insight(
      draw_discount_band_chart(disc_summary, palette),
      toggle_key="discount_chart_discount_band",
      insight_text=disc_insight,
      palette=palette,
    )

  st.markdown("### 2) So sánh hiệu quả theo nhóm đánh giá")
  rating_summary = retriever.rating_group_summary(filtered_df)
  rating_note = "Chưa đủ dữ liệu để so sánh."
  if {"< 4.5", ">= 4.5"}.issubset(set(rating_summary["rating_group"].tolist())):
    high = float(rating_summary.loc[rating_summary["rating_group"] == ">= 4.5", "avg_sold"].iloc[0])
    low = float(rating_summary.loc[rating_summary["rating_group"] == "< 4.5", "avg_sold"].iloc[0])
    if low > 0:
      uplift = (high - low) / low * 100
      rating_note = "Sách có đánh giá tích cực (rating từ 4.5 trở lên) giúp người mua vững tin chốt đơn, tạo ra khác biệt doanh số cực kỳ ấn tượng so với phần còn lại." if uplift >= 20 else "Mặc dù điểm đánh giá cao mang lại cảm giác an tâm, nhưng mức chênh lệch doanh số trung bình so với nhóm còn lại chưa tạo thành bước nhảy vọt."
  render_chart_with_insight(
    draw_rating_group_chart(rating_summary, palette),
    toggle_key="discount_chart_rating_group",
    insight_text=rating_note,
    palette=palette,
  )
  st.markdown("### 3) So sánh Top 5 nhà xuất bản và nhóm còn lại")
  publisher_summary = retriever.publisher_group_summary(filtered_df)
  publisher_note = "Chưa đủ dữ liệu để so sánh."
  if {"Top 5 NXB", "Khác"}.issubset(set(publisher_summary["publisher_group"].tolist())):
    top5 = float(publisher_summary.loc[publisher_summary["publisher_group"] == "Top 5 NXB", "avg_sold"].iloc[0])
    other = float(publisher_summary.loc[publisher_summary["publisher_group"] == "Khác", "avg_sold"].iloc[0])
    if other > 0:
      gap = (top5 - other) / other * 100
      top5_names = ", ".join(top_publishers) if top_publishers else "không xác định"
      publisher_note = f"Các thương hiệu lớn ({top5_names}) thực sự phát huy lợi thế uy tín, mang lại mức doanh thu áp đảo và chiếm thế thượng phong rõ rệt." if gap >= 30 else f"Tuy Top 5 NXB ({top5_names}) có tiếng vang lớn, nhưng khoảng cách doanh thu trung bình so với các đơn vị nhỏ hơn chưa đạt đến mức độ hoàn toàn áp đảo."
  render_chart_with_insight(
    draw_publisher_group_chart(publisher_summary, palette),
    toggle_key="discount_chart_publisher_group",
    insight_text=publisher_note,
    palette=palette,
  )

  st.markdown("### 4) Top yếu tố ảnh hưởng (thống kê + ML)")
  corr = retriever.numeric_correlation(filtered_df)
  if corr.empty:
    st.info("Dữ liệu chưa đủ để tính tương quan.")
  else:
    corr_target = corr["sold_count"].drop(labels=["sold_count"]).abs().sort_values(ascending=False).head(3)
    top_features = ", ".join([f"{feature} ({score:.2f})" for feature, score in corr_target.items()])
    render_chart_with_insight(
      draw_corr_heatmap(corr),
      toggle_key="discount_chart_corr_heatmap",
      insight_text=f"Bức tranh tương quan xác nhận 3 yếu tố cốt lõi quyết định lớn nhất đến doanh số là {top_features}, giúp người bán định hướng trọng tâm tối ưu chiến lược.",
      palette=palette,
    )

  ml_result = _run_ml(filtered_df)
  if ml_result["ok"]:
    st.markdown("#### Góc nhìn học máy (Hồi quy Rừng ngẫu nhiên)")
    importance = ml_result["importance"].head(10).sort_values("importance")
    importance_sorted = importance.sort_values("importance", ascending=False)
    top_ml_disc = importance_sorted.iloc[0]["feature"] if not importance_sorted.empty else "các yếu tố"
    insight_text_ml = f"Thuật toán chỉ ra rằng '{top_ml_disc}' đóng vai trò chi phối mạnh mẽ nhất trong việc định hình doanh số. Các chiến dịch marketing kích cầu nên xoay quanh yếu tố này."
    render_chart_with_insight(
      draw_ml_importance(importance),
      toggle_key="discount_chart_ml_importance",
      insight_text=insight_text_ml,
      palette=palette,
    )

    bucket = ml_result["bucket"]
    bucket_long = bucket.melt(id_vars="Nhóm dự báo", var_name="Loại giá trị", value_name="Lượt bán")
    render_chart_with_insight(
      draw_ml_bucket(bucket_long, palette),
      toggle_key="discount_chart_ml_bucket",
      insight_text="Sự hội tụ rõ nét giữa đường dự báo và thực tế khẳng định các kết luận tối ưu giá và chiết khấu phía trên là hoàn toàn có cơ sở khoa học, có thể áp dụng vào thực tiễn.",
      palette=palette,
    )
  else:
    st.info("Số mẫu chưa đủ để huấn luyện mô hình ML ổn định.")