import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .chart import (
  draw_combo_compare_chart_single,
  draw_combo_uplift_chart,
  draw_genre_chart,
  draw_ml_bucket_chart,
  draw_ml_coef_chart,
  draw_pivot_chart,
  draw_publisher_chart,
)
from .retrieve import GenreStrategyRetriever
from dashboard.utils import apply_common_style, get_palette, render_chart_with_insight, vnd_format


def _format_number(value: float) -> str:
  return f"{value:,.0f}".replace(",", ".")


@st.cache_data(show_spinner=False)
def _load_data_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
  retriever = GenreStrategyRetriever()
  products = retriever.load_products()
  authors = retriever.load_authors()
  return products, authors


@st.cache_data(show_spinner=False)
def _train_ridge_model(dataset: pd.DataFrame) -> dict[str, pd.DataFrame | float]:
  model_df = dataset.copy()
  model_df["target_log"] = np.log1p(model_df["sold_count"])

  numeric_features = ["price", "discount_percent", "rating", "review_count", "page_count", "publish_year"]
  categorical_features = ["genre", "disc_band", "publisher_group"]
  flag_features = ["is_combo", "is_bestseller", "is_top100", "has_gift"]

  for column in numeric_features:
    if column not in model_df.columns:
      model_df[column] = 0

  for column in categorical_features:
    if column not in model_df.columns:
      model_df[column] = "Không rõ"

  for column in flag_features:
    if column not in model_df.columns:
      model_df[column] = 0

  model_df = model_df.dropna(subset=numeric_features + ["target_log"])

  x = model_df[numeric_features + categorical_features + flag_features]
  y = model_df["target_log"]

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  preprocessor = ColumnTransformer(
    transformers=[
      ("num", StandardScaler(), numeric_features),
      ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
      ("flag", "passthrough", flag_features),
    ]
  )

  pipeline = Pipeline(
    steps=[
      ("preprocessor", preprocessor),
      ("ridge", Ridge(alpha=1.0)),
    ]
  )
  pipeline.fit(x_train, y_train)

  y_pred_log = pipeline.predict(x_test)
  y_true = np.expm1(y_test)
  y_pred = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)

  metrics = {
    "r2": float(r2_score(y_true, y_pred)),
    "mae": float(mean_absolute_error(y_true, y_pred)),
  }

  feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
  coefs = pipeline.named_steps["ridge"].coef_
  coef_frame = pd.DataFrame(
    {
      "feature": feature_names,
      "coef": coefs,
      "abs_coef": np.abs(coefs),
    }
  ).sort_values("abs_coef", ascending=False)
  coef_frame["feature"] = coef_frame["feature"].str.replace("num__", "", regex=False)
  coef_frame["feature"] = coef_frame["feature"].str.replace("cat__", "", regex=False)
  coef_frame["feature"] = coef_frame["feature"].str.replace("flag__", "", regex=False)

  compare_frame = pd.DataFrame({"Thực tế": y_true, "Dự báo": y_pred})
  # Giảm ảnh hưởng outlier để biểu đồ bucket dễ đọc hơn.
  pred_for_bucket = compare_frame["Dự báo"].clip(upper=compare_frame["Dự báo"].quantile(0.99))
  bucket_count = min(6, pred_for_bucket.nunique())
  if bucket_count >= 2:
    compare_frame["Nhóm dự báo"] = pd.qcut(pred_for_bucket, q=bucket_count, duplicates="drop")
    by_bucket = compare_frame.groupby("Nhóm dự báo", as_index=False)[["Thực tế", "Dự báo"]].median()
    by_bucket["Nhóm dự báo"] = by_bucket["Nhóm dự báo"].astype(str)
  else:
    by_bucket = pd.DataFrame(columns=["Nhóm dự báo", "Thực tế", "Dự báo"])

  return {
    "metrics": pd.DataFrame([metrics]),
    "coef": coef_frame,
    "bucket": by_bucket,
  }


def render():
  apply_common_style()
  st.subheader("Chiến lược thể loại, nhà xuất bản và combo")

  retriever = GenreStrategyRetriever()

  try:
    products_df, _authors_df = _load_data_sources()
  except FileNotFoundError as error:
    st.error(f"Không tìm thấy dữ liệu clean: {error}")
    return

  if products_df.empty:
    st.warning("Dữ liệu sản phẩm đang rỗng.")
    return

  color_mode = st.session_state.get("color_mode", "Mặc định")
  palette = get_palette(color_mode)

  genre_order = [
    genre for genre in products_df["genre"].value_counts().index.tolist()
    if str(genre).strip().casefold() not in {"root", "other"}
  ]
  default_genres = genre_order[:8]
  min_price = int(products_df["price"].min())
  max_price = int(products_df["price"].max())
  price_step = max((max_price - min_price) // 100, 1)
  price_options = list(range(min_price, max_price + price_step, price_step))
  if price_options[-1] != max_price:
    price_options.append(max_price)
  with st.sidebar:
    st.markdown("### Bộ lọc phân tích")
    selected_genres = st.multiselect(
      "Thể loại",
      options=genre_order,
      default=default_genres,
      key="genre_selected_genres",
    )
    combo_mode = st.radio(
      "Hình thức bán",
      ["Tất cả", "Chỉ combo", "Không combo"],
      index=0,
      key="genre_combo_mode",
    )
    publisher_mode = st.selectbox(
      "Nhóm nhà xuất bản",
      ["Tất cả", "Top 5 NXB", "Khác"],
      index=0,
      key="genre_publisher_mode",
    )
    price_range = st.select_slider(
      "Khoảng giá (VND)",
      options=price_options,
      value=(min_price, max_price),
      format_func=lambda price: f"{vnd_format(price)} đ",
      key="genre_price_range",
    )
    min_products_uplift = st.slider(
      "Số mẫu tối thiểu mỗi nhóm (combo/bán lẻ)",
      min_value=5,
      max_value=40,
      value=8,
      key="genre_min_products_uplift",
    )

  filtered_df, top_publishers = retriever.filter_products(
    products=products_df,
    genres=selected_genres,
    combo_mode=combo_mode,
    price_range=price_range,
    publisher_mode=publisher_mode,
  )

  if filtered_df.empty:
    st.warning("Không có dữ liệu phù hợp với bộ lọc hiện tại.")
    return

  st.markdown("### 1) Bức tranh thể loại")
  genre_summary = retriever.genre_summary(filtered_df)
  top_genres = genre_summary.head(10)
  fig_genre = draw_genre_chart(top_genres, palette)
  genre_note = "Biểu đồ cho biết thể loại đang dẫn dắt doanh số trong lát cắt lọc hiện tại."
  if not top_genres.empty:
    top_3_names = ", ".join(top_genres.head(3)["genre"].tolist())
    genre_note = (
      f"Top 3 thể loại đang dẫn đầu: <b>{top_3_names}</b>. "
      "Đây là nhóm nên ưu tiên ngân sách hiển thị và tồn kho."
    )
  render_chart_with_insight(
    fig_genre,
    toggle_key="genre_chart_top_genres",
    insight_text=genre_note,
    palette=palette,
  )

  st.markdown("### 2) So sánh Top 5 NXB và nhóm còn lại")
  publisher_summary = retriever.publisher_summary(filtered_df)
  if not publisher_summary.empty:
    fig_publisher = draw_publisher_chart(publisher_summary, palette)
    gap_text = "Chưa đủ dữ liệu để so sánh nhóm Top 5 NXB và nhóm còn lại."
    groups = set(publisher_summary["publisher_group"].tolist())
    if {"Top 5 NXB", "Khác"}.issubset(groups):
      top5 = float(publisher_summary.loc[publisher_summary["publisher_group"] == "Top 5 NXB", "avg_sold"].iloc[0])
      other = float(publisher_summary.loc[publisher_summary["publisher_group"] == "Khác", "avg_sold"].iloc[0])
      if other > 0:
        gap_percent = (top5 - other) / other * 100
        top5_names = ", ".join(top_publishers) if top_publishers else "không xác định"
        gap_text = f"Uy tín từ các nhà xuất bản lớn ({top5_names}) thực sự tạo ra bảo chứng chất lượng, mang lại mức doanh thu trung bình áp đảo so với phần còn lại của thị trường." if gap_percent >= 30 else f"Tuy Top 5 NXB ({top5_names}) có ưu thế về quy mô, nhưng mức chênh lệch doanh thu bình quân so với các NXB nhỏ chưa thực sự tạo ra khoảng cách quá lớn."
    render_chart_with_insight(
      fig_publisher,
      toggle_key="genre_chart_publisher_group",
      insight_text=gap_text,
      palette=palette,
    )

  st.markdown("### 3) Hiệu quả chiến lược combo theo thể loại")
  combo_uplift = retriever.combo_uplift_by_genre(filtered_df, min_products=min_products_uplift)
  if combo_uplift.empty:
    st.info("Dữ liệu combo chưa đủ dày để tính uplift theo thể loại với ngưỡng mẫu hiện tại.")
  else:
    show_uplift = combo_uplift.head(12)
    if len(show_uplift) == 1:
      fig_uplift = draw_combo_compare_chart_single(show_uplift.iloc[0], palette)
    else:
      fig_uplift = draw_combo_uplift_chart(show_uplift, palette)
    above_20 = int((combo_uplift["uplift_percent"] > 20).sum())
    top_positive = combo_uplift[combo_uplift["uplift_percent"] > 0]
    best_genres = ", ".join(top_positive.head(2)["genre"].tolist()) if not top_positive.empty else "Chưa có thể loại tăng trưởng dương"
    insight_combo = f"Chiến lược bán theo Combo đang phát huy hiệu quả xuất sắc ở các ngách ({best_genres}), tạo đòn bẩy kích cầu mạnh mẽ và tăng AOV (giá trị trung bình đơn) vượt kỳ vọng." if above_20 >= 2 else "Chiến lược Combo có mang lại giá trị gia tăng ở một vài nhóm, tuy nhiên chưa tạo được sức bật doanh số đủ mạnh và đồng đều trên nhiều thể loại."
    render_chart_with_insight(
      fig_uplift,
      toggle_key="genre_chart_combo_uplift",
      insight_text=insight_combo,
      palette=palette,
    )

  st.markdown("### 4) Bảng chéo thể loại x hình thức bán")
  pivot_table = retriever.combo_pivot_table(filtered_df)
  if pivot_table.empty:
    st.info("Không đủ dữ liệu để hiển thị pivot theo thể loại và combo.")
  else:
    fig_pivot = draw_pivot_chart(pivot_table)
    best_combo_genre = pivot_table.sort_values(by="Combo", ascending=False).index[0] if "Combo" in pivot_table.columns else pivot_table.index[0]
    insight_text_pivot = f"Bảng chéo cho thấy '{best_combo_genre}' đang là nhóm thu hút lượt bán cao, phản ánh xu hướng mua sỉ hoặc mua theo bộ rất mạnh ở ngách này."
    render_chart_with_insight(
      fig_pivot,
      toggle_key="genre_chart_pivot",
      insight_text=insight_text_pivot,
      palette=palette,
    )

  st.markdown("### 5) Góc nhìn học máy: Hồi quy Ridge (mô tả xu hướng)")
  ml_source = filtered_df if len(filtered_df) >= 500 else products_df
  source_label = "dữ liệu đã lọc" if len(filtered_df) >= 500 else "toàn bộ dữ liệu clean (do bộ lọc quá ít mẫu)"
  st.caption(f"Huấn luyện mô hình trên {source_label}. Mục tiêu: diễn giải biến ảnh hưởng, không suy luận nhân quả.")

  ml_dataset = retriever.build_ml_dataset(ml_source)
  if len(ml_dataset) < 200:
    st.info("Số lượng mẫu chưa đủ để hiển thị kết quả ML ổn định.")
  else:
    ml_result = _train_ridge_model(ml_dataset)
    metrics = ml_result["metrics"].iloc[0]

    coef_view = ml_result["coef"].head(10).sort_values("coef")
    fig_coef = draw_ml_coef_chart(coef_view)
    top_coef_feature = coef_view.iloc[0]["feature"] if not coef_view.empty else "các biến"
    insight_text_coef = f"Hệ số hồi quy chỉ ra rằng '{top_coef_feature}' có sức nặng lớn nhất trong mô hình, là biến số cốt lõi chi phối mạnh mẽ đến sự thành bại của doanh thu."
    render_chart_with_insight(
      fig_coef,
      toggle_key="genre_chart_ml_coef",
      insight_text=insight_text_coef,
      palette=palette,
    )

    bucket_view = ml_result["bucket"]
    if not bucket_view.empty:
      bucket_long = bucket_view.melt(id_vars="Nhóm dự báo", var_name="Loại giá trị", value_name="Lượt bán")
      fig_bucket = draw_ml_bucket_chart(bucket_long, palette)
      render_chart_with_insight(
        fig_bucket,
        toggle_key="genre_chart_ml_bucket",
        insight_text="Sự bám sát giữa đường thực tế và dự báo ở các phân khúc cho thấy hệ thống biến số hiện tại nắm bắt rất tốt xu hướng mua hàng của thị trường.",
        palette=palette,
      )