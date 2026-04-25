"""
Layout tab Overview: Filter → KPI → Main 65/35 → Insights.
Khung bố cục theo plan; biểu đồ dùng dữ liệu đã lọc tối thiểu để kiểm tra tương tác.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.tabs.discount_publisher.retrieve import DiscountPublisherRetriever
from dashboard.utils import apply_common_style, get_palette, render_chart_with_insight, vnd_format


def _overview_styles() -> None:
  st.markdown(
    """
    <style>
      div[data-testid="stVerticalBlock"] > div:has(> div.overview-kpi-row) {
        gap: 0.75rem;
      }
      @media (max-width: 768px) {
        .overview-main-split [data-testid="column"] {
          min-width: 100% !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
  )


@st.cache_data(show_spinner=False)
def _load_products() -> pd.DataFrame:
  return DiscountPublisherRetriever().load_products()


def _filter_products(
  df: pd.DataFrame,
  genres: list[str],
  price_min: float,
  price_max: float,
) -> pd.DataFrame:
  out = df.copy()
  if genres:
    out = out[out["genre"].isin(genres)]
  if "price" in out.columns:
    out = out[(out["price"] >= price_min) & (out["price"] <= price_max)]
  return out


def _kpi_snapshot(df: pd.DataFrame) -> dict[str, float | int]:
  n = len(df)
  total_sold = float(df["sold_count"].sum()) if "sold_count" in df.columns else 0.0
  avg_rating = float(df["rating"].mean()) if n and "rating" in df.columns else 0.0
  avg_price = float(df["price"].mean()) if n and "price" in df.columns else 0.0
  return {"n_products": n, "total_sold": total_sold, "avg_rating": avg_rating, "avg_price": avg_price}


def _fig_top_genres(df: pd.DataFrame, palette: list[str], top_n: int) -> go.Figure:
  if df.empty or "genre" not in df.columns or "sold_count" not in df.columns:
    fig = go.Figure()
    fig.add_annotation(
      text="Không đủ dữ liệu để vẽ biểu đồ",
      xref="paper",
      yref="paper",
      x=0.5,
      y=0.5,
      showarrow=False,
    )
    fig.update_layout(title="Thể loại — tổng lượt bán (placeholder)", height=320)
    return fig
  g = (
    df.groupby("genre", as_index=False)["sold_count"]
    .sum()
    .sort_values("sold_count", ascending=False)
    .head(max(1, top_n))
  )
  fig = px.bar(
    g,
    x="genre",
    y="sold_count",
    color_discrete_sequence=[palette[0]],
    labels={"sold_count": "Tổng lượt bán", "genre": "Thể loại"},
    title="Thể loại — tổng lượt bán (tương tác: Top-N)",
  )
  fig.update_layout(height=320, showlegend=False)
  return fig


def _fig_price_distribution(df: pd.DataFrame, palette: list[str]) -> go.Figure:
  if df.empty or "price" not in df.columns:
    fig = go.Figure()
    fig.add_annotation(
      text="Không có cột giá",
      xref="paper",
      yref="paper",
      x=0.5,
      y=0.5,
      showarrow=False,
    )
    fig.update_layout(title="Phân bố giá (placeholder)", height=280)
    return fig
  fig = px.histogram(
    df,
    x="price",
    nbins=30,
    color_discrete_sequence=[palette[1]],
    labels={"price": "Giá (VND)", "count": "Số sản phẩm"},
    title="Phân bố giá trong lát cắt đã lọc",
  )
  fig.update_layout(height=280, showlegend=False)
  return fig


def _fig_rating_group_sold(df: pd.DataFrame, palette: list[str]) -> go.Figure:
  """Trung bình lượt bán theo nhóm rating (tránh scatter với trục integer)."""
  if df.empty or "sold_count" not in df.columns:
    fig = go.Figure()
    fig.add_annotation(
      text="Thiếu sold_count",
      xref="paper",
      yref="paper",
      x=0.5,
      y=0.5,
      showarrow=False,
    )
    fig.update_layout(title="Nhóm đánh giá — lượt bán TB (placeholder)", height=260)
    return fig
  group_col = "rating_group" if "rating_group" in df.columns else None
  if group_col is None:
    fig = go.Figure()
    fig.add_annotation(
      text="Thiếu rating_group",
      xref="paper",
      yref="paper",
      x=0.5,
      y=0.5,
      showarrow=False,
    )
    fig.update_layout(title="Nhóm đánh giá — lượt bán TB (placeholder)", height=260)
    return fig
  g = df.groupby(group_col, as_index=False)["sold_count"].mean().sort_values(group_col)
  fig = px.bar(
    g,
    x=group_col,
    y="sold_count",
    color_discrete_sequence=[palette[2]],
    labels={"sold_count": "Lượt bán trung bình", group_col: "Nhóm đánh giá"},
    title="Nhóm đánh giá — lượt bán trung bình",
  )
  fig.update_layout(height=260, showlegend=False)
  return fig


def _fig_publisher_share(df: pd.DataFrame, palette: list[str]) -> go.Figure:
  if df.empty or "publisher_group" not in df.columns or "sold_count" not in df.columns:
    fig = go.Figure()
    fig.add_annotation(
      text="Thiếu nhóm NXB",
      xref="paper",
      yref="paper",
      x=0.5,
      y=0.5,
      showarrow=False,
    )
    fig.update_layout(title="Nhóm NXB — tỷ trọng bán (placeholder)", height=260)
    return fig
  g = df.groupby("publisher_group", as_index=False)["sold_count"].sum()
  total = g["sold_count"].sum()
  if total <= 0:
    fig = go.Figure()
    fig.update_layout(title="Nhóm NXB — tỷ trọng bán (placeholder)", height=260)
    return fig
  g["pct"] = g["sold_count"] / total * 100
  g = g.sort_values("pct", ascending=False).head(8)
  fig = px.pie(
    g,
    names="publisher_group",
    values="sold_count",
    color_discrete_sequence=palette,
    title="Top nhóm NXB theo tổng lượt bán",
  )
  fig.update_layout(height=260, showlegend=True)
  return fig


def render() -> None:
  apply_common_style()
  _overview_styles()

  # st.subheader("Tổng quan")
  # st.caption(
  #   "Bức tranh tổng thể về các yếu tố ảnh hưởng đến hiệu quả bán hàng sách trên Tiki."
  # )

  try:
    raw = _load_products()
  except FileNotFoundError:
    raw = pd.DataFrame()
  except Exception as e:  # noqa: BLE001
    st.error(f"Không đọc được dữ liệu: {e}")
    raw = pd.DataFrame()

  genre_options = sorted(raw["genre"].dropna().unique().tolist()) if not raw.empty and "genre" in raw.columns else []
  p_min = float(raw["price"].min()) if not raw.empty and "price" in raw.columns else 0.0
  p_max = float(raw["price"].max()) if not raw.empty and "price" in raw.columns else 1.0
  if p_max <= p_min:
    p_max = p_min + 1.0

  color_mode = st.session_state.get("color_mode", "Mặc định")
  palette = get_palette(color_mode)

  with st.sidebar:
    st.markdown("### Bộ lọc phân tích")
    genres_sel = st.multiselect(
      "Thể loại",
      options=genre_options,
      default=[],
      key="overview_genres_sel",
      help="Để trống = tất cả thể loại.",
    )
    price_range = st.slider(
      "Khoảng giá (VND)",
      min_value=p_min,
      max_value=p_max,
      value=(p_min, p_max),
      key="overview_price_range",
      format="%d",
    )
    top_n = st.number_input(
      "Top-N thể loại",
      min_value=3,
      max_value=30,
      value=10,
      step=1,
      key="overview_top_n",
    )

  filtered = _filter_products(raw, genres_sel, price_range[0], price_range[1])
  if not filtered.empty:
    filtered, _ = DiscountPublisherRetriever().with_publisher_group(filtered)

  kpi = _kpi_snapshot(filtered)
  st.markdown("### Tổng quan")
  kpi_col_1, kpi_col_2, kpi_col_3, kpi_col_4 = st.columns(4)
  with kpi_col_1:
    st.metric("Số sản phẩm", f"{int(kpi['n_products']):,}".replace(",", "."))
  with kpi_col_2:
    st.metric("Tổng lượt bán", f"{int(kpi['total_sold']):,}".replace(",", "."))
  with kpi_col_3:
    st.metric("Rating TB", f"{kpi['avg_rating']:.2f}")
  with kpi_col_4:
    st.metric("Giá TB", f"{vnd_format(kpi['avg_price'])} đ")



  st.markdown("### 1) Tổng quan Thể loại và Phân bố giá")
  col_a, col_b = st.columns(2)
  with col_a:
    top_genre = filtered.groupby("genre")["sold_count"].sum().idxmax() if not filtered.empty and "genre" in filtered.columns else "nhóm dẫn đầu"
    insight_text_genres = f"Nhìn vào biểu đồ có thể thấy '{top_genre}' đang là nhóm thể loại đóng góp doanh số lớn nhất, là điểm sáng cần tập trung khai thác và phân bổ ngân sách hiển thị."
    render_chart_with_insight(
      _fig_top_genres(filtered, palette, int(top_n)),
      toggle_key="overview_chart_top_genres",
      insight_text=insight_text_genres,
      palette=palette,
    )
  with col_b:
    render_chart_with_insight(
      _fig_price_distribution(filtered, palette),
      toggle_key="overview_chart_price_distribution",
      insight_text="Cột cao nhất trên biểu đồ chỉ ra vùng giá 'ngọt' (sweet spot) đang có nhiều sản phẩm và thanh khoản tốt nhất, giúp nhà bán hàng tự tin định giá sản phẩm mới.",
      palette=palette,
    )

  st.markdown("### 2) Nhóm đánh giá và Nhóm NXB")
  col_c, col_d = st.columns(2)
  with col_c:
    top_rating_group = filtered.groupby("rating_group")["sold_count"].mean().idxmax() if "rating_group" in filtered.columns and not filtered.empty else "cao"
    insight_text_rating = f"Nhóm sách có mức đánh giá '{top_rating_group}' đạt lượng tiêu thụ trung bình xuất sắc nhất, tái khẳng định rằng review tốt là công cụ marketing 0 đồng mạnh mẽ."
    render_chart_with_insight(
      _fig_rating_group_sold(filtered, palette),
      toggle_key="overview_chart_rating_group",
      insight_text=insight_text_rating,
      palette=palette,
    )
  with col_d:
    top_publisher = filtered.groupby("publisher_group")["sold_count"].sum().idxmax() if "publisher_group" in filtered.columns and not filtered.empty else "NXB"
    insight_text_publisher = f"Nhóm '{top_publisher}' chiếm miếng bánh thị phần doanh số lớn nhất, chứng tỏ khách hàng mua sách vẫn đặt niềm tin rất lớn vào uy tín của các thương hiệu quen thuộc."
    render_chart_with_insight(
      _fig_publisher_share(filtered, palette),
      toggle_key="overview_chart_publisher_share",
      insight_text=insight_text_publisher,
      palette=palette,
    )

