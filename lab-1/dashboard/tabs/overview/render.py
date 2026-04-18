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
from dashboard.utils import apply_common_style, get_palette, vnd_format


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

  st.subheader("Tổng quan")
  st.caption(
    "Lát cắt dữ liệu toàn cục: điều chỉnh bộ lọc bên dưới để cập nhật KPI và các panel."
  )

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

  # —— Khu vực 1: Filter bar ——
  with st.container():
    st.markdown("**Bộ lọc toàn cục**")
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    with fc1:
      genres_sel = st.multiselect(
        "Thể loại",
        options=genre_options,
        default=[],
        help="Để trống = tất cả thể loại.",
      )
    with fc2:
      price_range = st.slider(
        "Khoảng giá (VND)",
        min_value=p_min,
        max_value=p_max,
        value=(p_min, p_max),
        format="%d",
      )
    with fc3:
      top_n = st.number_input("Top-N thể loại (panel trái)", min_value=3, max_value=30, value=10, step=1)

    filtered = _filter_products(raw, genres_sel, price_range[0], price_range[1])
    if not filtered.empty:
      filtered, _ = DiscountPublisherRetriever().with_publisher_group(filtered)
    filter_hint = f"Đang hiển thị **{len(filtered):,}** sản phẩm".replace(",", ".")
    if genres_sel:
      filter_hint += f" · Thể loại: {', '.join(genres_sel[:5])}" + ("…" if len(genres_sel) > 5 else "")
    st.markdown(filter_hint)

  st.divider()

  kpis = _kpi_snapshot(filtered)

  # —— Khu vực 2: KPI row ——
  st.markdown('<div class="overview-kpi-row"></div>', unsafe_allow_html=True)
  k1, k2, k3, k4 = st.columns(4)
  with k1:
    st.metric("Sản phẩm (sau lọc)", f"{int(kpis['n_products']):,}".replace(",", "."))
  with k2:
    st.metric("Tổng lượt bán", f"{int(kpis['total_sold']):,}".replace(",", "."))
  with k3:
    st.metric("Đánh giá TB", f"{kpis['avg_rating']:.2f}")
  with k4:
    st.metric("Giá TB", f"{vnd_format(kpis['avg_price'])} đ" if kpis["n_products"] else "—")

  st.divider()

  # —— Khu vực 3: Main 65 / 35 ——
  st.markdown('<div class="overview-main-split"></div>', unsafe_allow_html=True)
  left, right = st.columns([0.65, 0.35])

  with left:
    st.markdown("**Khu vực chính (65%)**")
    st.plotly_chart(_fig_top_genres(filtered, palette, int(top_n)), width='stretch')
    st.plotly_chart(_fig_price_distribution(filtered, palette), width='stretch')

  with right:
    st.markdown("**Khu vực bổ trợ (35%)**")
    st.plotly_chart(_fig_rating_group_sold(filtered, palette), width='stretch')
    st.plotly_chart(_fig_publisher_share(filtered, palette), width='stretch')

  st.divider()

  # —— Khu vực 4: Insights ——
  st.markdown("**Nhận xét nhanh (dẫn sang chuyên đề chi tiết)**")
  ts = f"{int(kpis['total_sold']):,}".replace(",", ".")
  np = f"{int(kpis['n_products']):,}".replace(",", ".")
  st.markdown(
    f"""
<div class="insight-box">
  <p style="margin:0;">
    Trong lát cắt hiện tại, tổng lượt bán ghi nhận là <strong>{ts}</strong>
    trên <strong>{np}</strong> sản phẩm.
    Dùng các tab <em>Giá & Nhà xuất bản</em>, <em>Chiến lược thể loại</em>, … để đi sâu từng mục tiêu SMART.
  </p>
</div>
""",
    unsafe_allow_html=True,
  )
