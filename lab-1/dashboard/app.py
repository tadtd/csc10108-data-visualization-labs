import sys
from pathlib import Path

import streamlit as st

# Ensure absolute package imports work when running `streamlit run dashboard/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.tabs import (
  render_discount_publisher,
  render_fomo_gift,
  render_genre_strategy,
  render_info_review,
  render_keywords_popular,
  render_overview,
)

def main():
  st.set_page_config(
    page_title="Các yếu tố ảnh hưởng đến hiệu quả bán hàng của mặt hàng sách trên Tiki",
    layout="wide",
  )

  st.title("Các yếu tố ảnh hưởng đến hiệu quả bán hàng của mặt hàng sách trên Tiki")
  st.caption("Dashboard tương tác phân tích các yếu tố ảnh hưởng đến hiệu quả bán sách trên sàn Tiki.")

  tab_renderers = {
    "Tổng quan": render_overview,
    "Giá, giảm giá và nhà xuất bản": render_discount_publisher,
    "Huy hiệu bán chạy và quà tặng": render_fomo_gift,
    "Thể loại, nhà xuất bản và combo": render_genre_strategy,
    "Thông tin sách và phản hồi người mua": render_info_review,
    "Từ khóa tiêu đề và tác giả": render_keywords_popular,
  }

  with st.sidebar:
    st.markdown("### Tùy chỉnh nhanh")
    st.session_state["color_mode"] = st.selectbox(
      "Bảng màu",
      ["Mặc định", "Thân thiện mù màu"],
      index=0 if st.session_state.get("color_mode", "Mặc định") == "Mặc định" else 1,
      key="global_color_mode",
    )

  selected_tab = st.segmented_control(
    "Chọn chuyên đề phân tích",
    options=list(tab_renderers.keys()),
    key="dashboard_selected_tab",
    selection_mode="single",
    default="Tổng quan",
    label_visibility="collapsed",
  )
  if selected_tab is None:
    selected_tab = "Tổng quan"
  tab_renderers[selected_tab]()

if __name__ == "__main__":
  main()