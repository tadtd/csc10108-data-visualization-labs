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
)

def main():
  st.set_page_config(
    page_title="Các yếu tố ảnh hưởng đến hiệu quả bán hàng của mặt hàng sách trên Tiki",
    layout="wide",
  )

  st.title("Các yếu tố ảnh hưởng đến hiệu quả bán hàng của mặt hàng sách trên Tiki")

  tab_renderers = {
    "Giảm giá và Nhà xuất bản": render_discount_publisher,
    "Hiệu ứng FOMO và quà tặng": render_fomo_gift,
    "Chiến lược thể loại": render_genre_strategy,
    "Thông tin sách và review": render_info_review,
    "Từ khóa tiêu đề và tác giả phổ biến (Tuấn)": render_keywords_popular,
  }

  with st.sidebar:
    selected_tab = st.selectbox("Chọn tab phân tích", list(tab_renderers.keys()))

  tab_renderers[selected_tab]()

if __name__ == "__main__":
  main()