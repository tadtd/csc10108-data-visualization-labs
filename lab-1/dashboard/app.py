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
  st.set_page_config(page_title="Các yếu tố ảnh hưởng đến hiệu quả bán hàng của mặt hàng sách trên Tiki")

  st.title("Các yếu tố ảnh hưởng đến hiệu quả bán hàng của mặt hàng sách trên Tiki")

  with st.sidebar:
    selected_tab = st.selectbox("Chọn tab", ["Discount Publisher", "Fomo Gift", "Genre Strategy", "Info Review", "Keywords Popular"])

  if selected_tab == "Discount Publisher":
    render_discount_publisher()
  elif selected_tab == "Fomo Gift":
    render_fomo_gift()
  elif selected_tab == "Genre Strategy":
    render_genre_strategy()
  elif selected_tab == "Info Review":
    render_info_review()
  elif selected_tab == "Keywords Popular":
    render_keywords_popular()

if __name__ == "__main__":
  main()