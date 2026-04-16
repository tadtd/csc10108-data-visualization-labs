import sys
from pathlib import Path

import streamlit as st

# Ensure absolute package imports work when running `streamlit run dashboard/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.config import APP_TITLE, SIDEBAR_SETTINGS
from dashboard.tabs import (
  render_discount_publisher,
  render_fomo_gift,
  render_genre_strategy,
  render_info_review,
  render_keywords_popular,
)


TAB_RENDERERS = {
  "Giá & Nhà xuất bản": render_discount_publisher,
  "FOMO & Quà tặng": render_fomo_gift,
  "Chiến lược thể loại": render_genre_strategy,
  "Thông tin & Đánh giá": render_info_review,
  "Từ khóa phổ biến": render_keywords_popular,
}


def main():
  st.set_page_config(page_title=APP_TITLE)

  st.title(APP_TITLE)

  with st.sidebar:
    st.markdown(f"### {SIDEBAR_SETTINGS['title']}")
    
    color_mode = st.selectbox(
      'Chế độ màu',
      ['Mặc định', 'Thân thiện mù màu'],
      index=1,
    )
    
    selected_tab = st.selectbox(
      SIDEBAR_SETTINGS["tab_label"],
      SIDEBAR_SETTINGS["tabs"],
      index=0,
    )

  TAB_RENDERERS[selected_tab]()

if __name__ == "__main__":
  main()