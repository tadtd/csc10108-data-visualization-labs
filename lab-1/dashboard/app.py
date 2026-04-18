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
  render_genre_strategy,
  render_overview,
)


TAB_RENDERERS = {
  "Tổng quan": render_overview,
  "Giá & Nhà xuất bản": render_discount_publisher,
  "Chiến lược thể loại": render_genre_strategy,
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
    st.session_state["color_mode"] = color_mode
    
    selected_tab = st.selectbox(
      SIDEBAR_SETTINGS["tab_label"],
      list(TAB_RENDERERS.keys()),
      index=0,
    )

  TAB_RENDERERS[selected_tab]()

if __name__ == "__main__":
  main()