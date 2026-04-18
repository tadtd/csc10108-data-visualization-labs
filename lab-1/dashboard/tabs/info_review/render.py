import streamlit as st

from dashboard.utils import load_data

def render():
  st.subheader("Info Review")
  st.caption("Demo tab: load a CSV file and preview the first 10 rows.")

  csv_path = st.text_input("CSV path", value="data/info_review.csv", key="info_review_csv_path")
  if st.button("Load data", key="info_review_load_button"):
    try:
      df = load_data(csv_path)
      st.success(f"Loaded {len(df)} rows.")
      st.dataframe(df.head(10), width='stretch')
    except FileNotFoundError as error:
      st.error(str(error))
