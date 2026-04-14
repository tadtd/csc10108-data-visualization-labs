import streamlit as st

from dashboard.utils import load_data

def render():
  st.subheader("Genre Strategy")
  st.caption("Demo tab: load a CSV file and preview the first 10 rows.")

  csv_path = st.text_input("CSV path", value="data/genre_strategy.csv", key="genre_strategy_csv_path")
  if st.button("Load data", key="genre_strategy_load_button"):
    try:
      df = load_data(csv_path)
      st.success(f"Loaded {len(df)} rows.")
      st.dataframe(df.head(10), use_container_width=True)
    except FileNotFoundError as error:
      st.error(str(error))