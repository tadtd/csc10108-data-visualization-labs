import streamlit as st
import pandas as pd
import os

def load_data():
    path = "data/processed/products.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

st.set_page_config(page_title="Tiki Data Analysis Dashboard")
st.title("Tiki Products Dashboard")

df = load_data()

if not df.empty:
    st.header("Overview")
    st.write(f"Total products: {len(df)}")
    
    st.header("Data Preview")
    st.dataframe(df.head())
    
    st.header("Price Distribution")
    if 'price' in df.columns:
        st.bar_chart(df['price'])
else:
    st.warning("No data found in data/processed/products.csv. Please run the pipeline first.")
