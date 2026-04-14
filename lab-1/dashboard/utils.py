import streamlit as st
import pandas as pd
import os


def load_data(path: str) -> pd.DataFrame:
  if os.path.exists(path):
    return pd.read_csv(path)
  raise FileNotFoundError(f"File {path} not found")