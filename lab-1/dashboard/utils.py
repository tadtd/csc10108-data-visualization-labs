import pandas as pd
import os
import base64

import streamlit as st

DEFAULT_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
COLORBLIND_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

def load_data(path: str) -> pd.DataFrame:
  if os.path.exists(path):
    return pd.read_csv(path)
  raise FileNotFoundError(f"File {path} not found")


def encode_svg(svg_string: str):
  b64 = base64.b64encode(svg_string.encode('utf-8')).decode("utf-8")
  html = f'<img src="data:image/svg+xml;base64,{b64}"/>'
  return html

def decode_svg(html: str):
  b64 = html.split("base64,")[1]
  svg_string = base64.b64decode(b64).decode("utf-8")
  return svg_string

def render_svg(svg_string: str):
  html = encode_svg(svg_string)
  st.markdown(html, unsafe_allow_html=True)


def to_bool(series: pd.Series) -> pd.Series:
  if series.dtype == bool:
    return series.fillna(False)
  normalized = series.fillna(False).astype(str).str.strip().str.lower()
  return normalized.isin({"1", "true", "yes", "y"})


def vnd_format(value: float) -> str:
  return f"{value:,.0f}".replace(",", ".")


def get_palette(color_mode: str) -> list[str]:
  if color_mode == "Thân thiện mù màu":
    return COLORBLIND_PALETTE
  return DEFAULT_PALETTE


def apply_common_style() -> None:
  st.markdown(
    """
    <style>
      html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
      }
      .insight-box {
        border-left: 4px solid #1f77b4;
        background: rgba(31, 119, 180, 0.08);
        padding: 8px 12px;
        border-radius: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
  )


def load_data(path: str) -> pd.DataFrame:
  if os.path.exists(path):
    return pd.read_csv(path)
  raise FileNotFoundError(f"File {path} not found")