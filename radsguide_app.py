import streamlit as st
import pandas as pd

st.set_page_config(page_title="RadsGuide Minimal Diagnostic", layout="centered")
st.title("🔧 RadsGuide Minimal CSV Test")

st.write("✅ Reached before CSV load")

try:
    df = pd.read_csv("test.csv")
    st.write("✅ CSV loaded successfully")
    st.write(df.head())
except Exception as e:
    st.error(f"❌ CSV failed to load: {e}")

st.markdown("---")
st.caption("Debug version of RadsGuide · Kent Kleinschmidt")
