import streamlit as st
from src.ui import render_qa_tab, render_practice_tab

st.set_page_config(
    page_title="PSLE Math Study Bot",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 PSLE Math Study Bot")

tab_qa, tab_practice = st.tabs(["Ask a Question", "Practice Mode"])

with tab_qa:
    render_qa_tab()

with tab_practice:
    render_practice_tab()
