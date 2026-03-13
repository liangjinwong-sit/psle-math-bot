import streamlit as st
from src.ui import render_qa_tab, render_practice_tab

st.set_page_config(
    page_title="PSLE Math Study Bot",
    page_icon=":material/calculate:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for a clean look ──
st.markdown("""
<style>
    /* Rounded containers */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }

    /* Bigger tab labels */
    button[data-baseweb="tab"] > div {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
    }

    /* Rounded buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
    }

    /* Primary buttons */
    .stButton > button[kind="primary"] {
        font-size: 1.05rem !important;
    }

    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        font-size: 1rem !important;
    }

    /* Alert boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    /* Expander */
    details {
        border-radius: 12px !important;
    }

    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        border-radius: 0 16px 16px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown(
    "<h1 style='text-align: center;'>PSLE Math Study Bot</h1>"
    "<p style='text-align: center; font-size: 1.1rem; opacity: 0.8;'>"
    "Your friendly math helper for Primary 5 &amp; 6! Ask questions, practise, and improve."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

tab_qa, tab_practice = st.tabs(["Ask a Question", "Practice Mode"])

with tab_qa:
    render_qa_tab()

with tab_practice:
    render_practice_tab()
