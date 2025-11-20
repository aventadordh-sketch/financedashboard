import streamlit as st
from streamlit_option_menu import option_menu
import os

# -----------------------------------------------
# 🎯 Load ticker list from file
def load_ticker_list(filepath="all_tickers.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []

ticker_list = load_ticker_list()
# -----------------------------------------------

# 🛠 Page config
st.set_page_config(layout="wide", page_title="Finance Dashboard", initial_sidebar_state="collapsed")

# 🔒 Hide sidebar
hide_sidebar_style = """
    <style>
        section[data-testid="stSidebar"] {
            display: none;
        }
        div[data-testid="stAppViewContainer"] {
            margin-left: 0px;
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# 🧠 App title
st.title("📊 Finance Dashboard")

# 🔘 Menu
selected = option_menu(
    menu_title=None,
    options=["Dashboard", "Personal Finance", "Single Stock", "Comparison", "Analysis"],
    icons=["bar-chart-line", "graph-up", "columns-gap", "columns-gap"],
    orientation="horizontal",
    default_index=0
)

# 🔁 Helper to load page scripts safely
def run_script(path):
    full_path = os.path.join("pages", path)
    if os.path.exists(full_path):
        with open(full_path, encoding="utf-8") as f:
            code = f.read()
        exec(code, globals())  # Ensures shared variable scope across scripts
    else:
        st.error(f"Missing file: {full_path}")

# 🧩 Page logic
if selected == "Dashboard":
    st.markdown("Welcome to the Dashboard Overview!")

elif selected == "Personal Finance":
    run_script("1_Personal_Finance.py")

elif selected == "Single Stock":
    run_script("single_stock/main.py")

elif selected == "Comparison":
    run_script("3_Comparison_Mode.py")

elif selected == "Analysis":
    run_script("4_Analysis_Mode.py")