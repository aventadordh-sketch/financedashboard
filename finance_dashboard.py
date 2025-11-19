import streamlit as st
from streamlit_option_menu import option_menu

# -----------------------------------------------
# 🎯 Load ticker list from file
def load_ticker_list(filepath="all_tickers.txt"):
    try:
        with open(filepath, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []

ticker_list = load_ticker_list()
# -----------------------------------------------

# 🛠 Page config
st.set_page_config(layout="wide", page_title="Finance Dashboard")

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

# 🧩 Page logic
if selected == "Dashboard":
    st.markdown("Welcome to the Dashboard Overview!")

elif selected == "Personal Finance":
    exec(open("pages/1_Personal_Finance.py").read())

elif selected == "Single Stock":
    # 💡 Pass ticker_list via globals if needed in child script
    exec(open("pages/2_Single_Stock.py").read())

elif selected == "Comparison":
    exec(open("pages/3_Comparison_Mode.py").read())

elif selected == "Analysis":
    exec(open("pages/4_Analysis_Mode.py").read())
