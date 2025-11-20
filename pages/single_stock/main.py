# main.py

import streamlit as st
import pandas as pd

from pages.single_stock.single_stock_overview import display_overview, get_yf_data
from pages.single_stock.single_stock_fundamentals import display_fundamentals
from pages.single_stock.single_stock_charts import display_charts
from pages.single_stock.utils import (
    get_dolthub_income_statement,
    get_dolthub_balance_sheet_assets,
    get_dolthub_balance_sheet_liabilities,
    get_dolthub_balance_sheet_equity,
    get_dolthub_cash_flow,
)

# -----------------------------
# Page Setup
# -----------------------------
st.title("📈 Company Overview")

# -----------------------------
# Global Styling
# -----------------------------
st.markdown(
    """
<style>
/* Metric cards (your existing style) */
.metric-card {
    border-radius:10px;
    padding:15px;
    margin-bottom:12px;
    color:white;
    text-align: center;
}
.metric-label {
    font-size:13px;
    opacity:0.85;
}
.metric-value {
    font-size:19px;
    font-weight:600;
}

/* ✅ ONLY affect Streamlit segmented controls (globally) */
div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background-color: #4CAF50 !important;  /* green */
    color: #ffffff !important;
    border-color: #4CAF50 !important;
}

/* Optional: make unselected segments neutral */
div[data-testid="stSegmentedControl"] button {
    color: inherit;
    border-radius: 999px !important;  /* keeps pill shape */
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Ticker Input
# -----------------------------
ticker = st.text_input("Enter a stock ticker (e.g., AAPL, MSFT, or 2330.TW):").strip().upper()

# -----------------------------
# Display Logic
# -----------------------------
if ticker:
    # -------- DoltHub Income Statement (Primary Source) --------
    dolt_df_raw = get_dolthub_income_statement(ticker)

    if isinstance(dolt_df_raw, str) and dolt_df_raw.startswith("ERROR::"):
        st.warning(f"DoltHub error: {dolt_df_raw}")
        dolt_df = pd.DataFrame()
    else:
        dolt_df = dolt_df_raw

    using_dolthub = not dolt_df.empty

    # -------- Yahoo Finance Fallback --------
    yf_df, _ = get_yf_data(ticker)

    # -------- Overview Section --------
    with st.expander("Overview", expanded=True):
        display_overview(ticker)

    # -------- Charts Section --------
    with st.expander("Charts", expanded=True):
        display_charts(ticker, yf_df, dolt_df, using_dolthub)

    # -------- Financial Statements Section --------
    with st.expander("Financial Statements", expanded=True):

        # Statement selector
        statement_tab = st.segmented_control(
            "Select Statement:",
            options=["Income Statement", "Balance Sheet", "Cash Flow", "Key Ratios"],
        )

        # Period selector (Annual / Quarterly)
        statement_period = st.segmented_control(
            "",
            options=["Annual", "Quarterly"],
        )

        # Prepare balance sheet tables
        assets_df = None
        liabilities_df = None
        equity_df = None

        if using_dolthub:
            assets_df = get_dolthub_balance_sheet_assets(ticker)
            liabilities_df = get_dolthub_balance_sheet_liabilities(ticker)
            equity_df = get_dolthub_balance_sheet_equity(ticker)

            # If all BS tables are empty, fall back to Yahoo
            if assets_df.empty and liabilities_df.empty and equity_df.empty:
                st.warning("📭 No DoltHub financials found. Falling back to Yahoo Finance.")
                using_dolthub = False
                assets_df = None
                liabilities_df = None
                equity_df = None

        # -------- Unified Fundamentals Rendering --------
        display_fundamentals(
            statement_tab=statement_tab,
            statement_period=statement_period,   # <- Annual / Quarterly toggle
            dolt_df=dolt_df if using_dolthub else None,
            assets_df=assets_df if using_dolthub else None,
            liabilities_df=liabilities_df if using_dolthub else None,
            equity_df=equity_df if using_dolthub else None,
            get_dolthub_cash_flow=get_dolthub_cash_flow if using_dolthub else None,
            ticker=ticker,
            yf_df=None if using_dolthub else yf_df,
        )

else:
    st.info("Please enter a stock ticker to view its information.")
