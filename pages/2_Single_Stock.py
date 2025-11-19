import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from sqlalchemy import create_engine
import pymysql

CASH_FLOW_FIELDS = [
    "net_income",
    "depreciation_amortization_and_depletion",
    "net_change_from_assets",
    "net_cash_from_discontinued_operations",
    "other_operating_activities",
    "net_cash_from_operating_activities",
    "property_and_equipment",
    "acquisition_of_subsidiaries",
    "investments",
    "other_investing_activities",
    "net_cash_from_investing_activities",
    "issuance_of_capital_stock",
    "issuance_of_debt",
    "increase_short_term_debt",
    "payment_of_dividends_and_other_distributions",
    "other_financing_activities",
    "net_cash_from_financing_activities",
    "effect_of_exchange_rate_changes",
    "net_change_in_cash_and_equivalents",
    "cash_at_beginning_of_period",
    "cash_at_end_of_period",
    "diluted_net_eps"
]

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(layout="wide", page_title="Single Stock View")
st.title("📈 Single Stock Financial Overview")

# -----------------------------
# Load Ticker List from File
# -----------------------------
@st.cache_data
def load_ticker_list(filepath="all_tickers.txt"):
    try:
        with open(filepath, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers
    except FileNotFoundError:
        st.warning(f"Ticker file not found: {filepath}")
        return []

# Load tickers
ticker_list = load_ticker_list()

# Prepend a placeholder to avoid auto-select
options = ["Select a ticker..."] + ticker_list
selected_ticker = st.selectbox("Enter a stock ticker:", options)

# Use empty string if no selection made
ticker = selected_ticker if selected_ticker != "Select a ticker..." else ""
# Ensure these exist before UI attempts to reference them
using_dolthub = False
dolt_df = pd.DataFrame()
yf_df = pd.DataFrame()
info = {}


# -----------------------------
# Data Sources
# -----------------------------
@st.cache_data(ttl=86400)
def get_dolthub_income_statement(ticker):
    try:
        engine = create_engine("mysql+pymysql://root@localhost:3307/earnings")
        query = f"""
            SELECT * FROM income_statement
            WHERE act_symbol = '{ticker.upper()}'
            ORDER BY date DESC;
        """
        df = pd.read_sql(query, con=engine)
        return df
    except Exception as e:
        return f"ERROR::{str(e)}"

# -----------------------------
# NEW: Balance Sheet Loaders
# -----------------------------
@st.cache_data(ttl=86400)
def get_dolthub_balance_sheet_assets(ticker):
    try:
        engine = create_engine("mysql+pymysql://root@localhost:3307/earnings")
        query = f"""
            SELECT * FROM balance_sheet_assets
            WHERE act_symbol = '{ticker.upper()}'
            ORDER BY date DESC;
        """
        return pd.read_sql(query, con=engine)
    except Exception as e:
        return f"ERROR::{str(e)}"

@st.cache_data(ttl=86400)
def get_dolthub_balance_sheet_liabilities(ticker):
    try:
        engine = create_engine("mysql+pymysql://root@localhost:3307/earnings")
        query = f"""
            SELECT * FROM balance_sheet_liabilities
            WHERE act_symbol = '{ticker.upper()}'
            ORDER BY date DESC;
        """
        return pd.read_sql(query, con=engine)
    except Exception as e:
        return f"ERROR::{str(e)}"


@st.cache_data(ttl=86400)
def get_dolthub_balance_sheet_equity(ticker):
    try:
        engine = create_engine("mysql+pymysql://root@localhost:3307/earnings")
        query = f"""
            SELECT * FROM balance_sheet_equity
            WHERE act_symbol = '{ticker.upper()}'
            ORDER BY date DESC;
        """
        return pd.read_sql(query, con=engine)
    except Exception as e:
        return f"ERROR::{str(e)}"

@st.cache_data(ttl=86400)
def get_dolthub_cash_flow(ticker):
    try:
        engine = create_engine("mysql+pymysql://root@localhost:3307/earnings")
        query = f"""
            SELECT * FROM cash_flow_statement
            WHERE act_symbol = '{ticker.upper()}'
            ORDER BY date DESC;
        """
        return pd.read_sql(query, con=engine)
    except Exception as e:
        return f"ERROR::{str(e)}"

@st.cache_data(ttl=86400)
def get_yf_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        df = stock.financials.T
        df.index = pd.to_datetime(df.index)
        return df, info
    except Exception as e:
        return None, {"error": str(e)}

@st.cache_data(ttl=86400)
def get_historical_data(ticker, period="6mo", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)
        return hist
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")
        return pd.DataFrame()

# -----------------------------
# Format Helpers
# -----------------------------
def format_num(n):
    if n is None: return "—"
    try: n = float(n)
    except: return "—"
    if abs(n) >= 1e9: return f"{n/1e9:.2f}B"
    if abs(n) >= 1e6: return f"{n/1e6:.2f}M"
    return f"{n:,.2f}"

def safe(info, key): return info.get(key, "—")

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

def render_card(label, value, color):
    return f"""
    <div class="metric-card" style="background-color:{color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

def format_ratio(val):
    if val is None or val == "—":
        return "—"
    try:
        return f"{float(val):.2f}"
    except:
        return "—"

# -----------------------------
# Main Display
# -----------------------------
if ticker:
    available_metrics = []

    # Get Yahoo Info always for summary
    yf_df, info = get_yf_data(ticker)

    # Try DoltHub for financials
    dolt_df = get_dolthub_income_statement(ticker.upper())
    if isinstance(dolt_df, str) and dolt_df.startswith("ERROR::"):
        dolt_error = dolt_df.replace("ERROR::", "")
        dolt_df = pd.DataFrame()
    else:
        dolt_error = None

    using_dolthub = not dolt_df.empty

    # ✅ NEW: Load DoltHub balance sheet data
    assets_df = get_dolthub_balance_sheet_assets(ticker.upper())
    liabilities_df = get_dolthub_balance_sheet_liabilities(ticker.upper())
    equity_df = get_dolthub_balance_sheet_equity(ticker.upper())

    # ✅ Load Cash Flow data in DoltHub field order
    available_meta = [col for col in ["date", "period", "act_symbol"] if col in dolt_df.columns]
    available_cash_cols = [col for col in CASH_FLOW_FIELDS if col in dolt_df.columns]
    cash_flow_df = dolt_df[available_meta + available_cash_cols].copy()

    # Header
    st.subheader(f"{info.get('shortName', ticker.upper())} ({ticker.upper()})")
    st.caption(f"**Data Source:** {'📦 DoltHub (EDGAR)' if using_dolthub else '🌐 Yahoo Finance'}")
    if dolt_error:
        st.warning(f"DoltHub error: {dolt_error} (falling back to Yahoo Finance)")

    # Company Info (Blue)
    row1 = st.columns(3)
    row1[0].markdown(render_card("Sector", safe(info, "sector"), "#2A4E96"), unsafe_allow_html=True)
    row1[1].markdown(render_card("Industry", safe(info, "industry"), "#2A4E96"), unsafe_allow_html=True)
    row1[2].markdown(render_card("Country", safe(info, "country"), "#2A4E96"), unsafe_allow_html=True)

    # Overview (Teal)
    row2 = st.columns(3)
    row2[0].markdown(render_card("Market Cap", format_num(info.get("marketCap")), "#1AA3A3"), unsafe_allow_html=True)
    row2[1].markdown(render_card("Current Price", format_num(info.get("currentPrice")), "#1AA3A3"), unsafe_allow_html=True)
    row2[2].markdown(render_card("Beta", format_ratio(info.get("beta")), "#1AA3A3"), unsafe_allow_html=True)

    # Valuation (Purple)
    row3 = st.columns(3)
    row3[0].markdown(render_card("P/E Ratio", format_ratio(info.get("trailingPE")), "#7B3F99"), unsafe_allow_html=True)
    row3[1].markdown(render_card("P/S Ratio", format_ratio(info.get("priceToSalesTrailing12Months")), "#7B3F99"), unsafe_allow_html=True)
    row3[2].markdown(render_card("P/B Ratio", format_ratio(info.get("priceToBook")), "#7B3F99"), unsafe_allow_html=True)

if ticker:

    # -----------------------------
    # Price Chart
    # -----------------------------
    st.markdown("### 📉 Price Chart")
    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.radio("Chart Type", ["Line", "Candlestick"], horizontal=True)
    with col2:
        time_range = st.selectbox(
            "Time Range", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2
        )

    hist = get_historical_data(ticker, period=time_range)

    if not hist.empty:
        fig = go.Figure()
        if chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                name="Close"
            ))
        else:
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name="Candlestick"
            ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            showlegend=False,
            margin=dict(l=40, r=40, t=30, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff")
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Re-initialize metrics and df_plot in case of re-renders
# -----------------------------
metrics = {
    "Sales": "sales",
    "Cost of Goods Sold": "cost_of_goods",
    "Gross Profit": "gross_profit",
    "Selling & Admin Expenses": "selling_administrative_expense",
    "Income After Depreciation": "income_after_depreciation",
    "Non-Operating Income": "non_operating_income",
    "Interest Expense": "interest_expense",
    "Pretax Income": "pretax_income",
    "Income Taxes": "income_taxes",
    "Minority Interest": "minority_interest",
    "Investment Gains": "investment_gains",
    "Other Income": "other_income",
    "Income from Continuing Ops": "income_from_continuing_operations",
    "Extras & Discontinued Ops": "extras_and_discontinued_operations",
    "Net Income": "net_income",
    "Income Before Depreciation": "income_before_depreciation",
    "Depreciation & Amortization": "depreciation_and_amortization",
    "Average Shares": "average_shares",
    "Diluted EPS (Before Non Recurring)": "diluted_eps_before_non_recurring",
    "Diluted Net EPS": "diluted_net_eps"
}

if using_dolthub:
    available_metrics = [k for k, v in metrics.items() if v in dolt_df.columns and dolt_df[v].notna().any()]
    df_plot = dolt_df
else:
    metrics = {col: col for col in yf_df.columns}
    available_metrics = sorted(list(metrics.keys()))
    df_plot = yf_df

statement_period = st.radio("Select Frequency:", ["Annual", "Quarterly"], horizontal=True)
# -----------------------------
# Financial Chart Section (Fixed for Annual/Quarterly)
# -----------------------------

if available_metrics:
    st.markdown("### 📊 Financial Metric Over Time")
    selected_label = st.selectbox("Choose a Metric to Visualize:", available_metrics)
    selected_col = metrics[selected_label]

    df = df_plot.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter by statement period
    if using_dolthub and "period" in df.columns:
        df["period"] = df["period"].astype(str).str.upper()
        df = df[df["period"] == ("YEAR" if statement_period == "Annual" else "QUARTER")]

    # Create period labels
    if statement_period == "Annual":
        df["Period"] = df["date"].dt.year.astype(str)
    else:
        q = (df["date"].dt.month - 1) // 3 + 1
        df["Period"] = df["date"].dt.year.astype(str) + " Q" + q.astype(str)

    # Group and aggregate
    grouped = df.groupby("Period")[selected_col].sum().dropna().reset_index()

    periods = grouped["Period"].tolist()
    values = grouped[selected_col].astype(float).tolist()

    if values:
        # Unit scale detection
        max_val = max(abs(v) for v in values)
        if max_val >= 1e12:
            scale, suffix = 1e12, "T"
        elif max_val >= 1e9:
            scale, suffix = 1e9, "B"
        elif max_val >= 1e6:
            scale, suffix = 1e6, "M"
        elif max_val >= 1e3:
            scale, suffix = 1e3, "K"
        else:
            scale, suffix = 1, ""

        def format_scaled(v): return f"{v / scale:,.2f}"

        # Color logic
        max_val = max(values)
        min_val = min(values)
        bar_colors = [
            "#6FCF97" if v == max_val else "#D87C6E" if v == min_val else "cornflowerblue"
            for v in values
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=periods,
            y=values,
            text=[format_scaled(v) for v in values],
            textposition="outside",
            marker=dict(color=bar_colors)
        ))

        fig.update_layout(
            title=selected_label,
            xaxis_title="Period",
            yaxis_title=f"{selected_label} ({suffix})",
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff")
        )

        st.caption(f"Chart values shown in **{suffix}** (e.g., 1.25{suffix} = {int(scale):,})")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for '{selected_label}'")


# -----------------------------
# Raw Financial Statement View
# -----------------------------
statement_tab = st.radio(
    "Select Statement:",
    ["Income Statement", "Balance Sheet", "Cash Flow", "Key Ratios"],
    horizontal=True
)
st.markdown("### 🧾 Raw Financial Statements")

# -----------------------------
# Income Statement
# -----------------------------
if statement_tab == "Income Statement":
    label = "Income Statement"
    filtered_df = pd.DataFrame()

    if not dolt_df.empty:
        if "period" in dolt_df.columns:
            dolt_df["period"] = dolt_df["period"].astype(str).str.upper()
            period_value = "YEAR" if statement_period == "Annual" else "QUARTER"
            filtered_df = dolt_df[dolt_df["period"] == period_value].copy()
        else:
            filtered_df = dolt_df.copy()

        if filtered_df.empty:
            st.info(f"No {statement_period.lower()} income statement data available for this ticker.")
        else:
            for col in filtered_df.select_dtypes(include=["float", "int"]).columns:
                filtered_df[col] = filtered_df[col].apply(
                    lambda x: f"{x:,.0f}" if pd.notnull(x) else "—"
                )

            ordered_cols = ["date"] + [c for c in filtered_df.columns if c != "date"]
            df_clean = filtered_df[ordered_cols].copy()

            if "date" not in df_clean.columns:
                st.error("No date column found in income statement.")
            else:
                df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
                df_clean = df_clean.sort_values("date")

                df_transposed = (
                    df_clean.set_index("date")
                            .T
                            .reset_index()
                            .rename(columns={"index": label})
                )

                # Remove metadata
                df_transposed = df_transposed[~df_transposed[label].isin(["act_symbol", "period"])]
                df_transposed[label] = df_transposed[label].str.replace("_", " ").str.title()

                # Format column names
                formatted_cols = []
                for col in df_transposed.columns:
                    if isinstance(col, pd.Timestamp):
                        if statement_period == "Annual":
                            formatted_cols.append(col.strftime("%Y"))
                        else:
                            q = (col.month - 1) // 3 + 1
                            formatted_cols.append(f"{col.year} Q{q}")
                    else:
                        formatted_cols.append(col)
                df_transposed.columns = formatted_cols

                # Scale logic
                def try_parse(x):
                    try:
                        return float(str(x).replace(",", ""))
                    except:
                        return None

                numeric_values = []
                for col in df_transposed.columns:
                    if col != label:
                        numeric_values.extend([try_parse(x) for x in df_transposed[col] if try_parse(x) is not None])

                max_abs_val = max(abs(v) for v in numeric_values) if numeric_values else 1
                if max_abs_val >= 1e12:
                    scale, suffix = 1e12, "T"
                elif max_abs_val >= 1e9:
                    scale, suffix = 1e9, "B"
                elif max_abs_val >= 1e6:
                    scale, suffix = 1e6, "M"
                elif max_abs_val >= 1e3:
                    scale, suffix = 1e3, "K"
                else:
                    scale, suffix = 1, ""

                def format_scaled(x):
                    val = try_parse(x)
                    return f"{val / scale:,.2f}" if val is not None else "—"

                for col in df_transposed.columns:
                    if col != label:
                        df_transposed[col] = df_transposed[col].apply(format_scaled)
# -----------------------------
# Balance Sheet
# -----------------------------
elif statement_tab == "Balance Sheet":
    def format_balance_sheet(df, label):
        if df.empty:
            return st.info(f"No {label} data available.")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["period"] = df["period"].str.upper()
        df = df[df["period"] == ("YEAR" if statement_period == "Annual" else "QUARTER")]

        if df.empty:
            return st.info(f"No {statement_period.lower()} data for {label}.")

        df = df.sort_values("date")
        ordered_cols = ["date"] + [col for col in df.columns if col not in ["date", "act_symbol", "period"]]
        df_clean = df[ordered_cols].copy()

        df_transposed = (
            df_clean.set_index("date")
                    .T
                    .reset_index()
                    .rename(columns={"index": label})
        )

        df_transposed[label] = df_transposed[label].str.replace("_", " ").str.title()

        # Format column names
        formatted_cols = []
        for col in df_transposed.columns:
            if isinstance(col, pd.Timestamp):
                if statement_period == "Annual":
                    formatted_cols.append(col.strftime("%Y"))
                else:
                    q = (col.month - 1) // 3 + 1
                    formatted_cols.append(f"{col.year} Q{q}")
            else:
                formatted_cols.append(col)
        df_transposed.columns = formatted_cols

        # Scale logic
        def try_parse(x):
            try:
                return float(str(x).replace(",", ""))
            except:
                return None

        numeric_values = []
        for col in df_transposed.columns:
            if col != label:
                numeric_values.extend([try_parse(x) for x in df_transposed[col] if try_parse(x) is not None])

        max_abs_val = max(abs(v) for v in numeric_values) if numeric_values else 1
        if max_abs_val >= 1e12:
            scale, suffix = 1e12, "T"
        elif max_abs_val >= 1e9:
            scale, suffix = 1e9, "B"
        elif max_abs_val >= 1e6:
            scale, suffix = 1e6, "M"
        elif max_abs_val >= 1e3:
            scale, suffix = 1e3, "K"
        else:
            scale, suffix = 1, ""

        def format_scaled(x):
            val = try_parse(x)
            return f"{val / scale:,.2f}" if val is not None else "—"

        for col in df_transposed.columns:
            if col != label:
                df_transposed[col] = df_transposed[col].apply(format_scaled)

        st.markdown(f"#### 📦 {label}")
        st.caption(f"All values shown in **{suffix}** (e.g., 1.25{suffix} = {int(scale):,})")
        st.data_editor(
            df_transposed,
            use_container_width=True,
            hide_index=True,
            key=f"{label.lower().replace(' ', '_')}_editor",  # ✅ Safe for all balance sheet types
            column_config={label: st.column_config.TextColumn(label)}
        )

    format_balance_sheet(assets_df, "Assets")
    format_balance_sheet(liabilities_df, "Liabilities")
    format_balance_sheet(equity_df, "Equity")

# -----------------------------
# Cash Flow Statement
# -----------------------------
elif statement_tab == "Cash Flow":
    dolt_df = get_dolthub_cash_flow(ticker)

    # ✅ Step 1: Define cash flow fields in the order shown in DoltHub
    CASH_FLOW_FIELDS = [
        "net_income",
        "depreciation_amortization_and_depletion",
        "net_change_from_assets",
        "net_cash_from_discontinued_operations",
        "other_operating_activities",
        "net_cash_from_operating_activities",
        "property_and_equipment",
        "acquisition_of_subsidiaries",
        "investments",
        "other_investing_activities",
        "net_cash_from_investing_activities",
        "issuance_of_capital_stock",
        "issuance_of_debt",
        "increase_short_term_debt",
        "payment_of_dividends_and_other_distributions",
        "other_financing_activities",
        "net_cash_from_financing_activities",
        "effect_of_exchange_rate_changes",
        "net_change_in_cash_and_equivalents",
        "cash_at_beginning_of_period",
        "cash_at_end_of_period",
        "diluted_net_eps"
    ]

    # ✅ Step 2: Build cash_flow_df using ordered fields
    meta_columns = ["date", "period", "act_symbol"]
    available_meta = [col for col in meta_columns if col in dolt_df.columns]
    available_cash_cols = [col for col in CASH_FLOW_FIELDS if col in dolt_df.columns]
    cash_flow_df = dolt_df[available_meta + available_cash_cols].copy()

    # ✅ Step 3: Format and render
    def format_cash_flow(df, label="Cash Flow"):
        if df.empty:
            return st.info(f"No {label} data available.")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["period"] = df["period"].astype(str).str.upper()
        df = df[df["period"] == ("YEAR" if statement_period == "Annual" else "QUARTER")]

        if df.empty:
            return st.info(f"No {statement_period.lower()} cash flow data available.")

        df = df.sort_values("date")
        ordered_cols = ["date"] + [col for col in CASH_FLOW_FIELDS if col in df.columns]
        df_clean = df[ordered_cols].copy()

        df_transposed = (
            df_clean.set_index("date")
                    .T
                    .reset_index()
                    .rename(columns={"index": label})
        )

        df_transposed[label] = df_transposed[label].str.replace("_", " ").str.title()

        formatted_cols = []
        for col in df_transposed.columns:
            if isinstance(col, pd.Timestamp):
                if statement_period == "Annual":
                    formatted_cols.append(col.strftime("%Y"))
                else:
                    q = (col.month - 1) // 3 + 1
                    formatted_cols.append(f"{col.year} Q{q}")
            else:
                formatted_cols.append(col)
        df_transposed.columns = formatted_cols

        def try_parse(x):
            try:
                return float(str(x).replace(",", ""))
            except:
                return None

        numeric_values = []
        for col in df_transposed.columns:
            if col != label:
                numeric_values.extend([try_parse(x) for x in df_transposed[col] if try_parse(x) is not None])

        max_abs_val = max(abs(v) for v in numeric_values) if numeric_values else 1
        if max_abs_val >= 1e12:
            scale, suffix = 1e12, "T"
        elif max_abs_val >= 1e9:
            scale, suffix = 1e9, "B"
        elif max_abs_val >= 1e6:
            scale, suffix = 1e6, "M"
        elif max_abs_val >= 1e3:
            scale, suffix = 1e3, "K"
        else:
            scale, suffix = 1, ""

        def format_scaled(x):
            val = try_parse(x)
            return f"{val / scale:,.2f}" if val is not None else "—"

        for col in df_transposed.columns:
            if col != label:
                df_transposed[col] = df_transposed[col].apply(format_scaled)

        st.caption(f"All values shown in **{suffix}** (e.g., 1.25{suffix} = {int(scale):,})")
        st.data_editor(
            df_transposed,
            use_container_width=True,
            hide_index=True,
            key="cash_flow_editor",
            column_config={label: st.column_config.TextColumn(label)}
        )
    # ✅ Render it
    format_cash_flow(cash_flow_df)

#### Financial Ratios ####

elif statement_tab == "Key Ratios":
    st.markdown("### 📊 Key Financial Ratios (By Year)")

    if any(df is None or df.empty for df in [assets_df, liabilities_df, equity_df, cash_flow_df, dolt_df]):
        st.warning("Missing data from one or more financial statements. Please ensure all statements are loaded.")
    else:
        # --- Standardize date + period ---
        for df in [assets_df, liabilities_df, equity_df, cash_flow_df, dolt_df]:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["period"] = df["period"].astype(str).str.upper()

        period_filter = "YEAR" if statement_period == "Annual" else "QUARTER"
        dolt_period = dolt_df[dolt_df["period"] == period_filter].copy()
        assets_period = assets_df[assets_df["period"] == period_filter].copy()
        liabilities_period = liabilities_df[liabilities_df["period"] == period_filter].copy()
        equity_period = equity_df[equity_df["period"] == period_filter].copy()
        cash_period = cash_flow_df[cash_flow_df["period"] == period_filter].copy()

        merged = (
            dolt_period
            .merge(assets_period, on="date", suffixes=("", "_assets"))
            .merge(liabilities_period, on="date", suffixes=("", "_liab"))
            .merge(equity_period, on="date", suffixes=("", "_eq"))
            .merge(cash_period, on="date", suffixes=("", "_cf"))
        )

        merged.sort_values("date", inplace=True)
        merged["Period"] = merged["date"].dt.year.astype(str) if statement_period == "Annual" else merged["date"].dt.strftime("%Y Q%q")

        # --- Calculate ratios ---
        merged["Net Profit Margin"] = merged["net_income"] / merged["sales"]
        merged["Gross Margin"] = (merged["sales"] - merged["cost_of_goods"]) / merged["sales"]
        merged["ROA"] = merged["net_income"] / merged["total_assets"]
        merged["ROE"] = merged["net_income"] / merged["total_equity"]
        merged["Debt to Equity"] = merged["total_liabilities"] / merged["total_equity"]
        merged["Debt Ratio"] = merged["total_liabilities"] / merged["total_assets"]
        merged["Equity Ratio"] = merged["total_equity"] / merged["total_assets"]
        merged["Interest Coverage"] = merged["pretax_income"] / merged["interest_expense"]
        if "current_liabilities" in merged.columns:
            merged["Operating Cash Flow Ratio"] = merged["net_cash_from_operating_activities"] / merged["current_liabilities"]
        else:
            merged["Operating Cash Flow Ratio"] = None

        # Select and pivot
        ratio_cols = [
            "Period",
            "Net Profit Margin",
            "Gross Margin",
            "ROA",
            "ROE",
            "Debt to Equity",
            "Debt Ratio",
            "Equity Ratio",
            "Interest Coverage",
            "Operating Cash Flow Ratio",
        ]

        ratio_df = merged[ratio_cols].drop_duplicates("Period").set_index("Period").T

        # Format: percentage and decimal separation
        percent_rows = ["Net Profit Margin", "Gross Margin", "ROA", "ROE", "Debt Ratio", "Equity Ratio"]
        float_rows = ["Debt to Equity", "Interest Coverage", "Operating Cash Flow Ratio"]

        def try_format(val, as_percent=False):
            try:
                val = float(val)
                return f"{val * 100:.1f}%" if as_percent else f"{val:.2f}"
            except:
                return "—"

        for row in ratio_df.index:
            if row in percent_rows:
                ratio_df.loc[row] = ratio_df.loc[row].apply(lambda x: try_format(x, as_percent=True))
            elif row in float_rows:
                ratio_df.loc[row] = ratio_df.loc[row].apply(lambda x: try_format(x, as_percent=False))

        # Reset index so ratios become a column
        ratio_df.reset_index(inplace=True)
        ratio_df.rename(columns={"index": "Ratio"}, inplace=True)

        # Display
        st.data_editor(
            ratio_df,
            use_container_width=True,
            hide_index=True
        )

# -------------------------
# TRANSPOSE + CLEANUP + CONSISTENT UNIT FORMATTING
# -------------------------
if statement_tab == "Income Statement" and not filtered_df.empty:

    label = "Income Statement"   # ✅ define label here

    if "date" not in filtered_df.columns:
        st.error("No date column found in income statement.")
    else:
        # Keep raw numeric values
        ordered_cols = ["date"] + [c for c in filtered_df.columns if c != "date"]
        df_clean = filtered_df[ordered_cols].copy()

        # Convert and sort by date
        df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
        df_clean = df_clean.sort_values("date")

        # Transpose
        df_transposed = (
            df_clean.set_index("date")
                    .T
                    .reset_index()
                    .rename(columns={"index": label})
        )

    # Remove act_symbol & period rows
    df_transposed = df_transposed[~df_transposed[label].isin(["act_symbol", "period"])]

    # Clean metric names
    df_transposed[label] = df_transposed[label].str.replace("_", " ").str.title()

    # Format date columns
    formatted_cols = []
    for col in df_transposed.columns:
        if isinstance(col, pd.Timestamp):
            if statement_period == "Annual":
                formatted_cols.append(col.strftime("%Y"))
            else:
                q = (col.month - 1) // 3 + 1
                formatted_cols.append(f"{col.year} Q{q}")
        else:
            formatted_cols.append(col)

    df_transposed.columns = formatted_cols

    # Determine best unit scale
    def try_parse(x):
        try:
            return float(str(x).replace(",", ""))
        except:
            return None

    numeric_values = []
    for col in df_transposed.columns:
        if col != label:
            numeric_values.extend(
                [try_parse(x) for x in df_transposed[col] if try_parse(x) is not None]
            )

    max_abs_val = max(abs(v) for v in numeric_values) if numeric_values else 1

    if max_abs_val >= 1e12:
        scale, suffix = 1e12, "T"
    elif max_abs_val >= 1e9:
        scale, suffix = 1e9, "B"
    elif max_abs_val >= 1e6:
        scale, suffix = 1e6, "M"
    elif max_abs_val >= 1e3:
        scale, suffix = 1e3, "K"
    else:
        scale, suffix = 1, ""

    # Format numeric columns
    def format_scaled(x):
        val = try_parse(x)
        return f"{val / scale:,.2f}" if val is not None else "—"

    for col in df_transposed.columns:
        if col != label:
            df_transposed[col] = df_transposed[col].apply(format_scaled)

    # Caption
    st.caption(f"All values shown in **{suffix}** (e.g., 1.25{suffix} = {int(scale):,})")

    # Output table
    st.data_editor(
        df_transposed,
        use_container_width=True,
        hide_index=True,
        key="income_statement_transposed_editor",
        column_config={
            label: st.column_config.TextColumn(label)
        }
    )
