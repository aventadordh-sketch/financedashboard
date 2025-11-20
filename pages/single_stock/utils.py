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
    
def style_growth_from_prev(df, label_col):
    import numpy as np

    def try_float(x):
        try:
            s = str(x).replace(",", "").strip()
            if s.endswith("%"):
                return float(s[:-1])     # convert "5.23%" → 5.23
            return float(s)
        except:
            return np.nan

    def apply_row_gradient(row):
        result = [""]

        for i in range(1, len(row)):
            prev, curr = row[i - 1], row[i]
            prev_f = try_float(prev)
            curr_f = try_float(curr)

            if np.isnan(prev_f) or np.isnan(curr_f):
                result.append("")
            elif curr_f > prev_f:
                result.append("color: #6FCF97")   # green
            elif curr_f < prev_f:
                result.append("color: #D87C6E")   # red
            else:
                result.append("color: white")
        
        return result

    return df.style.apply(apply_row_gradient, axis=1, subset=df.columns[1:])

def style_yoy_percent(df, label_col):
    import numpy as np

    def try_float(x):
        try:
            s = str(x).replace("%", "").replace(",", "").strip()
            return float(s)
        except:
            return np.nan

    def apply_color(row):
        result = [""]  # skip label column
        for val in row[1:]:
            f = try_float(val)
            if np.isnan(f):
                result.append("")
            elif f > 0:
                result.append("color: #6FCF97")  # green
            elif f < 0:
                result.append("color: #D87C6E")  # red
            else:
                result.append("color: white")
        return result

    return df.style.apply(apply_color, axis=1)


