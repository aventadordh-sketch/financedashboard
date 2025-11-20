#single_stock_charts
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pages.single_stock.utils import get_historical_data

def display_charts(ticker, yf_df, dolt_df, using_dolthub):
    # -----------------------------
    # Initial check
    # -----------------------------
    if not ticker:
        st.info("Please select a ticker.")
        return

    # -----------------------------
    # Choose Data Source
    # -----------------------------
    df = dolt_df.copy() if using_dolthub else yf_df.copy()

    # -----------------------------
    # Normalize the 'date' Column
    # -----------------------------
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        elif "Date" in df.columns:
            df.rename(columns={"Date": "date"}, inplace=True)
        else:
            df = df.reset_index()
            if "index" in df.columns:
                try:
                    df["date"] = pd.to_datetime(df["index"], errors="coerce")
                except Exception:
                    st.error("❌ Could not parse any valid date column.")
                    return

    if "date" not in df.columns:
        st.error("❌ No recognizable 'date' column found.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().all():
        st.error("❌ All date values are invalid. Cannot display charts.")
        return

    df = df.sort_values("date")

    # -----------------------------
    # 📉 Price Chart
    # -----------------------------
    st.markdown("### 📉 Price Chart")
    col1, col2 = st.columns(2)

    with col1:
        chart_type = st.radio("Chart Type", ["Line", "Candlestick"], horizontal=True)
    with col2:
        time_range = st.selectbox("Time Range", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=2)

    hist = get_historical_data(ticker, period=time_range)

    if not hist.empty:
        fig = go.Figure()

        if chart_type == "Line":
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"))
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
            yaxis_title="Price",
            showlegend=False,
            margin=dict(l=40, r=40, t=30, b=40),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#fff")
        )

        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 📊 Financial Metric Chart
    # -----------------------------
    metrics_map = {
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

    df_plot = dolt_df if using_dolthub else yf_df
    if "date" in df_plot.columns:
        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
    else:
    # if no 'date' column at all → reset index
        df_plot = df_plot.reset_index()
        if "index" in df_plot.columns:
            df_plot.rename(columns={"index": "date"}, inplace=True)

    df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")

    if using_dolthub:
        available_metrics = [label for label, col in metrics_map.items() if col in df_plot.columns and df_plot[col].notna().any()]
    else:
        available_metrics = sorted(df_plot.columns.drop("date").tolist())
        metrics_map = {col: col for col in available_metrics}

    # -----------------------------
    # Determine Currency Label
    # -----------------------------
    if using_dolthub:
        currency = "USD"
    else:
        currency = "USD"
        if "currency" in df_plot.columns:
            unique_currency = df_plot["currency"].dropna().unique()
            if len(unique_currency) == 1:
                currency = unique_currency[0]

    if available_metrics:
        st.markdown("### 📊 Financial Metric Over Time")
        statement_period = st.radio("Select Frequency:", ["Annual", "Quarterly"], horizontal=True)
        selected_label = st.selectbox("Choose a Metric to Visualize:", available_metrics, key="metric_select")
        selected_col = metrics_map[selected_label]

        df_metric = df_plot.copy()
        if using_dolthub and "period" in df_metric.columns:
            df_metric["period"] = df_metric["period"].astype(str).str.upper()
            period_key = "YEAR" if statement_period == "Annual" else "QUARTER"
            df_metric = df_metric[df_metric["period"] == period_key]

        df_metric["Period"] = df_metric["date"].dt.to_period("Q" if statement_period == "Quarterly" else "Y").astype(str)
        grouped = df_metric.groupby("Period")[selected_col].sum().dropna().reset_index()

        if not grouped.empty:
            values = grouped[selected_col].astype(float).tolist()
            periods = grouped["Period"].tolist()
            max_val = max(abs(v) for v in values)
            scale, suffix = 1, ""
            if max_val >= 1e12:
                scale, suffix = 1e12, "T"
            elif max_val >= 1e9:
                scale, suffix = 1e9, "B"
            elif max_val >= 1e6:
                scale, suffix = 1e6, "M"
            elif max_val >= 1e3:
                scale, suffix = 1e3, "K"

            formatted_values = [f"{v/scale:,.2f}" for v in values]
            max_val_actual = max(values)
            min_val_actual = min(values)
            colors = [
                "#6FCF97" if v == max_val_actual else "#D87C6E" if v == min_val_actual else "cornflowerblue"
                for v in values
            ]

            fig = go.Figure(go.Bar(
                x=periods,
                y=values,
                text=formatted_values,
                textposition="outside",
                marker=dict(color=colors)
            ))

            fig.update_layout(
                title=selected_label,
                xaxis_title="Period",
                yaxis_title=f"{selected_label} ({currency}, {suffix})",
                margin=dict(l=40, r=40, t=60, b=40),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fff")
            )

            st.caption(f"Chart values shown in **{suffix}** ({currency}) — e.g., 1.25{suffix} = {int(scale):,} {currency}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available for '{selected_label}'")

