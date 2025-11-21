import streamlit as st
import pandas as pd
from .utils import (
    get_dolthub_cash_flow,
    get_dolthub_income_statement,
    get_dolthub_balance_sheet_assets,
    get_dolthub_cash_flow,
    format_num,
    render_card,
    style_growth_from_prev,
    style_yoy_percent,
    CASH_FLOW_FIELDS,
)

# Optional: If you need get_dolthub_cash_flow or other shared utilities
# from .utils import get_dolthub_cash_flow
def render_income_statement_values(dolt_df, statement_period, style_growth_from_prev):
    label = "Income Statement"

    if dolt_df is None or dolt_df.empty:
        st.info("No income statement data found.")
        return

    df = dolt_df.copy()
    df["period"] = df["period"].astype(str).str.upper()
    period_value = "YEAR" if statement_period == "Annual" else "QUARTER"
    filtered_df = df[df["period"] == period_value].copy()

    if filtered_df.empty:
        st.info(f"No {statement_period.lower()} income statement data available.")
        return

    # Format numeric columns
    for col in filtered_df.select_dtypes(include=["float", "int"]).columns:
        filtered_df[col] = filtered_df[col].apply(
            lambda x: f"{x:,.0f}" if pd.notnull(x) else "—"
        )

    ordered_cols = ["date"] + [c for c in filtered_df.columns if c != "date"]
    df_clean = filtered_df[ordered_cols].copy()

    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")
    df_clean = df_clean.sort_values("date")

    df_transposed = (
        df_clean.set_index("date")
                .T
                .reset_index()
                .rename(columns={"index": label})
    )

    df_transposed = df_transposed[~df_transposed[label].isin(["act_symbol", "period"])]
    df_transposed[label] = df_transposed[label].str.replace("_", " ").str.title()

    # Format column headers (Years, Quarters)
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

    # Scaling logic
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

    def format_scaled(x):
        val = try_parse(x)
        return f"{val / scale:,.2f}" if val is not None else "—"

    for col in df_transposed.columns:
        if col != label:
            df_transposed[col] = df_transposed[col].apply(format_scaled)

    st.caption(f"All values shown in **{suffix}**")

    styled_df = style_growth_from_prev(df_transposed, label)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def compute_yoy_table(df, label_name, statement_period):
    """
    Computes YoY % change table based on date, for Income, Balance Sheet, Cash Flow.
    - Filters by period (YEAR / QUARTER) if 'period' column exists
    - Returns a transposed table with human-readable column labels (YYYY or YYYY Qx)
    - Output is ready to pass into style_yoy_percent(...)
    """
    if df is None or df.empty:
        return None

    df = df.copy()

    # Must have a date column
    if "date" not in df.columns:
        return None

    # Normalize date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # If period exists, filter by Annual / Quarterly
    if "period" in df.columns:
        df["period"] = df["period"].astype(str).str.upper()
        period_filter = "YEAR" if statement_period == "Annual" else "QUARTER"
        df = df[df["period"] == period_filter]

    # Drop rows with no date and sort
    df = df.dropna(subset=["date"]).sort_values("date")

    # Need at least 2 periods to compute change
    if df["date"].nunique() < 2:
        return None

    # Keep only numeric value columns (exclude metadata)
    numeric_cols = []
    for c in df.columns:
        if c in ["date", "act_symbol", "period"]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)

    if not numeric_cols:
        return None

    # Wide format: index = date, columns = metrics
    wide = (
        df[["date"] + numeric_cols]
        .drop_duplicates(subset="date")
        .set_index("date")
    )

    # YoY % change by date
    yoy = wide.pct_change() * 100
    yoy = yoy.iloc[1:]  # drop the first NaN row

    if yoy.empty:
        return None

    # Transpose for display: rows = metrics, columns = dates
    yoy_t = yoy.T.reset_index()
    yoy_t = yoy_t.rename(columns={"index": label_name})

    # Build readable period labels from the date index
    # yoy.index is a DatetimeIndex of the dates we used
    dates = list(yoy.index)

    pretty = []
    for d in dates:
        if isinstance(d, pd.Timestamp):
            if statement_period == "Annual":
                pretty.append(d.strftime("%Y"))
            else:
                q = (d.month - 1) // 3 + 1
                pretty.append(f"{d.year} Q{q}")
        else:
            pretty.append(str(d))

    # Ensure unique column names (required for Styler)
    final_cols = [label_name]
    seen = {}
    for col in pretty:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 1
            final_cols.append(col)

    # Now length matches exactly: 1 label + len(dates)
    yoy_t.columns = final_cols

    # Format as percentages
    for c in yoy_t.columns[1:]:
        yoy_t[c] = yoy_t[c].apply(
            lambda x: f"{x:.2f}%" if pd.notnull(x) else "—"
        )

    # --- Clean metric names to match value tables ---
    yoy_t[label_name] = (
        yoy_t[label_name]
        .astype(str)
        .str.replace("_", " ")
        .str.title()
    )

    return yoy_t


# 🔍 Display financial statements + key ratios
def display_fundamentals(
    statement_tab,
    statement_period,
    dolt_df=None,
    assets_df=None,
    liabilities_df=None,
    equity_df=None,
    get_dolthub_cash_flow=None,
    ticker=None,
    yf_df=None  # 👈 Add this
):
    
    import pandas as pd

    # ---- 🔥 ALWAYS initialize cash_flow_df here ----
    cash_flow_df = pd.DataFrame()

    # Load from DoltHub once (so it is available for ALL tabs)
    if get_dolthub_cash_flow is not None and ticker:
        try:
            cf = get_dolthub_cash_flow(ticker)
            if isinstance(cf, pd.DataFrame):
                cash_flow_df = cf
        except Exception as e:
            st.warning(f"Error loading DoltHub cash flow: {e}")

    if dolt_df is not None:
        # Original DoltHub logic
        ...
    elif yf_df is not None:
        # Add Yahoo Finance fallback logic here
        st.subheader("📘 Yahoo Finance Financial Summary")
        st.write("Net Income (latest):", yf_df.get("net_income", pd.NA))
        st.write("Revenue (latest):", yf_df.get("Total Revenue", pd.NA))
        # You can expand this as needed
    else:
        st.warning("No financial data available.")

    
# -----------------------------
# Income Statement
# -----------------------------
    if statement_tab == "Income Statement":

        # SINGLE TOGGLE BUTTON
        yoy_toggle = st.checkbox("% Change", value=False)

        if not yoy_toggle:
            # Default values table
            render_income_statement_values(
                dolt_df, statement_period, style_growth_from_prev
            )

        else:
            # YoY table
            yoy_df = compute_yoy_table(
                df=dolt_df,
                label_name="Income Statement",
                statement_period=statement_period
            )

            if yoy_df is None:
                st.info("Not enough data for YoY calculations.")
            else:
                styled = style_yoy_percent(yoy_df, "Income Statement")
                st.dataframe(styled, use_container_width=True, hide_index=True)

    # -----------------------------
    # Balance Sheet
    # -----------------------------
    elif statement_tab == "Balance Sheet":

        # One toggle for all parts of the balance sheet
        yoy_toggle = st.checkbox("% Change", value=False)

        def render_balance_sheet_section(df, label):
            if df is None or df.empty:
                return st.info(f"No {label} data available.")

            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["period"] = df["period"].astype(str).str.upper()

            period_value = "YEAR" if statement_period == "Annual" else "QUARTER"
            df = df[df["period"] == period_value]

            if df.empty:
                return st.info(f"No {statement_period.lower()} data available for {label}.")

            df = df.sort_values("date")
            ordered_cols = ["date"] + [
                col for col in df.columns
                if col not in ["date", "act_symbol", "period"]
            ]
            df_clean = df[ordered_cols].copy()

            if not yoy_toggle:
                # -------------------------
                # VALUES MODE (default)
                # -------------------------
                df_transposed = (
                    df_clean.set_index("date")
                            .T
                            .reset_index()
                            .rename(columns={"index": label})
                )

                # Clean names
                df_transposed[label] = df_transposed[label].str.replace("_", " ").str.title()

                # Format columns as years / quarters
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

                # Determine scale
                def try_parse(x):
                    try:
                        return float(str(x).replace(",", ""))
                    except:
                        return None

                numeric_values = [
                    try_parse(x)
                    for col in df_transposed.columns if col != label
                    for x in df_transposed[col]
                    if try_parse(x) is not None
                ]

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

                st.caption(f"All values shown in **{suffix}**")

                styled = style_growth_from_prev(df_transposed, label)
                st.dataframe(styled, use_container_width=True, hide_index=True)
                return

            # -------------------------
            # YOY MODE
            # -------------------------
            yoy_df = compute_yoy_table(
                df=df,
                label_name=label,
                statement_period=statement_period
            )

            if yoy_df is None:
                st.info("Not enough data for YoY calculations.")
                return

            styled = style_yoy_percent(yoy_df, label)
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # Render each subsection
        st.subheader("Assets")
        render_balance_sheet_section(assets_df, "Assets")

        st.subheader("Liabilities")
        render_balance_sheet_section(liabilities_df, "Liabilities")

        st.subheader("Equity")
        render_balance_sheet_section(equity_df, "Equity")


    # -----------------------------
    # Cash Flow Statement
    # -----------------------------
    elif statement_tab == "Cash Flow":

        # Load cash flow from DoltHub
        cash_flow_df = get_dolthub_cash_flow(ticker)

        # SINGLE TOGGLE BUTTON
        yoy_toggle = st.checkbox("% Change", value=False)

        def render_cash_flow(df, label="Cash Flow"):
            if df is None or df.empty:
                return st.info(f"No {label} data available.")

            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["period"] = df["period"].astype(str).str.upper()

            period_value = "YEAR" if statement_period == "Annual" else "QUARTER"
            df = df[df["period"] == period_value]

            if df.empty:
                return st.info(f"No {statement_period.lower()} {label.lower()} data available.")

            df = df.sort_values("date")

            # Always restrict to standard cash flow fields
            ordered_cols = ["date"] + [col for col in CASH_FLOW_FIELDS if col in df.columns]
            df_clean = df[ordered_cols].copy()

            # -----------------------------
            # YOY MODE
            # -----------------------------
            if yoy_toggle:
                yoy_df = compute_yoy_table(
                    df=df_clean,
                    label_name=label,
                    statement_period=statement_period
                )
                if yoy_df is None:
                    return st.info("Not enough data for YoY calculations.")
                styled = style_yoy_percent(yoy_df, label)
                st.dataframe(styled, use_container_width=True, hide_index=True)
                return

            # -----------------------------
            # VALUES MODE (default)
            # -----------------------------
            df_transposed = (
                df_clean.set_index("date")
                        .T
                        .reset_index()
                        .rename(columns={"index": label})
            )

            df_transposed[label] = df_transposed[label].str.replace("_", " ").str.title()

            # Format column names (years or quarters)
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

            # Determine best scaling
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

            def format_scaled(x):
                val = try_parse(x)
                return f"{val / scale:,.2f}" if val is not None else "—"

            for col in df_transposed.columns:
                if col != label:
                    df_transposed[col] = df_transposed[col].apply(format_scaled)

            st.caption(f"All values shown in **{suffix}**")

            styled_df = style_growth_from_prev(df_transposed, label)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Render it
        render_cash_flow(cash_flow_df)


#### Financial Ratios ####
    elif statement_tab == "Key Ratios":

        # Load CF (needed for Operating Cash Flow Ratio)
        cash_flow_df = get_dolthub_cash_flow(ticker) if get_dolthub_cash_flow else None

        # Validate
        if any(df is None or df.empty for df in [assets_df, liabilities_df, equity_df, cash_flow_df, dolt_df]):
            st.warning("Missing data from one or more financial statements. Please ensure all statements are loaded.")
            return

        # Normalize date & period
        for df in [assets_df, liabilities_df, equity_df, cash_flow_df, dolt_df]:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["period"] = df["period"].astype(str).str.upper()

        # Annual or Quarterly filter
        period_filter = "YEAR" if statement_period == "Annual" else "QUARTER"

        dolt_period = dolt_df[dolt_df["period"] == period_filter].copy()
        assets_period = assets_df[assets_df["period"] == period_filter].copy()
        liabilities_period = liabilities_df[liabilities_df["period"] == period_filter].copy()
        equity_period = equity_df[equity_df["period"] == period_filter].copy()
        cash_period = cash_flow_df[cash_flow_df["period"] == period_filter].copy()

        # Merge all statements
        merged = (
            dolt_period
            .merge(assets_period, on="date", suffixes=("", "_assets"))
            .merge(liabilities_period, on="date", suffixes=("", "_liab"))
            .merge(equity_period, on="date", suffixes=("", "_eq"))
            .merge(cash_period, on="date", suffixes=("", "_cf"))
        )

        merged.sort_values("date", inplace=True)

        # ---- FIXED PERIOD LABEL ----
        if statement_period == "Annual":
            merged["Period"] = merged["date"].dt.year.astype(str)
        else:  # Quarterly
            q = merged["date"].dt.quarter.astype(str)
            y = merged["date"].dt.year.astype(str)
            merged["Period"] = y + " Q" + q

        # ---- Calculate Ratios ----
        merged["Net Profit Margin"] = merged["net_income"] / merged["sales"]
        merged["Gross Margin"] = (merged["sales"] - merged["cost_of_goods"]) / merged["sales"]
        merged["ROA"] = merged["net_income"] / merged["total_assets"]
        merged["ROE"] = merged["net_income"] / merged["total_equity"]
        merged["Debt to Equity"] = merged["total_liabilities"] / merged["total_equity"]
        merged["Debt Ratio"] = merged["total_liabilities"] / merged["total_assets"]
        merged["Equity Ratio"] = merged["total_equity"] / merged["total_assets"]
        merged["Interest Coverage"] = merged["pretax_income"] / merged["interest_expense"]

        if "current_liabilities" in merged.columns:
            merged["Operating Cash Flow Ratio"] = (
                merged["net_cash_from_operating_activities"] / merged["current_liabilities"]
            )
        else:
            merged["Operating Cash Flow Ratio"] = None

        # Select and pivot for display
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

        # ---- Formatting ----
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
            if row in float_rows:
                ratio_df.loc[row] = ratio_df.loc[row].apply(lambda x: try_format(x, as_percent=False))

        # Reset index
        ratio_df.reset_index(inplace=True)
        ratio_df.rename(columns={"index": "Ratio"}, inplace=True)

        # Apply cash-flow style coloring
        label_col = ratio_df.columns[0]  # "Ratio"
        styled_df = style_growth_from_prev(ratio_df, label_col)

        st.dataframe(styled_df, use_container_width=True, hide_index=True)


