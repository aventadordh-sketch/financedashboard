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
    import pandas as pd

    if df is None or df.empty:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["period"] = df["period"].astype(str).str.upper()

    period_filter = "YEAR" if statement_period == "Annual" else "QUARTER"
    df = df[df["period"] == period_filter]

    if df.empty:
        return None

    df = df.sort_values("date")

    # Metrics only — no metadata
    metric_cols = [c for c in df.columns if c not in ["date", "period", "act_symbol"]]
    df_clean = df[["date"] + metric_cols].set_index("date").T

    # Compute YoY % = (curr / prev - 1) * 100
    yoy = df_clean.pct_change(axis=1) * 100
    yoy = yoy.round(2)

    # Format as percent strings ("5.23%")
    yoy = yoy.applymap(lambda v: f"{v:.2f}%" if pd.notnull(v) else "—")

    # Insert label column
    yoy.insert(0, label_name, yoy.index.str.replace("_", " ").str.title())
    yoy.reset_index(drop=True, inplace=True)

    # Clean column names: 2022 or 2022 Q1
    yoy.columns = [label_name] + [
        col.strftime("%Y") if statement_period == "Annual"
        else f"{col.year} Q{((col.month - 1)//3)+1}"
        for col in df_clean.columns
    ]

    return yoy





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

        view_mode = st.segmented_control(
            "",
            options=["Values", "YoY Change"],
        )

        if view_mode == "Values":
            render_income_statement_values(dolt_df, statement_period, style_growth_from_prev)

        else:
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
        def format_balance_sheet(df, label):
            if df is None or df.empty:
                return st.info(f"No {label} data available.")

            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["period"] = df["period"].str.upper()
            df = df[df["period"] == ("YEAR" if statement_period == "Annual" else "QUARTER")]

            if df is None or df.empty:
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

            # Caption for unit scaling
            st.caption(f"All values shown in **{suffix}** (e.g., 1.25{suffix} = {int(scale):,})")

            # Apply growth-based styling
            styled_df = style_growth_from_prev(df_transposed, label)

            # Render with gradient based on previous column
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        format_balance_sheet(assets_df, "Assets")
        format_balance_sheet(liabilities_df, "Liabilities")
        format_balance_sheet(equity_df, "Equity")

# -----------------------------
# Cash Flow Statement
# -----------------------------
    elif statement_tab == "Cash Flow":
        dolt_df = get_dolthub_cash_flow(ticker)

        # ✅ Step 2: Build cash_flow_df using ordered fields (safe for DoltHub + fallback)
        meta_columns = ["date", "period", "act_symbol"]

        # Meta fields available in DoltHub
        available_meta = []
        if dolt_df is not None:
            available_meta = [col for col in meta_columns if col in dolt_df.columns]

        # Cash flow fields available in DoltHub
        available_cash_cols = []
        if dolt_df is not None:
            available_cash_cols = [col for col in CASH_FLOW_FIELDS if col in dolt_df.columns]

        # FINAL: Build DataFrame only if dolt_df exists
        if dolt_df is not None:
            selected_cols = available_meta + available_cash_cols

            if selected_cols:  # only build DF if at least 1 column exists
                cash_flow_df = dolt_df[selected_cols].copy()
            else:
                cash_flow_df = pd.DataFrame()
        else:
            # Yahoo fallback: no DoltHub data
            cash_flow_df = pd.DataFrame()


        # ✅ Step 3: Format and render
        def format_cash_flow(df, label="Cash Flow"):
            if df is None or df.empty:
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

            # Format column names (date columns)
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

            # Detect scaling
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

            styled_df = style_growth_from_prev(df_transposed, label)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # ✅ Render it
        format_cash_flow(cash_flow_df)

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

            # Apply growth-based styling
        styled_df = style_growth_from_prev(df_transposed, label)

            # Render with gradient based on previous column
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
