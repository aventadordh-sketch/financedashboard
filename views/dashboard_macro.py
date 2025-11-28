import streamlit as st
import plotly.express as px
from urllib.parse import urlparse
# Import all necessary functions, including the new synthesis one
from src.economic_utils import get_macro_data, synthesize_indicator_conclusion 

def render_macro_economic_section():
    st.markdown("### 🏦 Macroeconomic Indicators (FRED Data)")
    
    # --- 1. Define Indicator Groups with Units ---
    
    leading_indicators = [
        ("Average Weekly Hours (Manufacturing)", "AWHMAN", "Hours"),
        ("Initial Jobless Claims", "ICSA", "Thousands"),
        ("Manufacturers' New Orders", "AMDMNO-US", "Millions USD"), 
        ("Vendor Performance Index (ISM)", "PMICD", "Index"),
        ("Non-Defense Capital Goods Orders", "NEWORDER", "Millions USD"),
        ("Building Permits (New Housing)", "PERMIT", "Units"),
        ("S&P 500 Index (Stock Prices)", "SP500", "Index"),
        ("Consumer Expectations", "UMCSENT", "Index"), 
        ("Personal Consumption Expenditures", "PCE", "Billions USD"), 
    ]
    
    monetary_inflation = [
        ("Federal Funds Rate (Current)", "FEDFUNDS", "Percent"),
        ("10-Year Treasury Yield", "DGS10", "Percent"),
        ("US Core Inflation (CPI)", "CPILFESL", "Index"),
        ("Inflation, consumer prices for the United States", "FPCPITOTLZGUSA", "Percent"),
        ("M2 Money Supply", "M2SL", "Billions USD"),
    ]
    
    current_lagging = [
        ("Unemployment Rate", "UNRATE", "Percent")
    ]

    # Map group names to their options (used for synthesis)
    grouped_indicators = {
        "1. Leading Economic Indicators (Future Trends)": leading_indicators,
        "2. Monetary & Inflation (Policy Focus)": monetary_inflation,
        "3. Lagging Indicators (Past Confirmation)": current_lagging,
    }
    
    # --- 2. UI: Grouping and Slider ---
    
    col_select, col_slider = st.columns([2, 1])

    with col_select:
        st.markdown("##### 📈 Indicator Type Selection")
        selected_group_name = st.radio(
            "Select Indicator Group:",
            list(grouped_indicators.keys()),
            key='indicator_group',
            index=0 
        )
        
        selected_group_list = grouped_indicators[selected_group_name]
        
        st.markdown("###### Choose Indicator:")
        graph_option = st.selectbox(
            f"Indicator from '{selected_group_name}':",
            selected_group_list,
            format_func=lambda x: x[0],
            key='final_indicator_select',
            label_visibility='collapsed'
        )

    with col_slider:
        st.markdown("##### Historical Range (Years):")
        years = st.slider(
            "Years:",
            min_value=1,
            max_value=50, 
            value=5,      
            step=1,
            label_visibility='collapsed'
        )
        
        # --- MISSING UI ELEMENT 1: ANALYSIS FOCUS INPUT ---
        analysis_focus = st.text_input(
            "AI Analysis Focus:", 
            value="Impact of inflation and interest rates on recession risk"
        )
        
        # Initialize session state for the conclusion if it doesn't exist
        if 'macro_conclusion' not in st.session_state:
             st.session_state['macro_conclusion'] = None


    # --- 3. Fetch Data and Plot ---
    
    # Unpack the tuple
    if len(graph_option) == 3:
        label, series_id, unit = graph_option
    else:
        st.error("Error: Indicator structure is corrupted. Please restart.")
        return 

    # Fetch Data
    df = get_macro_data(series_id, label, years=years) 
    
    if df is not None:
        fig = px.line(df, x='Date', y='Value', title=f"{label} (Last {years} Years)")
        y_axis_title = f"{label} ({unit})"
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            height=400, 
            xaxis_title=None,
            yaxis_title=y_axis_title,
            template="plotly_white" 
        )
        st.plotly_chart(fig, use_container_width=True)

    
    # --- 4. NEW: AI SYNTHESIS BUTTON AND DISPLAY ---
    
    st.divider()
    
    # --- MISSING UI ELEMENT 2: SYNTHESIS BUTTON ---
    if st.button("🧠 Synthesize Economic Conclusion (Analyze ALL Indicators)", use_container_width=True):
        # We check if the synthesis logic exists in src/economic_utils.py 
        # (It was defined there in the previous step)
        if not get_macro_data: 
            st.error("Cannot synthesize: FRED API data fetching is currently failing.")
        else:
            with st.spinner("Asking Chief Economist (Gemini) to analyze current trends..."):
                # Call the backend synthesis function
                conclusion_text = synthesize_indicator_conclusion(grouped_indicators, analysis_focus)
                st.session_state['macro_conclusion'] = conclusion_text

    # Display the stored conclusion
    if st.session_state['macro_conclusion']:
        st.markdown("### **📝 Economic Synthesis & Outlook**")
        st.info(st.session_state['macro_conclusion'])