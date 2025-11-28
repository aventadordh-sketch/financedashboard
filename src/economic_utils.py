import streamlit as st
from fredapi import Fred
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any 

# CRITICAL FIX: Import the Gemini client getter from ai_functions.py
from src.ai_functions import get_gemini_client 
from google import genai # Added back because it's required for genai types in the function below
from google.genai import types # Added back because it's required for types in the function below

# --- FRED UTILS ---
@st.cache_resource
def get_fred_client():
    """Initializes and returns the FRED API client."""
    try:
        # FRED_API_KEY is retrieved from .streamlit/secrets.toml
        return Fred(api_key=st.secrets["FRED_API_KEY"])
    except KeyError:
        st.error("🔒 FRED API Key not found. Please ensure FRED_API_KEY is in your secrets.toml.")
        return None

@st.cache_data(ttl=86400)
def get_macro_data(series_id, label, years=2):
    """
    Fetches economic data series from FRED and returns it as a DataFrame.
    """
    fred = get_fred_client()
    if not fred: return None

    try:
        # Calculate start date using the 'years' parameter
        start_date = datetime.now() - timedelta(days=years*365)
        
        # Fetch data
        data = fred.get_series(series_id, observation_start=start_date)
        
        # Check if FRED returned valid data
        if data is None or data.empty:
            st.error(f"Error fetching {label} ({series_id}): FRED returned no data. The API key may be invalid, or the Series ID may be retired.")
            return None
            
        df = pd.DataFrame(data, columns=['Value'])
        df.index.name = 'Date'
        df.reset_index(inplace=True)
        df['Series'] = label
        return df
    except Exception as e:
        # Catch network or other exceptions
        st.error(f"Error fetching {label} ({series_id}): {e}")
        return None

# --- AI SYNTHESIS UTILITY ---

@st.cache_data(ttl=3600) # Cache for 1 hour to prevent rapid re-runs
def synthesize_indicator_conclusion(
    grouped_indicators: Dict[str, List[Tuple[str, str, str]]], 
    analysis_focus: str
) -> str:
    """
    Gathers the latest values for all indicators and sends them to Gemini 
    to synthesize a coherent economic conclusion.
    """
    
    # FIX: Initialize the correct clients!
    _gemini_client = get_gemini_client() # <-- CORRECTLY uses the function from ai_functions.py
    if not _gemini_client: 
        return "Synthesis failed: Gemini API key not found or client could not be initialized."

    _fred_client = get_fred_client()
    if not _fred_client:
         return "Synthesis failed: FRED client not initialized."


    latest_data = {}
    
    # 1. GATHER LATEST VALUES FOR ALL INDICATORS
    for group_name, indicators in grouped_indicators.items():
        latest_data[group_name] = []
        for label, series_id, unit in indicators:
            try:
                # Fetch only the latest value using the FRED client
                latest_value = _fred_client.get_series_latest_release(series_id).iloc[-1]
                series_info = _fred_client.get_series_info(series_id)
                frequency = series_info.get('frequency', 'Unknown')
                
                latest_data[group_name].append({
                    "Indicator": label,
                    "Series ID": series_id,
                    "Latest Value": latest_value,
                    "Unit": unit,
                    "Frequency": frequency
                })
            except Exception:
                # Skip indicators that fail to fetch (e.g., deleted series ID)
                continue

    # 2. FORMAT DATA FOR GEMINI
    data_string = "Economic Data Points:\n\n"
    for group_name, data_list in latest_data.items():
        if data_list:
            data_string += f"--- {group_name} ---\n"
            for item in data_list:
                # Format: [Indicator] (Series ID) [Frequency]: [Value] [Unit]
                data_string += (
                    f"{item['Indicator']} ({item['Series ID']}) [{item['Frequency']}]: "
                    f"{item['Latest Value']:.2f} {item['Unit']}\n"
                )
    
    # 3. PROMPT ENGINEERING
    system_instruction = (
        "You are a Chief Economist providing a macro-economic outlook. "
        "Your analysis must be objective, concise, and focused on the interaction between indicators. "
        "Use the provided data exclusively."
    )

    user_query = (
        f"Analyze the following economic data across all provided groups (Leading, Monetary, Lagging). "
        f"Formulate a conclusion based on how these recent values converge or diverge, especially concerning the user's focus on: '{analysis_focus}'. "
        f"Structure your response with:\n\n"
        f"1. **Summary of Key Trends:** (Identify 2-3 most important signals, e.g., strong labor market despite high rates).\n"
        f"2. **Conclusive Outlook:** (A single sentence predicting the short-term direction of the economy).\n\n"
        f"Data to Analyze:\n{data_string}"
    )

    try:
        # 4. CALL GEMINI API using the correct GEMINI client
        response = _gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_query,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text
    except Exception as e:
        return f"AI Synthesis Error: Failed to generate conclusion. Details: {e}"