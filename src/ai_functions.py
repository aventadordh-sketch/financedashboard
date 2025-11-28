import streamlit as st
from google import genai
from playwright.sync_api import sync_playwright
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any # Added typing for clarity

from src.config_gemini import get_gemini_client # import gemini client function

# IMPORTANT: We need to import the FRED-related functions from economic_utils 
# to make the synthesis function work, so we'll place the synthesis logic 
# in economic_utils.py as originally planned, and ensure this file provides 
# the necessary GEMINI client.

# --- 1. Client Initialization ---
# @st.cache_resource
# def get_gemini_client():
#     """
#     Initializes the Gemini client from secrets.
#     """
#     try:
#         # Check if the key exists before trying to initialize
#         if "GEMINI_API_KEY" not in st.secrets:
#              st.error("🔒 GEMINI_API_KEY not found in .streamlit/secrets.toml.")
#              return None
             
#         return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
#     except Exception as e:
#         st.error(f"Error initializing Gemini: {e}")
#         return None

# --- 2. Scraping & Summarization (Stealth Mode Enhanced) ---
@st.cache_data(ttl=86400)
def scrape_and_summarize(urls: list, topic: str, _client) -> dict:
    """
    Scrapes URLs using Playwright with advanced fingerprinting to bypass protections.
    """
    if not _client:
        return {}
        
    results = {}
    results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    progress_bar = st.progress(0, text="Starting news analysis...")
    
    with sync_playwright() as p:
        # Launch Chrome headless
        browser = p.chromium.launch(headless=True)
        
        for i, url in enumerate(urls):
            progress_bar.progress((i + 1) / len(urls), text=f"Reading {i+1}/{len(urls)}: {url}")
            
            try:
                # 🛠️ STEALTH FIX v2: Advanced Context Emulation
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                    timezone_id="America/New_York",
                    extra_http_headers={
                        "Accept-Language": "en-US,en;q=0.9",
                        "Referer": "https://www.google.com/",
                        "Upgrade-Insecure-Requests": "1"
                    }
                )
                page = context.new_page()
                
                # 1. Navigate
                try:
                    page.goto(url, timeout=60000, wait_until="domcontentloaded")
                    try:
                        page.wait_for_load_state("networkidle", timeout=5000)
                    except:
                        pass 
                except Exception as e:
                    print(f"Navigation warning for {url}: {e}")

                # 2. 🖱️ Human Behavior: Scroll
                page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
                time.sleep(1)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(2)
                
                # 3. Targeted Extraction
                content = ""
                selectors = ["article", "[role='main']", ".ArticleBody-articleBody", ".paywall-article", ".story-text", "main", "body"]
                
                for selector in selectors:
                    try:
                        if page.locator(selector).count() > 0:
                            candidate = page.locator(selector).first.inner_text()
                            if len(candidate.strip()) > 500: 
                                content = candidate
                                break
                    except:
                        continue
                
                raw_text = content if content else page.inner_text("body")
                
                # Cleanup tabs
                page.close()
                context.close()
                
                # 4. Validation
                if not raw_text or len(raw_text.strip()) < 200:
                    results[url] = "⚠️ Content too short or blocked. Try a different source URL."
                    continue

                # --- Summarization Logic ---
                max_input_length = 15000
                if len(raw_text) > max_input_length:
                    raw_text = raw_text[:max_input_length] 
                
                # 🎯 REFINED PROMPT FOR MARKET OVERVIEW PAGES
                prompt = (
                    f"You are a professional financial analyst. The text below is scraped from a market news overview page. "
                    f"Scan the headlines, short summaries, and data points to identify the **top 3 most significant** market-moving stories or trends related to '{topic}'. "
                    f"Synthesize these findings into 3 concise, high-impact bullet points. "
                    f"Ignore navigation menus, ads, and generic site links. Content: \n---\n{raw_text}"
                )

                response = _client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=prompt,
                )
                
                results[url] = response.text
                
            except Exception as e:
                results[url] = f"Error processing URL: {e}"
        
        browser.close()
            
    progress_bar.empty()
    return results