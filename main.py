import streamlit as st
import yfinance as yf
from typing import Dict, Optional
from groq import Groq
import pandas as pd
import altair as alt
import numpy as np
import os

API_KEY = os.getenv("GROQ_API_KEY")

# === SET API KEY ===

# === LLM EXTRACT FUNCTION ===
def extract_stock_from_query(query: str) -> str:
    try:
        client = Groq(api_key=API_KEY)
        system_prompt = {
            "role": "system",
            "content": (
                "You are a financial assistant. Extract stock symbols or company names from the user's query. "
                "Return only the company ticker. Example, if the company name is Apple, the end result should be just one word; AAPL. "
                "If unclear, just say 'general' to indicate it's a general query."
            )
        }
        messages = [
            system_prompt,
            {"role": "user", "content": query}
        ]

        completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    temperature=0.6,
    max_completion_tokens=4096,
    top_p=0.95,
    stream=False,
    stop=None,
)

        result = completion.choices[-1].message.content.strip()
        return result
    except Exception as e:
        st.error(f"❌ LLM Error: {str(e)}")
        return "general"

# === STOCK DATA FETCHER ===
def fetch_stock_info(symbol: str):
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        return {
            "name": info.get("longName", symbol.upper()),
            "symbol": symbol.upper(),
            "market_cap": info.get("marketCap", "N/A"),
            "sector": info.get("sector", "N/A"),
            "current_price": info.get("currentPrice", "N/A"),
            "summary": info.get("longBusinessSummary", "No description available.")
        }
    except Exception as e:
        st.error(f"❌ Could not retrieve data for '{symbol}'. Please check the company name or symbol.")
        return None

# === FORMAT STOCK INFO ===
def format_stock_info(stock: Dict) -> str:
    market_cap = stock.get("market_cap", "N/A")
    if isinstance(market_cap, (int, float)):
        market_cap = f"${market_cap / 1e9:.2f}B"

    current_price = stock.get("current_price", "N/A")
    if isinstance(current_price, (int, float)):
        current_price = f"${current_price:.2f}"

    return (
        f"**{stock['name']} ({stock['symbol']})**\n\n"
        f"📊 **Sector:** {stock['sector']}\n"
        f"💰 **Market Cap:** {market_cap}\n"
        f"📈 **Current Price:** {current_price}\n\n"
        f"📌 **About:** {stock['summary']}"
    )

# === FUNCTION TO FETCH STOCK DATA AND GENERATE A CHART ===
@st.cache_data  # Apply caching
def generate_stock_chart(symbol: str):
    try:
        data = yf.download(symbol, period="1y").reset_index()
        if data.empty:
            st.warning(f"⚠️ No data found for {symbol}. Please try again later.") # Use warning instead of error
            return None

        # Convert 'Date' column to datetime objects
        try:
            data['Date'] = pd.to_datetime(data['Date'])
        except KeyError:
            st.error(f"❌ Date column not found in data for {symbol}.")
            return None

        # Define the color gradient
        gradient_colors = ['#FFD700', '#FFA500', '#FF4500']  # Example: Gold to Orange to Red

        # Create Altair chart with gradient line
        try:
            chart = alt.Chart(data).mark_line().encode(
                x=alt.X('Date:T', title='Date'),  # T specifies time type
                y=alt.Y('Close:Q', title='Close Price'),
                color=alt.Color('Close:Q', scale=alt.Scale(range=gradient_colors)),
                tooltip=['Date:T', 'Close:Q']
            ).properties(
                title=f'{symbol.upper()} Stock Price (1 Year)'
            ).interactive()
            return chart
        except Exception as e:
            st.error(f"❌ Chart rendering error for {symbol}: {e}")
            return None


    except Exception as e:
        st.error(f"❌ Chart Generation Error for {symbol}: {str(e)}")
        return None

# === FUNCTION TO GET KPIS ===
def get_kpis(symbol: str):
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        # Extract KPIs (adjust as needed)
        kpis = {
            "Open": ticker.history(period="1d")["Open"][0] if not ticker.history(period="1d").empty else "N/A",  # Today's open
            "High": ticker.history(period="1d")["High"][0] if not ticker.history(period="1d").empty else "N/A",  # Today's high
            "Low": ticker.history(period="1d")["Low"][0] if not ticker.history(period="1d").empty else "N/A",  # Today's low
            "Volume": ticker.history(period="1d")["Volume"][0] if not ticker.history(period="1d").empty else "N/A",  # Today's volume
            "52 Week High": info.get("fiftyTwoWeekHigh", "N/A"),
            "52 Week Low": info.get("fiftyTwoWeekLow", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A")
        }

        return kpis
    except Exception as e:
        st.error(f"❌ KPI Error: {str(e)}")
        return {}

# === FUNCTION FOR SIMPLE STOCK PREDICTION (MOVING AVERAGE) ===
def predict_stock_price(symbol: str, window=20):  # window = days for moving average
    try:
        data = yf.download(symbol, period="1y")
        if data.empty:
            return None

        # Calculate moving average
        data['SMA'] = data['Close'].rolling(window=window).mean()
        last_sma = data['SMA'].iloc[-1]

        return last_sma

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        return None

# === STREAMLIT UI ===
st.set_page_config(page_title="StockPot 📊", page_icon="📈", layout="wide")

# --- THEME ---
st.markdown(
    """
    <style>
    body {
        color: #f0f2f6;
        background-color: #1a1a1a;
    }
    .stTextInput>div>div>input {
        color: #f0f2f6;
        background-color: #262730;
        border: none;
        border-radius: 4px;
    }
    .stChatInputContainer {
        background-color: #262730;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        background-color: #333;
        color: #f0f2f6;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 0.5rem;
    }
    .stChatMessage.user {
        background-color: #555;
    }
    .stMetric {
        background-color: #262730;
        color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .stMetric > label {
        color: #a8b1bd;
    }
    .stLineChart > div > div > svg {
        background-color: transparent !important;
    }
    div.block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center; color: #f0f2f6;'>📈 StockPot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a8b1bd;'>Your Real-Time Stock Assistant</p>", unsafe_allow_html=True)

# === SEARCH QUERY BOX (CENTERED) ===
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    query = st.chat_input("Ask about a company...")

# Initialize 'extracted' outside the conditional block
extracted = None

# === SESSION STATE FOR CHAT MESSAGES ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# === PROCESS USER QUERY ===
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Fetching information..."):
        extracted = extract_stock_from_query(query)
        st.markdown(f"<p style='color:#a8b1bd; font-size:0.8em;'>🧠 LLM Output: <code>{extracted}</code></p>", unsafe_allow_html=True)

        if extracted.lower() != "general":
            stock = fetch_stock_info(extracted)
            if stock:
                response = format_stock_info(stock)
                st.session_state.messages.append({"role": "assistant", "content": response})

                st.subheader(f"Stock Performance: {stock['name']} ({stock['symbol']})")
                dynamic_chart = generate_stock_chart(extracted)
                if dynamic_chart is not None:
                    st.altair_chart(dynamic_chart, use_container_width=True)

                prediction = predict_stock_price(extracted)
                if prediction:
                    st.markdown(f"📈 **Simple Prediction (20-day SMA):** <span style='color:#50fa7b;'>${prediction:.2f}</span>", unsafe_allow_html=True)
                else:
                    st.warning("Could not generate prediction for this stock.")

                kpis = get_kpis(extracted)
                if kpis:
                    st.subheader(f"Key Metrics for {extracted.upper()}")
                    kpi_cols = st.columns(len(kpis))
                    i = 0
                    for k, v in kpis.items():
                        with kpi_cols[i]:
                            st.metric(label=k, value=v)
                        i += 1

            else:
                st.session_state.messages.append({"role": "assistant", "content": f"❌ Could not fetch stock data for '{extracted}'."})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "🤖 Please ask about a specific company or stock symbol."})

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# === PRE-BUILT CHARTS ===
st.subheader("Popular Stocks at a Glance")
prebuilt_symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA"]
cols = st.columns(3)
for i, symbol in enumerate(prebuilt_symbols):
    with cols[i % 3]:
        chart = generate_stock_chart(symbol)
        if chart is not None:
            st.caption(symbol)
            st.altair_chart(chart, use_container_width=True)

prebuilt_symbols_bottom = ["JPM", "V", "UNH"]
cols_bottom = st.columns(3)
for i, symbol in enumerate(prebuilt_symbols_bottom):
    with cols_bottom[i % 3]:
        chart = generate_stock_chart(symbol)
        if chart is not None:
            st.caption(symbol)
            st.altair_chart(chart, use_container_width=True)

st.info("Enter a stock symbol in the search bar above for detailed information and predictions.")
