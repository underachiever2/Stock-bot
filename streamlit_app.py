import requests
import toml
import streamlit as st
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import random
import openai
from PIL import Image

# Load the secrets from the .streamlit/secrets.toml file
try:
    secrets = toml.load('.streamlit/secrets.toml')
    marketstack_api_key = secrets['marketstack']['api_key']
    openai_api_key = secrets['openai']['api_key']
except FileNotFoundError:
    st.error("The secrets.toml file was not found.")
    st.stop()
except KeyError as e:
    st.error(f"The API key was not found in the secrets.toml file: {e}")
    st.stop()

# Set up the OpenAI client using the openai module
openai.api_key = openai_api_key

# Load Theodora's avatar and display it in the sidebar
avatar = Image.open("Theodora-avatar.webp")
st.sidebar.image(avatar, use_column_width=True)

# Set the title of the Streamlit app
st.title("Welcome, I'm Theodora, your trading assistant.")

# Function to fetch stock tickers with details
def fetch_stock_tickers():
    base_url = 'http://api.marketstack.com/v1/'
    endpoint = 'tickers'
    params = {
        'access_key': marketstack_api_key,
        'limit': 1000,
    }
    api_url = f'{base_url}{endpoint}'

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data['data']
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch tickers: {e}")
        return []

# Function to fetch market data from MarketStack API for multiple symbols in batches
def fetch_market_data_batch(symbols, batch_size=50):
    base_url = 'http://api.marketstack.com/v1/'
    endpoint = 'eod'
    all_market_data = []

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        params = {
            'access_key': marketstack_api_key,
            'symbols': ','.join(batch_symbols),
            'limit': 100
        }
        api_url = f'{base_url}{endpoint}'

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            all_market_data.extend(data.get('data', []))
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data: {e}")

    return all_market_data

# Function to calculate technical indicators
def calculate_indicators(df):
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    
    bb = ta.bbands(df['close'], length=20)
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']
    
    return df

# Function to plot stock data with indicators
def plot_stock(df, symbol, buy_signals, short_signals):
    plt.figure(figsize=(8, 6))
    plt.plot(df['date'], df['close'], label=f'{symbol} Close Price')
    plt.plot(df['date'], df['MACD'], label='MACD')
    plt.plot(df['date'], df['MACD_signal'], label='MACD Signal')
    plt.plot(df['date'], df['RSI'], label='RSI')
    plt.plot(df['date'], df['BB_upper'], label='Bollinger Bands Upper', linestyle='--')
    plt.plot(df['date'], df['BB_middle'], label='Bollinger Bands Middle', linestyle='--')
    plt.plot(df['date'], df['BB_lower'], label='Bollinger Bands Lower', linestyle='--')
    
    for signal in buy_signals:
        plt.scatter(df['date'].iloc[signal], df['close'].iloc[signal], marker='^', color='g', s=100)

    for signal in short_signals:
        plt.scatter(df['date'].iloc[signal], df['close'].iloc[signal], marker='v', color='r', s=100)

    plt.title(f'{symbol} - Technical Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Function to get trade signals (buy, short, or hold)
def get_trade_signals(df):
    buy_signals = []
    short_signals = []
    
    for i in range(1, len(df)):
        if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and df['close'].iloc[i] < df['BB_lower'].iloc[i]:
            buy_signals.append(i)
        elif df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and df['close'].iloc[i] > df['BB_upper'].iloc[i]:
            short_signals.append(i)
    
    return buy_signals, short_signals

# Fetch and display stock indices
indices = ['DIA', 'QQQ', 'SPY']
for index_name in indices:
    market_data = fetch_market_data_batch([index_name])
    if market_data:
        df = pd.DataFrame(market_data)
        df['date'] = pd.to_datetime(df['date'])
        df = calculate_indicators(df)
        st.subheader(f"{index_name} - Analysis")
        plot_stock(df, index_name, [], [])

# Sidebar: Ask Theodora and Stock Price Range Selection
st.sidebar.markdown("### **Ask Theodora**")
user_input = st.sidebar.text_area("Enter your query:", "What are your thoughts on AAPL's future performance?", key="user_input_1", height=150)
if st.sidebar.button("Ask Theodora", key="ask_button"):
    response = "Due to rate limit issues, Theodora cannot provide real-time answers. Please try again later."
    st.sidebar.write("Theodora's Response:")
    st.sidebar.write(response)

# Sidebar: Stock Price Range Selection
min_price = st.sidebar.number_input('Minimum Stock Price', value=1, min_value=0, max_value=100, key="min_price")
max_price = st.sidebar.number_input('Maximum Stock Price', value=20, min_value=0, max_value=100, key="max_price")

# Sidebar: Stock Ticker Search
st.sidebar.markdown("### **Stock Ticker Search**")
ticker_search = st.sidebar.text_input("Enter Stock Ticker", key="ticker_search_1")

if ticker_search:
    st.sidebar.write(f"Searching for {ticker_search.upper()}...")
    market_data = fetch_market_data_batch([ticker_search.upper()])
    if market_data:
        df = pd.DataFrame(market_data)
        df['date'] = pd.to_datetime(df['date'])
        df = calculate_indicators(df)
        buy_signals, short_signals = get_trade_signals(df)
        st.write(f"**{ticker_search.upper()}** - Analysis")
        plot_stock(df, ticker_search.upper(), buy_signals, short_signals)
        summary = generate_summary(df, ticker_search.upper(), buy_signals, short_signals)
        st.markdown(summary)

