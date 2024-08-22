import requests
import toml
import streamlit as st
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import random  # Import the random module
import openai  # Import the openai module

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

# Set the title of the Streamlit app
st.title("Welcome, I'm Theodora, your trading assistant.")

# Function to fetch stock tickers with details
def fetch_stock_tickers():
    base_url = 'http://api.marketstack.com/v1/'
    endpoint = 'tickers'
    params = {
        'access_key': marketstack_api_key,
        'limit': 1000,  # Adjust this as necessary
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
    endpoint = 'eod'  # End of day data
    all_market_data = []

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        params = {
            'access_key': marketstack_api_key,
            'symbols': ','.join(batch_symbols),
            'limit': 100  # Fetch the last 100 days of data
        }
        api_url = f'{base_url}{endpoint}'

        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
            data = response.json()
            all_market_data.extend(data.get('data', []))
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch data: {e}")

    return all_market_data

# Function to filter stocks by price and volume
def filter_stocks_by_price_and_volume(tickers, min_price, max_price, min_volume):
    filtered_stocks = []
    ticker_symbols = [ticker['symbol'] for ticker in tickers]

    # Fetch market data for multiple tickers in batches
    market_data = fetch_market_data_batch(ticker_symbols)

    if market_data:
        for symbol_data in market_data:
            close_price = symbol_data.get('close')
            volume = symbol_data.get('volume')
            symbol = symbol_data.get('symbol')

            if close_price and volume and min_price <= close_price <= max_price and volume >= min_volume:
                filtered_stocks.append(symbol)

    return filtered_stocks

# Function to calculate technical indicators
def calculate_indicators(df):
    df['RSI'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    
    # Calculate Bollinger Bands and assign each column to the DataFrame
    bb = ta.bbands(df['close'], length=20)
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_middle'] = bb['BBM_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']
    
    return df

# Function to determine trade signals (buy, short, or hold) based on criteria
def get_trade_signals(df):
    buy_signals = []
    short_signals = []
    
    # Use MACD and Bollinger Bands for buy and short signals
    for i in range(1, len(df)):
        if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and df['close'].iloc[i] < df['BB_lower'].iloc[i]:
            buy_signals.append(i)
        elif df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and df['close'].iloc[i] > df['BB_upper'].iloc[i]:
            short_signals.append(i)
    
    return buy_signals, short_signals

# Function to generate a summary of the stock analysis
def generate_summary(df, symbol, buy_signals, short_signals):
    latest_close = df['close'].iloc[-1]
    latest_rsi = df['RSI'].iloc[-1]
    latest_macd = df['MACD'].iloc[-1]
    latest_macd_signal = df['MACD_signal'].iloc[-1]
    bb_upper = df['BB_upper'].iloc[-1]
    bb_lower = df['BB_lower'].iloc[-1]
    volume = df['volume'].iloc[-1]
    
    if buy_signals:
        target_price = df['BB_middle'].iloc[-1]
        summary = f"""
        **{symbol} Stock Analysis (Buy Recommendation):**
        
        - **Current Price**: ${latest_close:.2f}
        - **Volume**: {volume:,} shares
        - **RSI**: {latest_rsi:.2f} (Relative Strength Index)
        - **MACD**: {latest_macd:.2f} (Moving Average Convergence Divergence)
        - **MACD Signal**: {latest_macd_signal:.2f}
        - **Bollinger Bands**: Upper: ${bb_upper:.2f}, Lower: ${bb_lower:.2f}
        
        **Reasoning**:
        {symbol} was chosen based on its technical indicators showing a potential bullish reversal. The MACD has crossed above its signal line, and the price is near the lower Bollinger Band, indicating that the stock might be undervalued.
        
        **Prediction**:
        - **Price Target**: The price may rise to the middle Bollinger Band around ${target_price:.2f} within the next few weeks.
        - **Time Frame**: This movement is expected over the next 1-4 weeks.
        """
    elif short_signals:
        target_price = df['BB_middle'].iloc[-1]
        summary = f"""
        **{symbol} Stock Analysis (Short Recommendation):**
        
        - **Current Price**: ${latest_close:.2f}
        - **Volume**: {volume:,} shares
        - **RSI**: {latest_rsi:.2f} (Relative Strength Index)
        - **MACD**: {latest_macd:.2f} (Moving Average Convergence Divergence)
        - **MACD Signal**: {latest_macd_signal:.2f}
        - **Bollinger Bands**: Upper: ${bb_upper:.2f}, Lower: ${bb_lower:.2f}
        
        **Reasoning**:
        {symbol} was chosen because its indicators suggest a potential bearish movement. The MACD has crossed below its signal line, and the price is near the upper Bollinger Band, indicating the stock might be overvalued.
        
        **Prediction**:
        - **Price Target**: The price may drop to the middle Bollinger Band around ${target_price:.2f} within the next few weeks.
        - **Time Frame**: This movement is expected over the next 1-4 weeks.
        """
    else:
        summary = f"""
        **{symbol} Stock Analysis (Hold Recommendation):**
        
        - **Current Price**: ${latest_close:.2f}
        - **Volume**: {volume:,} shares
        - **RSI**: {latest_rsi:.2f} (Relative Strength Index)
        - **MACD**: {latest_macd:.2f} (Moving Average Convergence Divergence)
        - **MACD Signal**: {latest_macd_signal:.2f}
        - **Bollinger Bands**: Upper: ${bb_upper:.2f}, Lower: ${bb_lower:.2f}
        
        **Reasoning**:
        {symbol} does not currently show strong buy or short signals. The indicators suggest that it might be better to hold and observe further developments before making a trade decision.
        """
    
    return summary

# Function to plot stock data with indicators
def plot_stock(df, symbol, buy_signals, short_signals):
    plt.figure(figsize=(6, 4))
    plt.plot(df['date'], df['close'], label=f'{symbol} Close Price')
    plt.plot(df['date'], df['MACD'], label='MACD')
    plt.plot(df['date'], df['MACD_signal'], label='MACD Signal')
    plt.plot(df['date'], df['RSI'], label='RSI')
    plt.plot(df['date'], df['BB_upper'], label='Bollinger Bands Upper', linestyle='--')
    plt.plot(df['date'], df['BB_middle'], label='Bollinger Bands Middle', linestyle='--')
    plt.plot(df['date'], df['BB_lower'], label='Bollinger Bands Lower', linestyle='--')
    
    # Mark buy signals
    for signal in buy_signals:
        plt.scatter(df['date'].iloc[signal], df['close'].iloc[signal], marker='^', color='g', s=100)

    # Mark short signals
    for signal in short_signals:
        plt.scatter(df['date'].iloc[signal], df['close'].iloc[signal], marker='v', color='r', s=100)

    plt.title(f'{symbol} - Technical Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

# Sidebar: Ask Theodora and Stock Price Range Selection
st.sidebar.markdown("### **Ask Theodora**")
user_input = st.sidebar.text_area("Enter your query:", "What are your thoughts on AAPL's future performance?", key="user_input", height=150)
if st.sidebar.button("Ask Theodora"):
    response = "Due to rate limit issues, Theodora cannot provide real-time answers. Please try again later."
    st.sidebar.write("Theodora's Response:")
    st.sidebar.write(response)

# Sidebar: Stock Price Range Selection
min_price = st.sidebar.number_input('Minimum Stock Price', value=1, min_value=0, max_value=100)
max_price = st.sidebar.number_input('Maximum Stock Price', value=20, min_value=0, max_value=100)

# Sidebar: Stock Ticker Search
st.sidebar.markdown("### **Stock Ticker Search**")
ticker_search = st.sidebar.text_input("Enter Stock Ticker", key="ticker_search")

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

# Fetch and analyze stocks based on the selected price range and volume criteria
all_tickers = fetch_stock_tickers()
potential_stocks = filter_stocks_by_price_and_volume(all_tickers, min_price, max_price, 2000000)

# Ensure we're not sampling more stocks than are available
num_stocks_to_sample = min(len(potential_stocks), 10)

# Select random stocks for analysis based on user-defined criteria
random_stocks = random.sample(potential_stocks, num_stocks_to_sample)

st.subheader("Theodora's Picks of the Week")

cols = st.columns(2)
for i, stock in enumerate(random_stocks):
    market_data = fetch_market_data_batch([stock])
    if market_data:
        df = pd.DataFrame(market_data)
        df['date'] = pd.to_datetime(df['date'])
        df = calculate_indicators(df)
        buy_signals, short_signals = get_trade_signals(df)
        with cols[i % 2]:
            st.write(f"**{stock}** - Analysis")
            plot_stock(df, stock, buy_signals, short_signals)
            
            # Generate and display the summary
            summary = generate_summary(df, stock, buy_signals, short_signals)
            st.markdown(summary)


