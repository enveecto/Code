import requests
import time

# Define CoinGecko API URL for Bitcoin market data
coingecko_url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
binance_orderbook_url = "https://api.binance.com/api/v3/depth"

# Function to get Bitcoin market data from CoinGecko
def get_coingecko_data():
    print("Fetching data from CoinGecko...")
    params = {
        "ids": "bitcoin",
        "vs_currency": "usd",
        "days": "1"  # Fetch 1 day data to get the latest market chart
    }
    
    try:
        response = requests.get(coingecko_url, params=params)
        data = response.json()

        # Check if the response contains the expected keys
        if 'prices' in data and 'total_volumes' in data:
            current_price = data['prices'][-1][1]  # Latest price
            market_volume = data['total_volumes'][-1][1]  # Trading volume in USD
            print(f"Current Bitcoin Price: ${current_price:.2f}")
            print(f"Total Trading Volume (USD): ${market_volume:.2f}")
        else:
            print("Error: Unexpected data structure from CoinGecko API.")
            print("Data received:", data)
    except Exception as e:
        print(f"Error fetching data from CoinGecko: {e}")

# Function to get live trade data from Binance API (Order Book)
def get_binance_data():
    print("Fetching data from Binance...")
    try:
        params = {
            'symbol': 'BTCUSDT',
            'limit': 5  # Limit to top 5 bids and asks for simplicity
        }
        response = requests.get(binance_orderbook_url, params=params)
        data = response.json()

        if 'bids' in data and 'asks' in data:
            # Sum up the buy (bid) volume
            buy_volume = sum(float(bid[1]) for bid in data['bids'])  # Total buy volume (BTC)
            sell_volume = sum(float(ask[1]) for ask in data['asks'])  # Total sell volume (BTC)

            print(f"BTC/USDT 24h Buy Volume: {buy_volume} BTC")
            print(f"BTC/USDT 24h Sell Volume: {sell_volume} BTC")
        else:
            print("Error: Unexpected data structure from Binance API.")
            print("Data received:", data)

    except Exception as e:
        print(f"Error fetching data from Binance: {e}")

# Run the live update every 60 seconds
while True:
    get_coingecko_data()   # Get data from CoinGecko
    get_binance_data()     # Get trade volume from Binance

    print("\nWaiting for the next update...\n")
    time.sleep(60)  # Wait for 60 seconds before fetching again
