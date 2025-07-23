import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def fetch_historical_aapl(api_key):
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={api_key}&outputsize=full"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Time Series (Daily)" not in data:
            raise ValueError("No data returned. Check API key or rate limits.")
        
        df_historical = pd.DataFrame(data["Time Series (Daily)"]).T
        df_historical = df_historical[['1. open', '2. high', '3. low', '4. close', '5. volume']]
        df_historical.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_historical.index.name = 'Date'
        df_historical = df_historical[pd.to_datetime(df_historical.index) >= "2020-01-01"]
        df_historical = df_historical.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': int})
        df_historical.to_csv('aapl_historical.csv')
        print("Historical Data Saved to 'aapl_historical.csv':")
        print(df_historical.tail())
        return df_historical
    
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None

def fetch_realtime_aapl(api_key):
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" not in data:
            raise ValueError("No quote data returned.")
        
        quote = data["Global Quote"]
        data = {
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Ticker': ['AAPL'],
            'Price': [float(quote["05. price"])],
            'Change': [float(quote["09. change"])],
            'Change_Percent': [quote["10. change percent"]]
        }
        df_realtime = pd.DataFrame(data)
        df_realtime.to_csv('aapl_realtime.csv', mode='a', header=not pd.io.common.file_exists('aapl_realtime.csv'), index=False)
        print("Real-Time Data Appended to 'aapl_realtime.csv':")
        print(df_realtime)
        return df_realtime
    
    except Exception as e:
        print(f"Error fetching real-time data: {e}")
        return None

def combine_data():
    try:
        df_historical = pd.read_csv(r'D:\Stock_Price_Prediction\aapl_historical.csv')
        df_historical['Date'] = pd.to_datetime(df_historical['Date']).dt.strftime('%Y-%m-%d')
        df_historical = df_historical[['Date', 'Close']]
        df_realtime = pd.read_csv(r'D:\Stock_Price_Prediction\aapl_realtime.csv')
        df_realtime['Date'] = pd.to_datetime(df_realtime['Date']).dt.strftime('%Y-%m-%d')
        df_realtime = df_realtime[['Date', 'Price']].rename(columns={'Price': 'Close'})
        df_combined = pd.concat([df_historical, df_realtime]).drop_duplicates(subset='Date', keep='last')
        df_combined.sort_values('Date', inplace=True)
        df_combined.to_csv(r'D:\Stock_Price_Prediction\aapl_combined.csv', index=False)
        print("Combined Data Saved to 'aapl_combined.csv':")
        print(df_combined.tail())
        return df_combined
    except Exception as e:
        print(f"Error combining data: {e}")
        return None
    
def plot_data():
    try:
        df = pd.read_csv('aapl_combined.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['Close'], label='AAPL Closing Price', color='blue')
        plt.title('Apple (AAPL) Stock Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig('aapl_price_plot.png')
        plt.show()
        
    except Exception as e:
        print(f"Error plotting data: {e}")

# Replace with your API key
api_key = "UA8GZNI831U9ZLKJ"

# Run all steps
if __name__ == "__main__":
    fetch_historical_aapl(api_key)
    fetch_realtime_aapl(api_key)
    combine_data()
    plot_data()