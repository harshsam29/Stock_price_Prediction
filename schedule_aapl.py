import requests
import pandas as pd
from datetime import datetime
import schedule
import time

def fetch_realtime_aapl(api_key):
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" not in data:
            raise ValueError("No quote data returned. Check API key or rate limits.")
        
        quote = data["Global Quote"]
        data = {
            'Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Ticker': ['AAPL'],
            'Price': [float(quote["05. price"])],
            'Change': [float(quote["09. change"])],
            'Change_Percent': [quote["10. change percent"]]
        }
        df_realtime = pd.DataFrame(data)
        df_realtime.to_csv(r'D:\Stock_Price_Prediction\aapl_realtime.csv', mode='a', header=not pd.io.common.file_exists(r'D:\Stock_Price_Prediction\aapl_realtime.csv'), index=False)
        print(f"[{datetime.now()}] Real-Time Data Appended to 'aapl_realtime.csv':")
        print(df_realtime)
        return df_realtime
    
    except Exception as e:
        print(f"[{datetime.now()}] Error fetching real-time data: {e}")
        return None

def job():
    print(f"[{datetime.now()}] Fetching real-time data...")
    fetch_realtime_aapl(api_key)
    time.sleep(12)  # Respect Alpha Vantage's 5 calls/minute limit

# Replace with your API key
api_key = "UA8GZNI831U9ZLKJ"

# Schedule every 5 minutes
schedule.every(5).minutes.do(job)

try:
    while True:
        schedule.run_pending()
        time.sleep(60)
except KeyboardInterrupt:
    print(f"[{datetime.now()}] Scheduler stopped by user. Exiting gracefully...")