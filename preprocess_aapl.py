import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    try:
        # Load combined data
        df = pd.read_csv(r'D:\Stock_Price_Prediction\aapl_combined.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Handle missing values
        df = df.dropna(subset=['Close'])
        
        # Add features (7-day and 14-day moving averages)
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA14'] = df['Close'].rolling(window=14).mean()
        df = df.dropna()  # Remove rows with NaN from moving averages
        
        # Prepare data for modeling
        data = df[['Close', 'MA7', 'MA14']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Split into training and test sets (80% train, 20% test)
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        # Save preprocessed data
        np.save(r'D:\Stock_Price_Prediction\aapl_train.npy', train_data)
        np.save(r'D:\Stock_Price_Prediction\aapl_test.npy', test_data)
        df.to_csv(r'D:\Stock_Price_Prediction\aapl_preprocessed.csv', index=False)
        
        print("Preprocessed Data Saved to 'aapl_preprocessed.csv', 'aapl_train.npy', 'aapl_test.npy'")
        print(df.tail())
        return df, train_data, test_data, scaler
    
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None, None, None

if __name__ == "__main__":
    df, train_data, test_data, scaler = preprocess_data()