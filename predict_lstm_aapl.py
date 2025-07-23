import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def train_lstm_model():
    try:
        df = pd.read_csv(r'D:\Stock_Price_Prediction\aapl_preprocessed.csv')
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        sequence_length = 60
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        y_pred = model.predict(X_test)
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_unscaled = scaler.inverse_transform(y_pred)
        mse = np.mean((y_test_unscaled - y_pred_unscaled) ** 2)
        print(f"LSTM Mean Squared Error: {mse}")
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'].iloc[-len(y_test):], y_test_unscaled, label='Actual Close', color='blue')
        plt.plot(df['Date'].iloc[-len(y_test):], y_pred_unscaled, label='Predicted Close', color='red')
        plt.title('AAPL Stock Price Prediction (LSTM)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(r'D:\Stock_Price_Prediction\aapl_lstm_plot.png')
        plt.show()
        return model, mse
    except Exception as e:
        print(f"Error training LSTM model: {e}")
        return None, None

if __name__ == "__main__":
    model, mse = train_lstm_model()