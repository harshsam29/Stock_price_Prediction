import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler  # Added import
import matplotlib.pyplot as plt

def train_predict_model():
    try:
        # Load preprocessed data
        train_data = np.load(r'D:\Stock_Price_Prediction\aapl_train.npy')
        test_data = np.load(r'D:\Stock_Price_Prediction\aapl_test.npy')
        df = pd.read_csv(r'D:\Stock_Price_Prediction\aapl_preprocessed.csv')
        
        # Split features (MA7, MA14) and target (Close)
        X_train = train_data[:, 1:]  # MA7, MA14
        y_train = train_data[:, 0]   # Close
        X_test = test_data[:, 1:]
        y_test = test_data[:, 0]
        
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate error
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")
        
        # Inverse scale predictions for plotting
        test_data_full = np.concatenate((y_test.reshape(-1, 1), X_test), axis=1)
        pred_data_full = np.concatenate((y_pred.reshape(-1, 1), X_test), axis=1)
        scaler = MinMaxScaler()
        scaler.fit(df[['Close', 'MA7', 'MA14']])
        test_data_unscaled = scaler.inverse_transform(test_data_full)
        pred_data_unscaled = scaler.inverse_transform(pred_data_full)
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'].iloc[-len(y_test):], test_data_unscaled[:, 0], label='Actual Close', color='blue')
        plt.plot(df['Date'].iloc[-len(y_test):], pred_data_unscaled[:, 0], label='Predicted Close', color='red')
        plt.title('AAPL Stock Price Prediction (Linear Regression)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.savefig(r'D:\Stock_Price_Prediction\aapl_prediction_plot.png')
        plt.show()
        
        return model, mse
    
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None

if __name__ == "__main__":
    model, mse = train_predict_model()