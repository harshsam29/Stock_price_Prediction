# AAPL Stock Price Prediction Project

This repository contains a Python project that fetches historical and real-time stock price data for Apple Inc. (AAPL) using the Alpha Vantage API, preprocesses the data with moving averages, and predicts future closing prices using Linear Regression and an optional LSTM model. The project is designed for educational purposes and can be extended for personal stock analysis.

## Features
- Fetches historical daily data (since 2020) and real-time quotes for AAPL.
- Combines data into a unified dataset with daily closing prices.
- Calculates 7-day and 14-day moving averages as features.
- Trains a Linear Regression model to predict closing prices.
- Includes an optional LSTM model for advanced time-series predictions.
- Visualizes predictions and historical data using Matplotlib.
- Supports real-time data updates with a 5-minute scheduler.

## Requirements
- Python 3.10
- Required libraries:
  - `requests`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `tensorflow` (for LSTM)
  - `streamlit` (for optional visualization)

## Installation

### Prerequisites
- Install Python 3.10 from [python.org](https://www.python.org/downloads/release/python-3109/).
- Ensure `pip` is installed and updated.

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/aapl-stock-prediction.git
   cd aapl-stock-prediction
