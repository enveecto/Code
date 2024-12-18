import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Fetching Bitcoin data
print("Fetching BTC-USD data for 1 month with hourly interval...")
data = yf.download('BTC-USD', period='1mo', interval='1h')

if data.empty:
    raise ValueError("No data fetched. Please check the symbol or data constraints.")
data['Close'] = data['Close'].ffill()

# Technical Indicators
print("Calculating technical indicators...")

# Simple Moving Average (SMA)
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_20'] = data['Close'].rolling(window=20).mean()

# Exponential Moving Average (EMA)
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

# Bollinger Bands
data['Bollinger_Upper'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
data['Bollinger_Lower'] = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()

# Relative Strength Index (RSI)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0).flatten()  # Convert to 1D
    loss = np.where(delta < 0, -delta, 0).flatten()  # Convert to 1D
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = calculate_rsi(data['Close'])

# Preparing the dataset for LSTM
print("Preparing data for LSTM model...")
prices = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Creating a dataset for the LSTM
def create_dataset(dataset, look_back=3):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

look_back = 24  # Using 24 hours of data for prediction (1 day)
X, y = create_dataset(scaled_prices, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Splitting the dataset
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Building and training the LSTM model
print("Building LSTM model...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training LSTM model...")
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)
print("Model training complete.")

# Predicting the next 3 days (72 hours)
print("Predicting the next 3 days...")
last_lookback = scaled_prices[-look_back:]
predictions = []

for _ in range(72):  # Predicting hourly for the next 3 days (72 hours)
    input_data = last_lookback.reshape(1, look_back, 1)
    prediction = model.predict(input_data, verbose=0)
    predictions.append(prediction[0, 0])
    last_lookback = np.append(last_lookback[1:], prediction, axis=0)

predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
dates = [data.index[-1] + timedelta(hours=i) for i in range(1, 73)]
df_predictions = pd.DataFrame({
    "Date": pd.to_datetime(dates).dt.tz_localize(None),  # Making sure the dates are timezone-naive
    "Predicted Price (USD)": predicted_prices.flatten()
})

output_file = "bitcoin_predictions_with_analysis.xlsx"
df_predictions.to_excel(output_file, index=False)
print(f"Predictions saved to {output_file}.")

# Plotting the results
# Create subplots for three graphs
fig, axs = plt.subplots(3, 1, figsize=(14, 18), sharex=True)

# Graph 1: Prediction using LSTM
axs[0].plot(data.index, prices, label='Actual Prices (Past)', color='blue', marker='.')
all_dates = list(data.index) + dates
all_prices = np.append(prices, predicted_prices)
axs[0].plot(all_dates, all_prices, label='Predicted Prices (Future)', color='red', linestyle='--', marker='o')
axs[0].set_title("Graph 1: LSTM-based Prediction")
axs[0].set_ylabel("Price (USD)")
axs[0].legend(loc='upper left')
axs[0].grid()

# Graph 2: Prediction using technical indicators
axs[1].plot(data.index, prices, label='Actual Prices (Past)', color='blue', marker='.')
axs[1].plot(data.index, data['SMA_10'], label='SMA (10)', color='orange', linestyle='--')
axs[1].plot(data.index, data['EMA_10'], label='EMA (10)', color='green', linestyle='--')
axs[1].fill_between(data.index, data['Bollinger_Upper'], data['Bollinger_Lower'], color='gray', alpha=0.2, label='Bollinger Bands')
axs[1].set_title("Graph 2: Technical Analysis-based Prediction")
axs[1].set_ylabel("Price (USD)")
axs[1].legend(loc='upper left')
axs[1].grid()

# Graph 3: Combination of LSTM predictions and technical indicators
axs[2].plot(data.index, prices, label='Actual Prices (Past)', color='blue', marker='.')
axs[2].plot(all_dates, all_prices, label='Predicted Prices (Future)', color='red', linestyle='--', marker='o')
axs[2].plot(data.index, data['SMA_10'], label='SMA (10)', color='orange', linestyle='--')
axs[2].plot(data.index, data['EMA_10'], label='EMA (10)', color='green', linestyle='--')
axs[2].fill_between(data.index, data['Bollinger_Upper'], data['Bollinger_Lower'], color='gray', alpha=0.2, label='Bollinger Bands')
axs[2].set_title("Graph 3: Combined LSTM and Technical Analysis")
axs[2].set_xlabel("Date")
axs[2].set_ylabel("Price (USD)")
axs[2].legend(loc='upper left')
axs[2].grid()

plt.tight_layout()
plt.show()

print("Analysis and visualization complete.")
