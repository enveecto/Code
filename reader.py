import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pytz

# Fetching Bitcoin data (adjust period for 1-minute interval)
print("Fetching data for BTC-USD with period='5d' and interval='1m'...")
try:
    data = yf.download('BTC-USD', period='5d', interval='1m')
    if data.empty:
        raise ValueError("No data fetched. Please check the symbol or data constraints.")
    print("Data fetched successfully.")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# Convert timestamps to IST
data.index = data.index.tz_convert('UTC').tz_localize(None)  # Remove the timezone info
data.index = data.index.tz_localize('UTC').tz_convert('Asia/Kolkata')  # Convert to IST

# Preparing the dataset
data['Close'] = data['Close'].fillna(method='ffill')  # Handle missing values
prices = data['Close'].values.reshape(-1, 1)

# Check for empty dataset
if prices.shape[0] == 0:
    print("No valid price data available. Exiting.")
    exit()

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# Creating a dataset for the LSTM
def create_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i-look_back:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

look_back = 60
X, y = create_dataset(scaled_prices, look_back)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshaping for LSTM

# Splitting the dataset
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Building the LSTM model
print("Building LSTM model...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
print("LSTM model built.")

# Training the model
print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)  # Increased epochs for better training
print("Model training complete.")

# Predicting the next 20 minutes
print("Predicting the next 20 minutes...")
last_lookback = scaled_prices[-look_back:]  # Take the last 60 data points
predictions = []

for _ in range(20):
    # Prepare the input for the model
    input_data = last_lookback.reshape(1, look_back, 1)
    prediction = model.predict(input_data, verbose=0)
    predictions.append(prediction[0, 0])  # Append the prediction
    # Update the input for the next prediction
    last_lookback = np.append(last_lookback[1:], prediction, axis=0)

# Rescale predictions back to original scale
predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Logging predictions to Excel
print("Logging predictions to Excel...")
time_now = datetime.now(pytz.timezone('Asia/Kolkata'))  # Get current time in IST
time_intervals = [time_now + timedelta(minutes=i) for i in range(1, 21)]

# Remove the timezone from the time intervals before exporting to Excel
time_intervals_naive = [time.replace(tzinfo=None) for time in time_intervals]

# Create DataFrame with timezone-naive timestamps
df_predictions = pd.DataFrame({
    "Time": time_intervals_naive,
    "Predicted Price (USD)": predicted_prices.flatten()
})

output_file = "bitcoin_predictions_ist.xlsx"
df_predictions.to_excel(output_file, index=False)
print(f"Predictions logged to {output_file}.")

# Plotting the predicted prices
plt.figure(figsize=(10, 6))
plt.plot(time_intervals_naive, predicted_prices, marker='o', color='red', label='Predicted Prices')
plt.title("Bitcoin Price Predictions for Next 20 Minutes (IST)")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Evaluate the model by comparing actual vs predicted prices for the test set
predicted_test_prices = model.predict(X_test)
predicted_test_prices = scaler.inverse_transform(predicted_test_prices)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plotting actual vs predicted prices on test data
plt.figure(figsize=(10, 6))
plt.plot(real_prices, color='blue', label='Actual Prices')
plt.plot(predicted_test_prices, color='red', label='Predicted Prices')
plt.title("Actual vs Predicted Bitcoin Prices (Test Data)")
plt.xlabel("Time")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()

print("Script execution complete.")
