import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import math

# Load the trained model
model = load_model("stock_model.h5")

# Streamlit App
st.title("Stock Price Prediction App")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA):", "AAPL")
start_date = st.date_input("Start Date:", pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date:", pd.to_datetime("2023-12-31"))
lookback = st.slider("Lookback Period (days):", min_value=30, max_value=200, value=60)
forecast_days = st.slider("Days to Forecast:", min_value=1, max_value=30, value=5)

# Fetch Stock Data
data = yf.download(ticker, start=start_date, end=end_date)

# Display stock data
st.subheader(f"Stock Data for {ticker}")
st.write(data.tail())

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data[['Close']])

# Prepare data for prediction
X, y = [], []
for i in range(lookback, len(data_scaled)):
    X.append(data_scaled[i-lookback:i, 0])  # Input sequence (lookback days)
    y.append(data_scaled[i, 0])  # Next day's price

# Convert lists to numpy arrays
X, y = np.array(X), np.array(y)

# Reshape X for LSTM (samples, time_steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Model evaluation on test data
y_pred = model.predict(X_test)

# Calculate accuracy score (RMSE)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"Root Mean Squared Error (RMSE) on Test Data: {rmse:.2f}")

# Calculate R² score (accuracy in percentage)
r2 = r2_score(y_test, y_pred)
accuracy_percentage = r2 * 100
st.write(f"Model Accuracy (R² Score): {accuracy_percentage:.2f}%")

# Confidence interval (95% confidence level based on standard deviation of the predictions)
confidence_interval = 1.96 * np.std(y_pred)  # Assuming normal distribution

st.write(f"Confidence Interval (95% Confidence Level): ±{confidence_interval:.2f} USD")

# Make predictions for the forecast days
input_data = data_scaled[-lookback:].reshape(1, lookback, 1)
predictions = []

for _ in range(forecast_days):
    predicted_price = model.predict(input_data)
    predictions.append(predicted_price[0, 0])
    
    # Update the input data for the next day prediction
    input_data = np.append(input_data[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

# Rescale predictions to original scale
predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Prepare for plotting
forecast_dates = pd.date_range(data.index[-1], periods=forecast_days+1, freq='D')[1:]

# Plot the data using Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Plot historical data
ax.plot(data.index, data['Close'], label='Historical Prices')

# Plot predicted data
ax.plot(forecast_dates, predictions_rescaled.flatten(), label='Predicted Prices', linestyle='--', color='orange')

# Adding labels and title
ax.set_title(f"{ticker} Stock Price Prediction", fontsize=16)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Stock Price (USD)', fontsize=14)
ax.legend()

# Display the plot
st.pyplot(fig)

# Display predicted stock prices
st.subheader(f"Predicted Stock Prices for the Next {forecast_days} Days:")
for i, date in enumerate(forecast_dates):
    st.write(f"{date.date()}: ${predictions_rescaled[i][0]:.2f}")

# Display "By Pranab Dash" at the bottom right corner
st.markdown(
    """
    <div style="text-align: right; font-weight: bold;">
        By Pranab Dash
    </div>
    """, 
    unsafe_allow_html=True
)
