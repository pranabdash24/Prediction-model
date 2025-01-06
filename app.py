import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Stock Price Prediction App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load the trained model
model = load_model("stock_model.h5")

# Apply custom styling
st.markdown(
    """
    <style>
    /* Background styling */
    body {
        background-color: #f0f2f6;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #003366;
        color: white;
    }
    /* Header styling */
    header {
        color: #003366;
    }
    /* Title styling */
    .css-10trblm.e16nr0p33 {
        color: #003366 !important;
    }
    /* Footer styling */
    footer {
        font-weight: bold;
        text-align: right;
        color: #003366;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("<h1 style='text-align: center; color: #003366;'>📈 Stock Price Prediction App</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configure Settings")
    st.write("Adjust the parameters for stock price prediction.")
    ticker = st.text_input("Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")
    start_date = st.date_input("Start Date:", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date:", pd.to_datetime("2023-12-31"))
    lookback = st.slider("Lookback Period (days):", min_value=30, max_value=200, value=60)
    forecast_days = st.slider("Days to Forecast:", min_value=1, max_value=30, value=5)
    start_prediction = st.button("Start Prediction")

# Main Page
if start_prediction:
    # Create a placeholder for messages
    status_placeholder = st.empty()

    try:
        # Display fetching message
        status_placeholder.info("Fetching data... Please wait.")

        # Fetch Stock Data
        data = yf.download(ticker, start=start_date, end=end_date)

        # Clear the fetching data message
        status_placeholder.empty()

        # Check if data is valid
        if data.empty:
            st.error("No stock data available for the given ticker and date range.")
        else:
            # Display stock data
            st.subheader(f"Stock Data for {ticker}")
            st.dataframe(data.tail())

            # Preprocess the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(data[['Close']])

            # Prepare data for prediction
            X, y = [], []
            for i in range(lookback, len(data_scaled)):
                X.append(data_scaled[i - lookback:i, 0])  # Input sequence
                y.append(data_scaled[i, 0])  # Next day's price

            # Convert lists to numpy arrays
            X, y = np.array(X), np.array(y)

            # Reshape X for LSTM (samples, time_steps, features)
            X = X.reshape(X.shape[0], X.shape[1], 1)

            # Split the data into training and testing sets (80% train, 20% test)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

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

            # Prepare forecast dates
            forecast_dates = pd.date_range(data.index[-1], periods=forecast_days + 1, freq='D')[1:]

            # Create a DataFrame for Altair plotting
            historical_data = pd.DataFrame({'Date': data.index, 'Price': data['Close'].values.flatten(), 'Type': 'Historical'})
            forecast_data = pd.DataFrame({'Date': forecast_dates, 'Price': predictions_rescaled.flatten(), 'Type': 'Forecast'})
            combined_data = pd.concat([historical_data, forecast_data])

            # Altair Chart
            chart = alt.Chart(combined_data).mark_line(point=True).encode(
                x='Date:T',
                y='Price:Q',
                color='Type:N',
                tooltip=['Date:T', 'Price:Q', 'Type:N']
            ).properties(
                title=f"{ticker} Stock Price Prediction",
                width=900,
                height=500
            ).interactive()

            # Render the interactive chart
            st.altair_chart(chart, use_container_width=True)

            # Display predicted stock prices
            st.subheader(f"Predicted Stock Prices for the Next {forecast_days} Days:")
            for i, date in enumerate(forecast_dates):
                st.write(f"{date.date()}: ${predictions_rescaled[i][0]:.2f}")

            # Display success message
            st.success("Prediction Successful!")

    except Exception as e:
        # Clear the fetching data message and display the error
        status_placeholder.empty()
        st.error(f"An error occurred: {e}")

# Footer
st.markdown(
    """
    <div style="text-align: right; font-weight: bold; color: #003366;">
        By Pranab Dash
    </div>
    """,
    unsafe_allow_html=True
)
