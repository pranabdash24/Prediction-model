import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import altair as alt

# Set page configuration for better mobile responsiveness
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📉",
    layout="centered",  # Centered layout for better mobile experience
    initial_sidebar_state="collapsed",  # Start with sidebar collapsed
)

# Load the trained model
model = load_model("stock_model.h5")

# Custom mobile-friendly styling
st.markdown(
    """
    <style>
    /* Global styles */
    body {
        background-color: #121212;
        color: #DCDCDC;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
        font-size: 16px;
        font-weight: 500;
        padding-top: 10px;
    }

    /* Header */
    .css-10trblm.e16nr0p33 {
        font-size: 30px;
        font-weight: bold;
        color: #A9A9A9;
        text-align: center;
    }

    /* Title */
    h1 {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        color: #FF4500;
        padding-top: 20px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-size: 14px;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 30px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #218838;
    }

    /* Input fields */
    .stTextInput>div>input {
        background-color: #2F2F2F;
        color: white;
        border-radius: 10px;
        border: 1px solid #4CAF50;
        padding: 10px 12px;
        font-size: 14px;
    }

    .stSlider>div>input {
        background-color: #2F2F2F;
        color: white;
        border-radius: 10px;
        font-size: 14px;
    }

    /* Interactive chart */
    .stAltair .vega-embed {
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3);
    }

    /* Footer */
    footer {
        font-size: 12px;
        font-weight: bold;
        text-align: right;
        color: #808080;
        padding-right: 20px;
        padding-bottom: 20px;
    }

    @media (max-width: 768px) {
        h1 {
            font-size: 22px;
            padding-top: 10px;
        }

        .stButton>button {
            padding: 8px 18px;
            font-size: 12px;
        }

        .stTextInput>div>input {
            font-size: 12px;
            padding: 8px 10px;
        }

        .stSlider>div>input {
            font-size: 12px;
        }

        .stAltair .vega-embed {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Disclaimer and Terms Pop-up
disclaimer_accepted = st.session_state.get("disclaimer_accepted", False)

if not disclaimer_accepted:
    disclaimer = """
    **Disclaimer:**

    This is a stock price prediction app created for educational purposes. 
    The model is in beta and may not provide accurate predictions. 
    Neither the app nor the developer are legally responsible for any financial losses incurred 
    based on the predictions or information provided by this app. Use at your own risk.

    By clicking "Accept", you acknowledge that you have read and understood the disclaimer.
    """

    st.warning(disclaimer, icon="⚠️")
    if st.button("Accept"):
        st.session_state.disclaimer_accepted = True
        st.experimental_rerun()  # Rerun the app to remove the disclaimer

# Title of the web app
if disclaimer_accepted:
    st.markdown("<h1>📉 Stock Price Prediction</h1>", unsafe_allow_html=True)

    # Main content
    st.header("Configure Stock Prediction Model")
    st.write("Please configure the settings below to start prediction.")

    # Dropdowns and sliders inside the main area (outside of sidebar)
    ticker = st.text_input("Stock Ticker (e.g., AAPL, TSLA):", value="AAPL")
    start_date = st.date_input("Start Date:", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date:", pd.to_datetime("2023-12-31"))

    # Slider for Lookback Period (in days)
    lookback = st.slider("Lookback Period (days):", min_value=30, max_value=200, value=60, step=1)

    # Slider for Forecast Days (in days)
    forecast_days = st.slider("Days to Forecast:", min_value=1, max_value=30, value=5, step=1)

    # Start Prediction button
    start_prediction = st.button("Start Prediction", key="start_prediction")

    # Main page logic
    if start_prediction:
        # Create a placeholder for status message
        status_placeholder = st.empty()

        try:
            # Show "Fetching data" message
            status_placeholder.info("Fetching data... Please wait.")
            
            # Fetch stock data
            data = yf.download(ticker, start=start_date, end=end_date)

            # Clear the "Fetching data" message
            status_placeholder.empty()

            # If data is empty
            if data.empty:
                st.error("No stock data available for the provided ticker and date range.")
            else:
                # Display stock data
                st.subheader(f"Stock Data for {ticker}")
                st.dataframe(data.tail())

                # Preprocess the data for prediction
                scaler = MinMaxScaler(feature_range=(0, 1))
                data_scaled = scaler.fit_transform(data[['Close']])

                # Prepare data for LSTM
                X, y = [], []
                for i in range(lookback, len(data_scaled)):
                    X.append(data_scaled[i - lookback:i, 0])  # Input sequence
                    y.append(data_scaled[i, 0])  # Target value (Next day's price)

                # Convert lists to numpy arrays
                X, y = np.array(X), np.array(y)

                # Reshape X to match LSTM input format (samples, time_steps, features)
                X = X.reshape(X.shape[0], X.shape[1], 1)

                # Split into training and testing sets (80% train, 20% test)
                train_size = int(len(X) * 0.8)
                X_train, X_test = X[:train_size], X[train_size:]
                y_train, y_test = y[:train_size], y[train_size:]

                # Make predictions for forecast days
                input_data = data_scaled[-lookback:].reshape(1, lookback, 1)
                predictions = []

                for _ in range(forecast_days):
                    predicted_price = model.predict(input_data)
                    predictions.append(predicted_price[0, 0])

                    # Update input_data for next day's prediction
                    input_data = np.append(input_data[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

                # Rescale predictions back to original scale
                predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

                # Prepare forecast dates
                forecast_dates = pd.date_range(data.index[-1], periods=forecast_days + 1, freq='D')[1:]

                # Combine historical data with forecast data for visualization
                historical_data = pd.DataFrame({'Date': data.index, 'Price': data['Close'].values.flatten(), 'Type': 'Historical'})
                forecast_data = pd.DataFrame({'Date': forecast_dates, 'Price': predictions_rescaled.flatten(), 'Type': 'Forecast'})
                combined_data = pd.concat([historical_data, forecast_data])

                # Altair chart for visualization
                chart = alt.Chart(combined_data).mark_line(point=True).encode(
                    x='Date:T',
                    y='Price:Q',
                    color='Type:N',
                    tooltip=['Date:T', 'Price:Q', 'Type:N']
                ).properties(
                    title=f"{ticker} Stock Price Prediction",
                    width=800,
                    height=400
                ).interactive()

                # Display interactive chart
                st.altair_chart(chart, use_container_width=True)

                # Display forecasted stock prices
                st.subheader(f"Predicted Stock Prices for the Next {forecast_days} Days:")
                for i, date in enumerate(forecast_dates):
                    st.write(f"{date.date()}: ${predictions_rescaled[i][0]:.2f}")

                # Display success message
                st.success("Prediction Successful!")

        except Exception as e:
            # Clear "Fetching data" message and display error
            status_placeholder.empty()
            st.error(f"An error occurred: {e}")

    # Footer
    st.markdown(
        """
        <footer>
            <div style="text-align: right; font-weight: bold; color: #A9A9A9;">
                By Pranab Dash
            </div>
        </footer>
        """,
        unsafe_allow_html=True
    )
