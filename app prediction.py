
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import plotly.graph_objects as go

# Streamlit App
st.title("Stock Price Prediction App")

# Sidebar inputs
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT):", "AAPL")
start_date = st.sidebar.date_input("Start Date:", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date:", pd.to_datetime("2023-12-31"))
lookback = st.sidebar.slider("Lookback Period (days):", 10, 100, 60)
forecast_days = st.sidebar.slider("Forecast Days:", 1, 30, 5)

if st.sidebar.button("Fetch and Predict"):
    try:
        # Fetch stock data
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the specified stock ticker and date range.")
        else:
            st.write("Preview of Stock Data:")
            st.dataframe(data.head())

            # Plot historical prices
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
            fig.update_layout(title=f"Historical Stock Prices for {ticker}", xaxis_title="Date", yaxis_title="Close Price")
            st.plotly_chart(fig)

            # Preprocess data
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data[['Close']])

            # Prepare data for LSTM
            X, y = [], []
            for i in range(lookback, len(data_scaled)):
                X.append(data_scaled[i-lookback:i, 0])
                y.append(data_scaled[i, 0])
            X = np.array(X).reshape(len(X), lookback, 1)

            # Load model and predict
            model = load_model("stock_model.h5")
            predictions = []
            input_seq = data_scaled[-lookback:].reshape(1, lookback, 1)
            for _ in range(forecast_days):
                pred = model.predict(input_seq)[0][0]
                predictions.append(pred)
                input_seq = np.append(input_seq[:, 1:, :], [[pred]], axis=1)

            # Reverse scaling for predictions
            predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
            pred_df = pd.DataFrame(predictions, columns=["Predicted Price"])
            pred_df.index = pd.date_range(start=end_date, periods=forecast_days + 1, freq="B")[1:]
            
            # Plot predictions
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Prices'))
            fig_pred.add_trace(go.Scatter(x=pred_df.index, y=pred_df['Predicted Price'], mode='lines+markers', name='Predicted Prices'))
            fig_pred.update_layout(title="Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_pred)
            
            # Display predictions
            st.write("Predicted Prices:")
            st.dataframe(pred_df)

    except Exception as e:
        st.error(f"Error: {e}")
