import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Function to load CPI data from a CSV file
def load_cpi_data():
    # Load the data from the CSV file
    cpi_data = pd.read_csv("data.csv")
    
    # Ensure the 'Date' column is in datetime format
    cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
    
    # Set the 'Date' column as the index
    cpi_data.set_index('Date', inplace=True)
    
    return cpi_data

# Function to fit ARIMA model and forecast
def forecast_cpi(cpi_data, forecast_period):
    # Fit an ARIMA model (manually set order)
    model = ARIMA(cpi_data, order=(1, 1, 1))  # You can adjust (p, d, q) as needed
    model_fit = model.fit()
    
    # Forecast the next `forecast_period` months
    forecast = model_fit.forecast(steps=forecast_period)
    forecast_index = pd.date_range(start=cpi_data.index[-1] + timedelta(days=1), periods=forecast_period, freq='M')
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    # Calculate percent change from previous month
    last_cpi_value = cpi_data.iloc[-1]
    forecast_pct_change = forecast_series.pct_change() * 100
    forecast_pct_change.iloc[0] = ((forecast_series.iloc[0] - last_cpi_value) / last_cpi_value) * 100
    
    # Combine forecast and percent change into a DataFrame
    forecast_df = pd.DataFrame({
        'Forecasted CPI': forecast_series,
        'Percent Change': forecast_pct_change
    })
    
    return forecast_df

# Streamlit app
def main():
    st.title("CPI Time Series Forecast App")
    
    # Load CPI data
    cpi_data = load_cpi_data()
    
    # Display the raw data
    st.subheader("Raw CPI Data")
    st.write(cpi_data)
    
    # Plot the raw data
    st.subheader("CPI Time Series Plot")
    plt.figure(figsize=(10, 6))
    plt.plot(cpi_data.index, cpi_data['CPI'], label='CPI')
    plt.xlabel('Date')
    plt.ylabel('CPI')
    plt.title('CPI Time Series')
    plt.legend()
    st.pyplot(plt)
    
    # Forecast next 3 months
    st.subheader("Forecast Next 3 Months")
    forecast_3_months = forecast_cpi(cpi_data['CPI'], 3)
    st.write(forecast_3_months)
    
    # Plot the forecast for the next 3 months
    plt.figure(figsize=(10, 6))
    plt.plot(cpi_data.index, cpi_data['CPI'], label='Historical CPI')
    plt.plot(forecast_3_months.index, forecast_3_months['Forecasted CPI'], label='Forecasted CPI (3 Months)', color='red')
    plt.xlabel('Date')
    plt.ylabel('CPI')
    plt.title('CPI Forecast - Next 3 Months')
    plt.legend()
    st.pyplot(plt)
    
    # Forecast next 12 months
    st.subheader("Forecast Next 12 Months")
    forecast_12_months = forecast_cpi(cpi_data['CPI'], 12)
    st.write(forecast_12_months)
    
    # Plot the forecast for the next 12 months
    plt.figure(figsize=(10, 6))
    plt.plot(cpi_data.index, cpi_data['CPI'], label='Historical CPI')
    plt.plot(forecast_12_months.index, forecast_12_months['Forecasted CPI'], label='Forecasted CPI (12 Months)', color='red')
    plt.xlabel('Date')
    plt.ylabel('CPI')
    plt.title('CPI Forecast - Next 12 Months')
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    main()
