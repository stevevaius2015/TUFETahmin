import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from datetime import datetime, timedelta

# Function to load CPI data from a CSV file
def load_cpi_data():
    cpi_data = pd.read_csv("data.csv")
    cpi_data['Date'] = pd.to_datetime(cpi_data['Date'])
    cpi_data.set_index('Date', inplace=True)
    return cpi_data

# Function to prepare data for NeuralForecast (N-HiTS)
def prepare_nf_data(cpi_data):
    nf_df = cpi_data.reset_index()
    nf_df = nf_df.rename(columns={'Date': 'ds', 'CPI': 'y'})
    nf_df['unique_id'] = 'cpi_series'
    return nf_df[['unique_id', 'ds', 'y']]

# Function to fit ARIMA model and forecast
def forecast_cpi_arima(cpi_data, forecast_period):
    model = ARIMA(cpi_data, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_period)
    forecast_index = pd.date_range(start=cpi_data.index[-1] + timedelta(days=1), periods=forecast_period, freq='M')
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    last_cpi_value = cpi_data.iloc[-1]
    forecast_pct_change = forecast_series.pct_change() * 100
    forecast_pct_change.iloc[0] = ((forecast_series.iloc[0] - last_cpi_value) / last_cpi_value) * 100
    
    return pd.DataFrame({
        'Forecasted CPI': forecast_series,
        'Percent Change': forecast_pct_change
    })

# Function to fit N-HiTS model and forecast
def forecast_cpi_nhits(nf_df, forecast_period):
    # Initialize N-HiTS model with explicit stacking
    models = [NHITS(
        h=forecast_period,          # Forecast horizon
        input_size=24,              # Historical window (2 years of monthly data)
        max_steps=500,              # Training iterations
        stack_types=['identity'],   # Single stack for simplicity
        n_blocks=[1],               # One block per stack
        scaler_type='robust'
    )]
    
    # Create and fit NeuralForecast object
    nf = NeuralForecast(models=models, freq='M')
    nf.fit(df=nf_df)
    
    # Generate forecasts
    forecast = nf.predict()
    forecast_series = forecast['NHITS']
    forecast_series.index = pd.to_datetime(forecast['ds'])
    
    # Calculate percent change
    last_cpi_value = nf_df['y'].iloc[-1]
    forecast_pct_change = forecast_series.pct_change() * 100
    forecast_pct_change.iloc[0] = ((forecast_series.iloc[0] - last_cpi_value) / last_cpi_value) * 100
    
    return pd.DataFrame({
        'Forecasted CPI': forecast_series,
        'Percent Change': forecast_pct_change
    })

# Streamlit app
def main():
    st.title("CPI Time Series Forecast App")
    
    # Load and display raw data
    cpi_data = load_cpi_data()
    st.subheader("Raw CPI Data")
    st.write(cpi_data)
    
    # Plot raw data
    st.subheader("CPI Time Series Plot")
    plt.figure(figsize=(10, 6))
    plt.plot(cpi_data.index, cpi_data['CPI'], label='CPI')
    plt.xlabel('Date')
    plt.ylabel('CPI')
    plt.title('CPI Time Series')
    plt.legend()
    st.pyplot(plt)
    
    # Prepare data for N-HiTS
    nf_df = prepare_nf_data(cpi_data)
    
    # Forecasting horizons
    horizons = {'3 Months': 3, '12 Months': 12}
    
    for period_name, horizon in horizons.items():
        st.subheader(f"Forecast Next {period_name}")
        
        # ARIMA Forecast
        st.write("**ARIMA Model**")
        forecast_arima = forecast_cpi_arima(cpi_data['CPI'], horizon)
        st.write(forecast_arima)
        
        # N-HiTS Forecast
        st.write("**N-HiTS Model**")
        forecast_nhits = forecast_cpi_nhits(nf_df, horizon)
        st.write(forecast_nhits)
        
        # Plot both forecasts
        plt.figure(figsize=(12, 6))
        plt.plot(cpi_data.index, cpi_data['CPI'], label='Historical CPI', color='blue')
        plt.plot(forecast_arima.index, forecast_arima['Forecasted CPI'], label='ARIMA Forecast', color='red', linestyle='--')
        plt.plot(forecast_nhits.index, forecast_nhits['Forecasted CPI'], label='N-HiTS Forecast', color='green', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('CPI')
        plt.title(f'CPI Forecast - Next {period_name}')
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
