#!/usr/bin/env python3
"""
Model inference script with memory profiling.

This script performs inference using trained forecasting models and profiles
memory usage during prediction. It demonstrates how to use the models for
real-time forecasting and monitors resource consumption.

Key Features:
- Loads pre-trained models for inference
- Generates next-day forecasts
- Profiles memory usage during inference
- Supports all three model types (ARIMA, Prophet, LSTM)
- Handles GPU/CPU device placement

Memory Profiling:
- Uses memory_profiler to track peak memory usage
- Helps identify memory-intensive operations
- Useful for production deployment planning

Usage:
    python scripts/inference.py
    
Output:
    - Next-day forecasts for each model
    - Peak memory usage during inference
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import joblib
from src.data.loader import load_data, load_config
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel
from src.models.lstm_dataloader import get_lstm_dataloader
import os
from memory_profiler import memory_usage

config = load_config()

def arima_inference():
    """
    Perform ARIMA model inference with memory profiling.
    
    This function:
    1. Loads processed data and pre-trained ARIMA model
    2. Generates next-day forecast
    3. Profiles memory usage during inference
    4. Reports forecast value and peak memory consumption
    
    Note:
        ARIMA inference is typically very fast and memory-efficient
        since it only requires the fitted model and generates a single forecast.
    """
    df = load_data(processed=True)
    y = df['sales']
    model_path = 'models/arima/arima_model.joblib'
    
    if os.path.exists(model_path):
        # Load pre-trained ARIMA model
        fitted = joblib.load(model_path)
        
        # Define inference function for memory profiling
        def _infer():
            return fitted.forecast(steps=1)
        
        # Profile memory usage during inference
        mem_usage = memory_usage(_infer, retval=True)
        pred = mem_usage[1]
        
        print('ARIMA next day forecast:', pred.values)
        print(f'ARIMA inference peak memory: {max(mem_usage[0]):.2f} MiB')
    else:
        print('ARIMA model not found.')

def prophet_inference():
    """
    Perform Prophet model inference with memory profiling.
    
    This function:
    1. Loads processed data and pre-trained Prophet model
    2. Prepares future dates for forecasting
    3. Generates next-day forecast with all regressors
    4. Profiles memory usage during inference
    5. Reports forecast value and peak memory consumption
    
    Note:
        Prophet inference includes feature engineering and can be more
        memory-intensive than ARIMA due to additional regressors.
    """
    df = load_data(processed=True)
    
    # Use all engineered features as regressors
    regressor_cols = [c for c in df.columns if c not in ['date', 'sales', 'product_id']]
    prophet_df = df[['date', 'sales'] + regressor_cols].rename(columns={'date': 'ds', 'sales': 'y'})
    
    model_path = 'models/prophet/prophet_model.joblib'
    if os.path.exists(model_path):
        # Load pre-trained Prophet model
        model = joblib.load(model_path)
        
        # Prepare future dates for forecasting
        future = prophet_df[['ds'] + regressor_cols].copy()
        next_row = future.iloc[[-1]].copy()
        next_row['ds'] = next_row['ds'].max() + pd.Timedelta(days=1)
        future = pd.concat([future, next_row], ignore_index=True)
        
        # Define inference function for memory profiling
        def _infer():
            return model.model.predict(future)
        
        # Profile memory usage during inference
        mem_usage = memory_usage(_infer, retval=True)
        forecast = mem_usage[1]
        
        print('Prophet next day forecast:', forecast.iloc[-1]['yhat'])
        print(f'Prophet inference peak memory: {max(mem_usage[0]):.2f} MiB')
    else:
        print('Prophet model not found.')

def lstm_inference():
    """
    Perform LSTM model inference with memory profiling.
    
    This function:
    1. Loads processed data and pre-trained LSTM model
    2. Applies feature scaling
    3. Creates input sequence from recent data
    4. Generates next-day forecast
    5. Profiles memory usage during inference
    6. Reports forecast value and peak memory consumption
    
    Note:
        LSTM inference is typically the most memory-intensive due to:
        - PyTorch model loading and GPU operations
        - Feature scaling and sequence preparation
        - Neural network forward pass
    """
    df = load_data(processed=True)
    params = config['lstm']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from src.models.lstm_dataloader import LSTMDataset
    import joblib
    
    # Load saved feature columns and scaler
    feature_cols = joblib.load('models/lstm/feature_cols.joblib')
    scaler = joblib.load('models/lstm/feature_scaler.joblib')
    
    # Apply feature scaling
    df[feature_cols] = scaler.transform(df[feature_cols])
    
    # Initialize LSTM model with same architecture as training
    model = LSTMModel(
        input_size=len(feature_cols),
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout']
    ).to(device)
    
    model_path = 'models/lstm/lstm_model.pt'
    if os.path.exists(model_path):
        # Load pre-trained model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Use last sequence for prediction, with all features
        last_seq = df.sort_values('date').iloc[-params['sequence_length']:][feature_cols].values
        x = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Define inference function for memory profiling
        def _infer():
            with torch.no_grad():
                return model(x)
        
        # Profile memory usage during inference
        mem_usage = memory_usage(_infer, retval=True)
        pred = mem_usage[1]
        
        print('LSTM next day forecast:', pred.cpu().numpy().flatten())
        print(f'LSTM inference peak memory: {max(mem_usage[0]):.2f} MiB')
    else:
        print('LSTM model not found.')

def main():
    """
    Main function to run inference on all models.
    
    This function:
    1. Runs inference on ARIMA model
    2. Runs inference on Prophet model  
    3. Runs inference on LSTM model
    4. Reports forecasts and memory usage for each model
    
    Note:
        All models must be pre-trained and saved in their respective directories
        before running inference.
    """
    print("=" * 60)
    print("MODEL INFERENCE WITH MEMORY PROFILING")
    print("=" * 60)
    
    # Run inference on all models
    arima_inference()
    print("-" * 40)
    
    prophet_inference()
    print("-" * 40)
    
    lstm_inference()
    print("-" * 40)
    
    print("=" * 60)
    print("Inference complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
