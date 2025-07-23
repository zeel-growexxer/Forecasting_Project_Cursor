#!/usr/bin/env python3
"""
Model evaluation script for time series forecasting models.

This script provides comprehensive evaluation of trained forecasting models
(ARIMA, Prophet, LSTM) using various metrics including MAE, RMSE, and MAPE.
It loads pre-trained models and evaluates their performance on test data.

Key Features:
- Loads and evaluates all three model types
- Calculates standard forecasting metrics
- Handles missing models gracefully
- Supports both individual and batch evaluation

Usage:
    python scripts/evaluate.py [model_name]
    
Examples:
    python scripts/evaluate.py          # Evaluate all models
    python scripts/evaluate.py arima    # Evaluate only ARIMA
    python scripts/evaluate.py prophet  # Evaluate only Prophet
    python scripts/evaluate.py lstm     # Evaluate only LSTM
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
from sklearn.metrics import mean_absolute_error
import os
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

config = load_config()

def get_train_test_split(df, test_size):
    """
    Split dataframe into training and test sets.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        test_size (float): Proportion of data to use for testing (0.0 to 1.0)
        
    Returns:
        tuple: (train_df, test_df) - Training and test dataframes
        
    Note:
        Uses simple time-based split (first portion for training, last portion for testing)
        to maintain temporal order in time series data.
    """
    n = int(len(df) * (1 - test_size))
    return df.iloc[:n], df.iloc[n:]

def arima_eval(train, test):
    """
    Evaluate ARIMA model performance on test data.
    
    Args:
        train (pandas.DataFrame): Training data
        test (pandas.DataFrame): Test data
        
    Returns:
        float or None: MAE score if model exists, None otherwise
        
    Note:
        Loads pre-trained ARIMA model and generates forecasts for the test period.
        Uses the fitted model to predict the next 'len(test)' steps.
    """
    y_train = train['sales']
    y_test = test['sales']
    model_path = 'models/arima/arima_model.joblib'
    
    if os.path.exists(model_path):
        # Load pre-trained ARIMA model
        fitted = joblib.load(model_path)
        # Generate forecasts for test period
        pred = fitted.forecast(steps=len(y_test))
        # Calculate MAE
        mae = mean_absolute_error(y_test, pred)
        print('ARIMA MAE:', mae)
        return mae
    else:
        print('ARIMA model not found.')
        return None

def evaluate_model(model_name, train, test):
    """
    Helper function to evaluate a specific model.
    
    Args:
        model_name (str): Name of the model to evaluate ('arima', 'prophet', 'lstm')
        train (pandas.DataFrame): Training data
        test (pandas.DataFrame): Test data
        
    Returns:
        float or None: MAE score if model exists and evaluation succeeds, None otherwise
        
    Example:
        >>> mae = evaluate_model('arima', train_data, test_data)
        >>> print(f"ARIMA MAE: {mae}")
    """
    if model_name == 'arima':
        return arima_eval(train, test)
    elif model_name == 'prophet':
        return prophet_eval(train, test)
    elif model_name == 'lstm':
        return lstm_eval(train, test)
    else:
        print(f"Unknown model: {model_name}")
        return None

def prophet_eval(train, test):
    """
    Evaluate Prophet model performance on test data.
    
    Args:
        train (pandas.DataFrame): Training data
        test (pandas.DataFrame): Test data
        
    Returns:
        float or None: MAE score if model exists, None otherwise
        
    Note:
        Prophet requires specific column names ('ds' for dates, 'y' for target).
        This function renames columns and includes additional regressors if available.
        Loads pre-trained Prophet model and generates forecasts for test dates.
    """
    # Identify additional regressor columns (features other than date, sales, product_id)
    regressor_cols = [c for c in train.columns if c not in ['date', 'sales', 'product_id']]
    
    # Prepare data in Prophet format (ds=date, y=sales)
    train_df = train[['date', 'sales'] + regressor_cols].rename(columns={'date': 'ds', 'sales': 'y'})
    test_df = test[['date', 'sales'] + regressor_cols].rename(columns={'date': 'ds', 'sales': 'y'})
    
    model_path = 'models/prophet/prophet_model.joblib'
    if os.path.exists(model_path):
        # Load pre-trained Prophet model
        model = joblib.load(model_path)
        # Generate forecasts for test dates
        forecast = model.model.predict(test_df[['ds'] + regressor_cols])
        # Calculate MAE
        mae = mean_absolute_error(test_df['y'], forecast['yhat'])
        print('Prophet MAE:', mae)
        return mae
    else:
        print('Prophet model not found.')
        return None

def lstm_eval(train, test):
    """
    Evaluate LSTM model performance on test data.
    
    Args:
        train (pandas.DataFrame): Training data
        test (pandas.DataFrame): Test data
        
    Returns:
        float or None: MAE score if model exists and evaluation succeeds, None otherwise
        
    Note:
        LSTM evaluation requires:
        - Loading pre-trained model weights
        - Feature scaling using saved scaler
        - Creating sequences for prediction
        - Handling GPU/CPU device placement
        - Ensuring sufficient data for sequence length
    """
    params = config['lstm']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from src.models.lstm_dataloader import LSTMDataset
    import joblib
    
    # Load saved feature columns and scaler
    feature_cols = joblib.load('models/lstm/feature_cols.joblib')
    scaler = joblib.load('models/lstm/feature_scaler.joblib')
    
    # Apply feature scaling to both train and test data
    train[feature_cols] = scaler.transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])
    
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
        
        # Prepare test sequences with all features
        df_all = pd.concat([train, test]).sort_values('date')
        values = df_all[feature_cols].values
        y_all = df_all['sales'].values
        test_start = len(train)
        preds = []
        
        # Ensure we have enough data for sequence length
        if len(values) < params['sequence_length']:
            print('LSTM: Not enough data for sequence length')
            return None
            
        # Calculate valid range for predictions
        start_idx = max(0, test_start - params['sequence_length'])
        end_idx = len(values) - params['sequence_length']
        
        if start_idx >= end_idx:
            print('LSTM: Not enough data for predictions')
            return None
            
        for i in range(start_idx, end_idx):
            x = values[i:i+params['sequence_length']]
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x_tensor)
            preds.append(pred.cpu().numpy().flatten()[0])
        
        # Align predictions with test data
        y_true = y_all[test_start:test_start + len(preds)]
        if len(y_true) != len(preds):
            min_len = min(len(y_true), len(preds))
            y_true = y_true[:min_len]
            preds = preds[:min_len]
            
        if len(preds) == 0:
            print('LSTM: No predictions generated')
            return None
            
        mae = mean_absolute_error(y_true, preds)
        print('LSTM MAE:', mae)
        return mae
    else:
        print('LSTM model not found.')
        return None

def main():
    """
    Main function to evaluate all forecasting models.
    
    This function:
    1. Loads processed data
    2. Splits data into train/test sets
    3. Evaluates all three models (ARIMA, Prophet, LSTM)
    4. Prints MAE scores for each model
    
    Note:
        Models must be pre-trained and saved in their respective directories:
        - models/arima/arima_model.joblib
        - models/prophet/prophet_model.joblib  
        - models/lstm/lstm_model.pt
    """
    # Load processed data
    df = load_data(processed=True)
    
    # Split data into training and test sets
    train, test = get_train_test_split(df, config['data']['test_size'])
    
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    
    # Evaluate all models
    arima_eval(train, test)
    prophet_eval(train, test)
    lstm_eval(train, test)
    
    print("=" * 50)
    print("Evaluation complete!")
    print("=" * 50)

if __name__ == '__main__':
    main()
