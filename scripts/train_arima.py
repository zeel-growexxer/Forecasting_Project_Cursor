import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.data.loader import load_data, load_config
from src.models.arima_model import ARIMAModel
from src.tracking.mlflow_utils import set_experiment, start_run, log_params, log_metrics
import mlflow
import joblib
import os

config = load_config()
set_experiment(config['mlflow']['experiment_name'])

def main():
    df = load_data(processed=True)
    order = tuple(config['arima']['order'])
    y = df['sales']
    model = ARIMAModel(order=order)
    with start_run(run_name='arima_train'):
        # Add model name tag
        mlflow.set_tag("model_name", "arima")
        
        fitted = model.fit(y)
        log_params({'order': order})
        
        # Calculate and log performance metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        
        # Make predictions on training data for metrics
        predictions = fitted.predict(start=1, end=len(y))
        actual = y.iloc[1:len(predictions)+1]  # Align with predictions
        
        # Ensure same length
        min_len = min(len(actual), len(predictions))
        actual = actual.iloc[:min_len]
        predictions = predictions[:min_len]
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        log_metrics({
            'aic': fitted.aic,
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        })
        
        # Log model to MLflow
        mlflow.sklearn.log_model(fitted, 'arima_model')
        
        # Save model locally
        os.makedirs('models/arima', exist_ok=True)
        joblib.dump(fitted, 'models/arima/arima_model.joblib')

if __name__ == '__main__':
    main()
