from prefect import flow, task
import subprocess
import os
import logging
from datetime import datetime
import pandas as pd
from src.data.loader import load_config, load_data
from src.data.utils import preprocess_sales_data
from src.notifications.alert_manager import alert_manager
from src.models.model_selector import model_selector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@task
def check_new_data():
    """Check if new data is available and validate it"""
    config = load_config()
    raw_path = config['data']['raw_path']
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    
    # Check if file was modified recently (within last 24 hours)
    file_mtime = os.path.getmtime(raw_path)
    file_age_hours = (datetime.now().timestamp() - file_mtime) / 3600
    
    if file_age_hours > 24:
        logger.warning(f"Raw data file is {file_age_hours:.1f} hours old")
    
    # Basic validation of raw data
    try:
        df = pd.read_csv(raw_path)
        required_cols = ['Date', 'Product Category', 'Total Amount']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Raw data validation passed. Shape: {df.shape}")
        return True
    except Exception as e:
        logger.error(f"Raw data validation failed: {e}")
        raise

@task
def preprocess_data():
    """Preprocess raw data and save to processed path"""
    try:
        result = subprocess.run(['python', 'scripts/preprocess.py'], 
                              capture_output=True, text=True, check=True)
        logger.info("Data preprocessing completed successfully")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Data preprocessing failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise

@task
def validate_processed_data():
    """Validate that processed data is ready for training"""
    try:
        df = load_data(processed=True)
        required_cols = ['date', 'product_id', 'sales', 'day_of_week', 'month', 'is_weekend']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required processed columns: {missing_cols}")
        
        logger.info(f"Processed data validation passed. Shape: {df.shape}")
        return True
    except Exception as e:
        logger.error(f"Processed data validation failed: {e}")
        raise

@task
def retrain_arima():
    """Train ARIMA model with error handling"""
    try:
        result = subprocess.run(['python', 'scripts/train_arima.py'], 
                              capture_output=True, text=True, check=True)
        logger.info("ARIMA training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"ARIMA training failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise

@task
def retrain_prophet():
    """Train Prophet model with error handling"""
    try:
        result = subprocess.run(['python', 'scripts/train_prophet.py'], 
                              capture_output=True, text=True, check=True)
        logger.info("Prophet training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Prophet training failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise

@task
def retrain_lstm():
    """Train LSTM model with error handling"""
    try:
        result = subprocess.run(['python', 'scripts/train_lstm.py'], 
                              capture_output=True, text=True, check=True)
        logger.info("LSTM training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"LSTM training failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise

@task
def run_inference():
    """Run inference on latest models"""
    try:
        result = subprocess.run(['python', 'scripts/inference.py'], 
                              capture_output=True, text=True, check=True)
        logger.info("Inference completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        raise

@flow(name="retrain-all-models")
def retrain_all_models():
    """Main pipeline flow with proper error handling and logging"""
    start_time = datetime.now()
    logger.info("Starting retraining pipeline")
    
    try:
        # Step 1: Check for new data
        check_new_data()
        
        # Step 2: Preprocess data
        preprocess_data()
        
        # Step 3: Validate processed data
        validate_processed_data()
        
        # Step 4: Train models
        models_trained = []
        if retrain_arima():
            models_trained.append("ARIMA")
        if retrain_prophet():
            models_trained.append("Prophet")
        if retrain_lstm():
            models_trained.append("LSTM")
        
        # Step 5: Update model selection
        selection_result = model_selector.update_model_selection()
        logger.info(f"Model selection updated: {selection_result.get('selected_model', 'None')}")
        
        # Step 6: Run inference
        run_inference()
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        # Send success notification
        alert_manager.pipeline_success_alert(duration, models_trained)
        
        logger.info("Pipeline completed successfully!")
        return True
        
    except Exception as e:
        # Send failure notification
        alert_manager.pipeline_failure_alert(str(e), "Unknown step")
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == '__main__':
    retrain_all_models()
