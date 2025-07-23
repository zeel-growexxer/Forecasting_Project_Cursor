import os
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from src.data.loader import load_config

logger = logging.getLogger(__name__)

class DataMonitor:
    """Monitor for new data files and trigger preprocessing"""
    
    def __init__(self, config_path='config.ini'):
        self.config = load_config(config_path)
        self.raw_path = self.config['data']['raw_path']
        self.processed_path = self.config['data']['processed_path']
        self.last_processed_time = None
        self._load_last_processed_time()
    
    def _load_last_processed_time(self):
        """Load the timestamp of last processed data"""
        processed_file = Path(self.processed_path)
        if processed_file.exists():
            self.last_processed_time = datetime.fromtimestamp(processed_file.stat().st_mtime)
        else:
            self.last_processed_time = datetime.min
    
    def check_for_new_data(self):
        """Check if new raw data is available"""
        raw_file = Path(self.raw_path)
        
        if not raw_file.exists():
            logger.warning(f"Raw data file not found: {self.raw_path}")
            return False
        
        raw_file_time = datetime.fromtimestamp(raw_file.stat().st_mtime)
        
        if raw_file_time > self.last_processed_time:
            logger.info(f"New data detected! Raw file modified: {raw_file_time}")
            return True
        
        logger.info("No new data detected")
        return False
    
    def validate_raw_data(self):
        """Validate raw data format and content"""
        try:
            df = pd.read_csv(self.raw_path)
            
            # Check required columns
            required_cols = ['Date', 'Product Category', 'Total Amount']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Check for empty data
            if df.empty:
                raise ValueError("Raw data file is empty")
            
            # Check for reasonable date range
            df['Date'] = pd.to_datetime(df['Date'])
            date_range = df['Date'].max() - df['Date'].min()
            if date_range.days < 7:
                logger.warning("Data spans less than 7 days")
            
            logger.info(f"Raw data validation passed. Shape: {df.shape}, Date range: {df['Date'].min()} to {df['Date'].max()}")
            return True
            
        except Exception as e:
            logger.error(f"Raw data validation failed: {e}")
            return False
    
    def trigger_preprocessing(self):
        """Trigger data preprocessing"""
        try:
            import subprocess
            result = subprocess.run(['python', 'scripts/preprocess.py'], 
                                  capture_output=True, text=True, check=True)
            logger.info("Preprocessing triggered successfully")
            self._load_last_processed_time()  # Update timestamp
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Preprocessing failed: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def monitor_continuously(self, check_interval=300):  # 5 minutes
        """Continuously monitor for new data"""
        logger.info(f"Starting continuous monitoring. Check interval: {check_interval} seconds")
        
        while True:
            try:
                if self.check_for_new_data():
                    if self.validate_raw_data():
                        if self.trigger_preprocessing():
                            logger.info("New data processed successfully")
                        else:
                            logger.error("Failed to process new data")
                    else:
                        logger.error("New data validation failed")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(check_interval)

def main():
    """Main function for data monitoring"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    monitor = DataMonitor()
    
    # Check once
    if monitor.check_for_new_data():
        if monitor.validate_raw_data():
            monitor.trigger_preprocessing()
    
    # Or run continuously
    # monitor.monitor_continuously()

if __name__ == '__main__':
    main() 