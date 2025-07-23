#!/usr/bin/env python3
"""
Test script to validate pipeline functionality with new data
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import pandas as pd
from datetime import datetime, timedelta
from src.data.monitor import DataMonitor
from src.pipeline.retrain_flow import retrain_all_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data to simulate new data arrival"""
    from src.data.loader import load_config
    
    config = load_config()
    raw_path = config['data']['raw_path']
    
    # Create test data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    products = ['Electronics', 'Clothing', 'Books', 'Home & Garden']
    
    data = []
    for date in dates:
        for product in products:
            # Simulate some seasonality and trends
            base_sales = 100
            seasonal_factor = 1 + 0.3 * pd.Series(date).dt.dayofweek.values[0] / 7
            trend_factor = 1 + (date - pd.Timestamp('2024-01-01')).days / 365 * 0.1
            sales = int(base_sales * seasonal_factor * trend_factor)
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Product Category': product,
                'Total Amount': sales,
                'Product ID': f"{product[:3].upper()}{len(data):03d}"
            })
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    # Save test data
    df.to_csv(raw_path, index=False)
    logger.info(f"Test data created at {raw_path} with shape {df.shape}")
    
    return raw_path

def test_data_monitoring():
    """Test data monitoring functionality"""
    logger.info("Testing data monitoring...")
    
    monitor = DataMonitor()
    
    # Test new data detection
    has_new_data = monitor.check_for_new_data()
    logger.info(f"New data detected: {has_new_data}")
    
    # Test data validation
    is_valid = monitor.validate_raw_data()
    logger.info(f"Data validation passed: {is_valid}")
    
    return has_new_data and is_valid

def test_pipeline():
    """Test the complete pipeline"""
    logger.info("Testing complete pipeline...")
    
    try:
        # Run the pipeline
        result = retrain_all_models()
        logger.info("Pipeline test completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("Starting pipeline test...")
    
    # Step 1: Create test data
    create_test_data()
    
    # Step 2: Test data monitoring
    if not test_data_monitoring():
        logger.error("Data monitoring test failed")
        return False
    
    # Step 3: Test complete pipeline
    if not test_pipeline():
        logger.error("Pipeline test failed")
        return False
    
    logger.info("All tests passed! Pipeline is ready for production.")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1) 