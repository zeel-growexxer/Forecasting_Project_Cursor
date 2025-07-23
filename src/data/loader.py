#!/usr/bin/env python3
"""
Data loading utilities for the forecasting project.

This module provides functions to:
1. Load configuration from INI files with type conversion
2. Load and validate CSV data files (raw or processed)
3. Handle different data formats and column name standardization
4. Provide data preprocessing on-the-fly when needed

The configuration system supports both local development and cloud deployment
scenarios with automatic type conversion for booleans, numbers, and lists.

Key Functions:
- load_config(): Loads and parses configuration from INI files
- load_data(): Loads CSV data with optional preprocessing
"""
import pandas as pd
import configparser
from src.data.utils import preprocess_sales_data


def load_config(config_path='config.ini'):
    """
    Load and parse configuration from an INI file with automatic type conversion.
    
    This function reads an INI configuration file and converts string values to
    appropriate Python types (boolean, integer, float, list) based on their content.
    
    Args:
        config_path (str): Path to the INI configuration file. Defaults to 'config.ini'.
        
    Returns:
        dict: Dictionary containing the parsed configuration with proper data types.
              Structure: {section_name: {key: value}}
              
    Example:
        config = load_config()
        # Access values like: config['data']['test_size'] (returns float)
        #                     config['arima']['order'] (returns list)
        #                     config['notifications']['email_enabled'] (returns bool)
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Convert to dictionary format for compatibility with existing code
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config[section].items():
            # Handle boolean values (true/false strings)
            if value.lower() in ['true', 'false']:
                config_dict[section][key] = value.lower() == 'true'
            # Handle numeric values (integers and floats)
            elif value.replace('.', '').replace('-', '').isdigit():
                if '.' in value:
                    config_dict[section][key] = float(value)
                else:
                    config_dict[section][key] = int(value)
            # Handle lists (for ARIMA order parameters like "1,1,1")
            elif ',' in value:
                config_dict[section][key] = [int(x.strip()) for x in value.split(',')]
            else:
                # Keep as string for other values
                config_dict[section][key] = value
    
    return config_dict

def load_data(processed=True):
    """
    Load sales data from CSV files with optional preprocessing.
    
    This function can load either preprocessed data (ready for modeling) or raw data
    with on-the-fly preprocessing. It handles column name standardization and
    missing value imputation.
    
    Args:
        processed (bool): If True, loads preprocessed data from the processed_path.
                         If False, loads raw data and applies preprocessing.
                         Defaults to True.
        
    Returns:
        pandas.DataFrame: Loaded and cleaned sales data with standardized column names.
                         Columns include: date, product_category, total_amount, etc.
                         
    Raises:
        FileNotFoundError: If the specified data file doesn't exist.
        ValueError: If required columns are missing from the data.
        
    Example:
        # Load preprocessed data (recommended for training)
        df = load_data(processed=True)
        
        # Load and preprocess raw data on-the-fly
        df = load_data(processed=False)
    """
    config = load_config()
    
    if processed:
        # Load preprocessed data (faster, already cleaned)
        path = config['data']['processed_path']
        df = pd.read_csv(path, parse_dates=['date'])
        
        # Standardize column names to snake_case for consistency
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        # Handle any remaining missing values with zero-fill
        df = df.fillna(0)
        return df
    else:
        # Load raw data and apply preprocessing
        path = config['data']['raw_path']
        df = pd.read_csv(path)
        
        # Apply preprocessing pipeline on-the-fly
        processed = preprocess_sales_data(
            df,
            date_col='Date',                    # Expected column name in raw data
            product_col='Product Category',     # Expected column name in raw data
            sales_col='Total Amount',           # Expected column name in raw data
            fill_method='zero',                 # Fill missing values with zeros
            add_time_features=True,             # Add day_of_week, month, etc.
            outlier_clip=None                   # Don't clip outliers for now
        )
        
        # Standardize column names to snake_case
        processed.columns = [c.lower().replace(' ', '_') for c in processed.columns]
        
        # Final missing value handling
        processed = processed.fillna(0)
        return processed
