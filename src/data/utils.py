#!/usr/bin/env python3
"""
Data preprocessing utilities for the forecasting project.

This module provides comprehensive data preprocessing functions for time series
sales data, including feature engineering, scaling, and data cleaning operations.

Key Functions:
- preprocess_sales_data(): Main preprocessing pipeline
- standardize_column_names(): Normalize column names
- get_lstm_input_features(): Extract features for LSTM models
- scale_all_features(): Scale features using MinMaxScaler
- scale_sales_column(): Scale sales column specifically
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def standardize_column_names(df):
    """
    Standardize column names to snake_case format.
    
    Converts all column names to lowercase, removes extra spaces,
    and replaces spaces with underscores for consistency.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        pandas.DataFrame: Dataframe with standardized column names
        
    Example:
        >>> df = pd.DataFrame({'Product Category': [1], 'Total Amount': [100]})
        >>> df = standardize_column_names(df)
        >>> print(df.columns)
        Index(['product_category', 'total_amount'], dtype='object')
    """
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df

def get_lstm_input_features(df):
    """
    Extract feature columns suitable for LSTM model input.
    
    This function identifies numeric columns that can be used as features
    for LSTM models, excluding the target variable (sales) and non-numeric
    columns like dates and product IDs.
    
    Args:
        df (pandas.DataFrame): Input dataframe with processed features
        
    Returns:
        list: List of column names suitable for LSTM input features
        
    Note:
        Features include:
        - Time-based features: day_of_week, month, is_weekend
        - Lagged features: sales_1, sales_7 (previous day/week sales)
        - Rolling features: sales_rolling_7 (7-day moving average)
        
    Example:
        >>> features = get_lstm_input_features(df)
        >>> print(features)
        ['day_of_week', 'month', 'is_weekend', 'sales_1', 'sales_7', 'sales_rolling_7']
    """
    # Only use lagged/rolling/statistical features, not current sales
    exclude = ['date', 'product_id', 'sales']
    # Only include columns that are numeric and are not 'sales' itself
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c]) and (c.startswith('sales_') or c in ['day_of_week', 'month', 'is_weekend'])]
    return feature_cols

def scale_all_features(df, scaler_path=None, fit=True, feature_cols_path=None, feature_cols=None):
    """
    Scale all feature columns using MinMaxScaler with persistence.
    
    This function scales numeric features to the range [0, 1] using MinMaxScaler.
    It can either fit a new scaler or load an existing one, and saves both the
    scaler and feature column list for consistent inference.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        scaler_path (str, optional): Path to save/load the scaler
        fit (bool): If True, fit a new scaler; if False, load existing scaler
        feature_cols_path (str, optional): Path to save/load feature column list
        feature_cols (list, optional): List of feature columns to scale
        
    Returns:
        tuple: (df, scaler, feature_cols) where:
            - df: Dataframe with scaled features
            - scaler: Fitted or loaded MinMaxScaler
            - feature_cols: List of feature column names
            
    Example:
        >>> df_scaled, scaler, features = scale_all_features(df, fit=True)
        >>> # Later, for inference:
        >>> df_new, scaler, features = scale_all_features(df_new, fit=False, 
        ...                                              scaler_path='scaler.joblib')
    """
    if feature_cols is None:
        feature_cols = get_lstm_input_features(df)
    
    if fit:
        # Fit new scaler and transform data
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        # Save scaler and feature columns for later use
        if scaler_path:
            joblib.dump(scaler, scaler_path)
        if feature_cols_path:
            joblib.dump(feature_cols, feature_cols_path)
    else:
        # Load existing scaler and feature columns
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(feature_cols_path)
        df[feature_cols] = scaler.transform(df[feature_cols])
    
    return df, scaler, feature_cols

def preprocess_sales_data(
    df,
    date_col='date',
    product_col='product_category',
    sales_col='total_amount',
    fill_method='zero',
    add_time_features=True,
    outlier_clip='iqr',
    add_lag_features=True,
    scale_features=False,
    scaler_path=None,
    feature_cols_path=None,
    verbose=True
):
    """
    Comprehensive preprocessing pipeline for sales time series data.
    
    This function performs a complete preprocessing pipeline including:
    - Column name standardization
    - Date parsing and aggregation
    - Missing value handling
    - Outlier detection and clipping
    - Time-based feature engineering
    - Lagged feature creation
    - Feature scaling (optional)
    
    Args:
        df (pandas.DataFrame): Raw sales data
        date_col (str): Name of the date column
        product_col (str): Name of the product category column
        sales_col (str): Name of the sales/amount column
        fill_method (str): Method for handling missing values ('zero', 'ffill', 'interpolate')
        add_time_features (bool): Whether to add time-based features
        outlier_clip (str or float): Outlier clipping method ('iqr' or numeric threshold)
        add_lag_features (bool): Whether to add lagged and rolling features
        scale_features (bool): Whether to scale features using MinMaxScaler
        scaler_path (str, optional): Path to save the fitted scaler
        feature_cols_path (str, optional): Path to save the feature column list
        verbose (bool): Whether to print processing information
        
    Returns:
        tuple: (daily, feature_cols) where:
            - daily: Processed dataframe with all features
            - feature_cols: List of feature columns for modeling
            
    Example:
        >>> processed_df, features = preprocess_sales_data(
        ...     df, 
        ...     date_col='Date',
        ...     product_col='Product Category',
        ...     sales_col='Total Amount',
        ...     add_time_features=True,
        ...     add_lag_features=True
        ... )
    """
    # Step 1: Standardize column names
    df = standardize_column_names(df)
    date_col = date_col.lower()
    product_col = product_col.lower()
    sales_col = sales_col.lower()
    
    # Step 2: Parse dates and aggregate to daily level
    df[date_col] = pd.to_datetime(df[date_col])
    daily = df.groupby([date_col, product_col])[sales_col].sum().reset_index()
    
    # Step 3: Create complete time series (fill missing dates)
    all_dates = pd.date_range(daily[date_col].min(), daily[date_col].max())
    all_products = daily[product_col].unique()
    full_idx = pd.MultiIndex.from_product([all_dates, all_products], names=[date_col, product_col])
    daily = daily.set_index([date_col, product_col]).reindex(full_idx, fill_value=np.nan).reset_index()
    
    # Step 4: Handle missing values
    if fill_method == 'zero':
        daily[sales_col] = daily[sales_col].fillna(0)
    elif fill_method == 'ffill':
        daily[sales_col] = daily[sales_col].fillna(method='ffill').fillna(0)
    else:  # interpolate
        daily[sales_col] = daily[sales_col].interpolate().fillna(0)
    
    # Step 5: Handle outliers
    if outlier_clip == 'iqr':
        # Use IQR method to detect and clip outliers
        Q1 = daily[sales_col].quantile(0.25)
        Q3 = daily[sales_col].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        lower = Q1 - 1.5 * IQR
        daily[sales_col] = daily[sales_col].clip(lower, upper)
    elif isinstance(outlier_clip, (int, float)):
        # Use simple threshold clipping
        daily[sales_col] = daily[sales_col].clip(upper=outlier_clip)
    
    # Step 6: Add time-based features
    if add_time_features:
        daily['day_of_week'] = daily[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
        daily['month'] = daily[date_col].dt.month  # 1-12
        daily['is_weekend'] = (daily['day_of_week'] >= 5).astype(int)  # 1 for weekend, 0 for weekday
    
    # Step 7: Add lagged and rolling features
    if add_lag_features:
        daily = daily.sort_values([product_col, date_col])
        # Previous day sales
        daily['sales_1'] = daily.groupby(product_col)[sales_col].shift(1)
        # Previous week sales (7 days ago)
        daily['sales_7'] = daily.groupby(product_col)[sales_col].shift(7)
        # 7-day rolling average
        daily['sales_rolling_7'] = daily.groupby(product_col)[sales_col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        # Fill missing values in lagged features
        daily[['sales_1', 'sales_7']] = daily[['sales_1', 'sales_7']].fillna(0)
        daily['sales_rolling_7'] = daily['sales_rolling_7'].fillna(0)
    
    # Step 8: Standardize final column names
    daily = daily.rename(columns={date_col: 'date', product_col: 'product_id', sales_col: 'sales'})
    
    # Step 9: Get feature columns for LSTM
    feature_cols = get_lstm_input_features(daily)
    
    # Step 10: Scale features if requested
    if scale_features:
        daily, scaler, feature_cols = scale_all_features(
            daily, 
            scaler_path=scaler_path, 
            fit=True, 
            feature_cols_path=feature_cols_path, 
            feature_cols=feature_cols
        )
    
    # Step 11: Print summary if verbose
    if verbose:
        print(f"Processed shape: {daily.shape}")
        print(f"Columns: {daily.columns.tolist()}")
        print(f"LSTM input features: {feature_cols}")
    
    return daily, feature_cols


def scale_sales_column(df, scaler=None, fit=True):
    """
    Scale the sales column using MinMaxScaler.
    
    This function specifically scales the sales column to the range [0, 1],
    which is useful for models that require normalized target variables.
    
    Args:
        df (pandas.DataFrame): Input dataframe with 'sales' column
        scaler (MinMaxScaler, optional): Pre-fitted scaler. If None, creates new one
        fit (bool): If True, fit the scaler; if False, use existing scaler
        
    Returns:
        tuple: (df, scaler) where:
            - df: Dataframe with additional 'sales_scaled' column
            - scaler: Fitted MinMaxScaler
            
    Example:
        >>> df_scaled, scaler = scale_sales_column(df, fit=True)
        >>> # For inference with new data:
        >>> df_new_scaled, _ = scale_sales_column(df_new, scaler=scaler, fit=False)
    """
    if scaler is None:
        scaler = MinMaxScaler()
    
    if fit:
        # Fit scaler and transform sales column
        df['sales_scaled'] = scaler.fit_transform(df[['sales']])
    else:
        # Use existing scaler to transform sales column
        df['sales_scaled'] = scaler.transform(df[['sales']])
    
    return df, scaler
