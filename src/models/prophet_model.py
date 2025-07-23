#!/usr/bin/env python3
"""
Facebook Prophet model implementation for time series forecasting.

This module provides a wrapper around the Facebook Prophet library for
automated time series forecasting. Prophet is particularly effective for
time series with strong seasonal patterns and holiday effects.

Key Features:
- Automatic seasonality detection
- Holiday effects modeling
- Trend changepoint detection
- Robust to missing data and outliers

Prophet Advantages:
- Handles multiple seasonalities (yearly, weekly, daily)
- Automatically detects trend changes
- Provides uncertainty intervals
- Works well with irregular time series
"""
from prophet import Prophet
import pandas as pd


class ProphetModel:
    """
    Facebook Prophet model wrapper for time series forecasting.
    
    This class provides a simple interface to the Prophet library, making it
    easy to fit models and generate forecasts with automatic seasonality detection.
    
    Attributes:
        seasonality_mode (str): 'additive' or 'multiplicative' seasonality
        model: The underlying Prophet model
        fitted (bool): Whether the model has been fitted
        
    Example:
        >>> model = ProphetModel(seasonality_mode='additive')
        >>> model.fit(df)  # df must have 'ds' and 'y' columns
        >>> future = model.make_future_dataframe(periods=30)
        >>> forecast = model.predict(future)
    """
    
    def __init__(self, seasonality_mode='additive'):
        """
        Initialize Prophet model with specified seasonality mode.
        
        Args:
            seasonality_mode (str): Type of seasonality to model:
                - 'additive': Seasonality is added to trend (default)
                - 'multiplicative': Seasonality multiplies the trend
                
        Note:
            Additive seasonality is more common and works well for most cases.
            Multiplicative seasonality is useful when seasonal effects grow
            with the trend (e.g., percentage growth).
        """
        self.seasonality_mode = seasonality_mode
        self.model = Prophet(seasonality_mode=self.seasonality_mode)
        self.fitted = False

    def fit(self, df):
        """
        Fit the Prophet model to the provided time series data.
        
        Args:
            df (pandas.DataFrame): Training data with columns:
                - 'ds': Date column (datetime)
                - 'y': Target variable (numeric)
                
        Returns:
            None
            
        Raises:
            ValueError: If required columns are missing
            Exception: If Prophet model fitting fails
            
        Example:
            >>> df = pd.DataFrame({
            ...     'ds': pd.date_range('2020-01-01', periods=365),
            ...     'y': sales_data
            ... })
            >>> model.fit(df)
        """
        # Validate input data
        required_cols = ['ds', 'y']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Fit the Prophet model
        self.model.fit(df)
        self.fitted = True

    def predict(self, future):
        """
        Generate forecasts for the provided future dates.
        
        Args:
            future (pandas.DataFrame): Future dates with 'ds' column
                
        Returns:
            pandas.DataFrame: Forecast results with columns:
                - 'ds': Date
                - 'yhat': Predicted value
                - 'yhat_lower': Lower bound of prediction interval
                - 'yhat_upper': Upper bound of prediction interval
                
        Raises:
            ValueError: If model hasn't been fitted or 'ds' column is missing
            
        Example:
            >>> future = model.make_future_dataframe(periods=30)
            >>> forecast = model.predict(future)
            >>> print(f"Forecast shape: {forecast.shape}")
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if 'ds' not in future.columns:
            raise ValueError("Future dataframe must have 'ds' column")
        
        return self.model.predict(future)
