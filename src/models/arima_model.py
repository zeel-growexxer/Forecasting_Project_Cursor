#!/usr/bin/env python3
"""
ARIMA (AutoRegressive Integrated Moving Average) model implementation.

This module provides a wrapper around the statsmodels ARIMA implementation
for time series forecasting. ARIMA models are particularly effective for
univariate time series with trends and seasonality.

Key Features:
- Configurable ARIMA order parameters (p, d, q)
- Simple fit/predict interface
- Integration with the forecasting pipeline

ARIMA Parameters:
- p: Order of autoregression (number of lag observations)
- d: Degree of differencing (number of times data is differenced)
- q: Order of moving average (size of moving average window)
"""
import statsmodels.api as sm
import numpy as np


class ARIMAModel:
    """
    ARIMA model wrapper for time series forecasting.
    
    This class provides a simple interface to the statsmodels ARIMA implementation,
    making it easy to fit models and generate forecasts.
    
    Attributes:
        order (tuple): ARIMA order parameters (p, d, q)
        model: The underlying statsmodels ARIMA model
        fitted_model: The fitted ARIMA model ready for predictions
        
    Example:
        >>> model = ARIMAModel(order=(1, 1, 1))
        >>> model.fit(sales_data)
        >>> predictions = model.predict(steps=30)
    """
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model with specified order parameters.
        
        Args:
            order (tuple): ARIMA order (p, d, q) where:
                - p: Order of autoregression
                - d: Degree of differencing  
                - q: Order of moving average
                
        Note:
            Common orders:
            - (1,1,1): Simple ARIMA with one lag, one difference, one moving average
            - (2,1,2): More complex model with two lags and moving averages
            - (0,1,1): Moving average model (MA(1))
        """
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, y):
        """
        Fit the ARIMA model to the provided time series data.
        
        Args:
            y (array-like): Time series data to fit the model to.
                          Should be a 1D array or pandas Series.
                          
        Returns:
            statsmodels.tsa.arima.model.ARIMAResults: Fitted ARIMA model
            
        Raises:
            ValueError: If the data is not suitable for ARIMA modeling
            np.linalg.LinAlgError: If the model cannot be fitted (e.g., singular matrix)
            
        Example:
            >>> model = ARIMAModel(order=(1, 1, 1))
            >>> fitted_model = model.fit(sales_series)
        """
        self.model = sm.tsa.ARIMA(y, order=self.order)
        self.fitted_model = self.model.fit()
        return self.fitted_model

    def predict(self, steps):
        """
        Generate forecasts for the specified number of steps ahead.
        
        Args:
            steps (int): Number of time steps to forecast into the future
            
        Returns:
            array: Forecasted values for the next 'steps' periods
            
        Raises:
            ValueError: If the model hasn't been fitted yet
            
        Example:
            >>> predictions = model.predict(steps=30)  # 30-day forecast
            >>> print(f"Next 30 days forecast: {predictions}")
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.fitted_model.forecast(steps=steps)
