#!/usr/bin/env python3
"""
LSTM DataLoader implementation for time series forecasting.

This module provides PyTorch Dataset and DataLoader classes specifically
designed for LSTM time series forecasting. It handles sequence creation,
feature scaling, and batch generation for efficient training.

Key Features:
- Automatic sequence creation from time series data
- Feature scaling with persistence
- Support for multi-product time series
- Efficient batch generation for PyTorch training

Sequence Creation:
- Creates sliding windows of fixed length (sequence_length)
- Each sequence predicts the next value in the time series
- Handles both single and multi-product scenarios
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import joblib

class LSTMDataset(Dataset):
    """
    PyTorch Dataset for LSTM time series forecasting.
    
    This class creates sequences from time series data for LSTM training.
    It handles feature scaling, sequence creation, and supports both single
    and multi-product time series scenarios.
    
    Attributes:
        sequence_length (int): Length of input sequences
        target_col (str): Name of the target column
        id_col (str, optional): Product ID column for multi-product data
        scaler: Fitted scaler for feature normalization
        feature_cols (list): List of feature column names
        samples (list): List of (sequence, target) pairs
        
    Example:
        >>> dataset = LSTMDataset(df, sequence_length=30, target_col='sales')
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, df, sequence_length, target_col, id_col=None, scaler_path=None, feature_cols_path=None):
        """
        Initialize LSTM dataset with time series data.
        
        Args:
            df (pandas.DataFrame): Input dataframe with time series data
            sequence_length (int): Number of time steps in each input sequence
            target_col (str): Name of the target variable column
            id_col (str, optional): Product ID column for multi-product data
            scaler_path (str, optional): Path to load fitted scaler
            feature_cols_path (str, optional): Path to load feature column list
            
        Note:
            - sequence_length determines how many past time steps to use for prediction
            - If id_col is provided, sequences are created separately for each product
            - Feature scaling is applied if scaler_path is provided
        """
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.id_col = id_col
        self.scaler = None
        
        # Load or determine feature columns
        if feature_cols_path:
            self.feature_cols = joblib.load(feature_cols_path)
        else:
            # Default feature selection if no saved features
            self.feature_cols = [
                c for c in df.columns 
                if (c.startswith('sales_') or c in ['day_of_week', 'month', 'is_weekend']) 
                and pd.api.types.is_numeric_dtype(df[c])
            ]
        
        # Apply feature scaling if scaler is provided
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
            df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])
        
        self.samples = []
        self.prepare_samples(df)

    def prepare_samples(self, df):
        """
        Create training samples from time series data.
        
        This method creates sliding window sequences from the time series data.
        Each sequence contains 'sequence_length' time steps and predicts the
        next value in the series.
        
        Args:
            df (pandas.DataFrame): Input dataframe with time series data
            
        Note:
            - Creates sequences using sliding window approach
            - Each sequence: X[i:i+sequence_length] -> y[i+sequence_length]
            - For multi-product data, sequences are created separately per product
            - Sequences are sorted by date to maintain temporal order
        """
        if self.id_col:
            # Multi-product scenario: create sequences for each product separately
            for pid, group in df.groupby(self.id_col):
                # Sort by date to maintain temporal order
                group = group.sort_values('date')
                X = group[self.feature_cols].values
                y = group[self.target_col].values
                
                # Create sliding window sequences
                for i in range(len(y) - self.sequence_length):
                    x_seq = X[i:i+self.sequence_length]  # Input sequence
                    y_seq = y[i+self.sequence_length]    # Target value
                    self.samples.append((x_seq, y_seq))
        else:
            # Single time series scenario
            df = df.sort_values('date')
            X = df[self.feature_cols].values
            y = df[self.target_col].values
            
            # Create sliding window sequences
            for i in range(len(y) - self.sequence_length):
                x_seq = X[i:i+self.sequence_length]  # Input sequence
                y_seq = y[i+self.sequence_length]    # Target value
                self.samples.append((x_seq, y_seq))

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of training samples
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (x, y) where:
                - x: Input sequence tensor of shape (sequence_length, num_features)
                - y: Target value tensor of shape (1,)
        """
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def get_lstm_dataloader(df, batch_size, sequence_length, target_col, id_col=None, shuffle=True, scaler_path=None, feature_cols_path=None):
    """
    Create a PyTorch DataLoader for LSTM time series training.
    
    This function creates an LSTM dataset and wraps it in a DataLoader for
    efficient batch generation during training.
    
    Args:
        df (pandas.DataFrame): Input dataframe with time series data
        batch_size (int): Number of samples per batch
        sequence_length (int): Length of input sequences
        target_col (str): Name of the target variable column
        id_col (str, optional): Product ID column for multi-product data
        shuffle (bool): Whether to shuffle the data during training
        scaler_path (str, optional): Path to load fitted scaler
        feature_cols_path (str, optional): Path to load feature column list
        
    Returns:
        torch.utils.data.DataLoader: PyTorch DataLoader for LSTM training
        
    Example:
        >>> dataloader = get_lstm_dataloader(
        ...     df, batch_size=32, sequence_length=30, target_col='sales'
        ... )
        >>> for batch_x, batch_y in dataloader:
        ...     # batch_x shape: (batch_size, sequence_length, num_features)
        ...     # batch_y shape: (batch_size,)
        ...     pass
    """
    dataset = LSTMDataset(df, sequence_length, target_col, id_col, scaler_path, feature_cols_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
