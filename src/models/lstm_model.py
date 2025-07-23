#!/usr/bin/env python3
"""
LSTM (Long Short-Term Memory) neural network model for time series forecasting.

This module provides a PyTorch implementation of an LSTM model specifically
designed for time series forecasting. LSTM models are effective for capturing
complex temporal dependencies and non-linear patterns in time series data.

Key Features:
- Multi-layer LSTM architecture
- Configurable hidden size and number of layers
- Dropout regularization for preventing overfitting
- Batch-first input format for efficient training

LSTM Advantages:
- Captures long-term dependencies
- Handles variable-length sequences
- Can learn complex non-linear patterns
- Robust to noise in time series data
"""
import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM neural network for time series forecasting.
    
    This class implements a multi-layer LSTM with a fully connected output layer
    for predicting the next value in a time series based on a sequence of inputs.
    
    Architecture:
        Input -> LSTM Layers -> Dropout -> Fully Connected -> Output
        
    Attributes:
        lstm: Multi-layer LSTM module
        fc: Fully connected output layer
        
    Example:
        >>> model = LSTMModel(input_size=6, hidden_size=64, num_layers=2, dropout=0.2)
        >>> # Input shape: (batch_size, sequence_length, input_size)
        >>> output = model(x)  # Output shape: (batch_size,)
    """
    
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        """
        Initialize LSTM model with specified architecture.
        
        Args:
            input_size (int): Number of input features per time step
            hidden_size (int): Number of hidden units in LSTM layers
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate between LSTM layers (0.0 to 1.0)
            
        Note:
            - input_size should match the number of features in your data
            - hidden_size controls model capacity (larger = more complex patterns)
            - num_layers > 1 can capture more complex temporal relationships
            - dropout helps prevent overfitting (0.1-0.5 is typical)
        """
        super(LSTMModel, self).__init__()
        
        # Multi-layer LSTM with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input format: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0  # Dropout only between layers
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
                
        Returns:
            torch.Tensor: Predicted values of shape (batch_size,)
            
        Note:
            The model uses only the last time step output from the LSTM
            for making predictions, which is typical for time series forecasting.
        """
        # Pass through LSTM layers
        # out shape: (batch_size, sequence_length, hidden_size)
        # hidden states are not used for prediction
        out, _ = self.lstm(x)
        
        # Take only the last time step output
        # out[:, -1, :] shape: (batch_size, hidden_size)
        out = out[:, -1, :]
        
        # Pass through fully connected layer
        # out shape: (batch_size, 1)
        out = self.fc(out)
        
        # Remove the last dimension to get (batch_size,)
        return out.squeeze(-1)
