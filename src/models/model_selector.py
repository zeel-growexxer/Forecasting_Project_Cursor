#!/usr/bin/env python3
"""
Intelligent model selection system for time series forecasting.

This module provides an automated model selection system that evaluates
different forecasting models (ARIMA, Prophet, LSTM) and selects the best
performing model based on various metrics and strategies.

Key Features:
- Performance-based model selection
- Multiple selection strategies (composite score, individual metrics)
- Model drift detection
- Historical performance tracking
- Automated recommendations

Selection Strategies:
- Composite: Weighted combination of MAE, RMSE, MAPE
- Best MAE: Model with lowest Mean Absolute Error
- Best RMSE: Model with lowest Root Mean Square Error
- Best MAPE: Model with lowest Mean Absolute Percentage Error
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import mlflow
from src.data.loader import load_config
from src.notifications.alert_manager import alert_manager

logger = logging.getLogger(__name__)


class ModelSelector:
    """
    Intelligent model selection based on performance metrics.
    
    This class provides automated model selection capabilities by evaluating
    the performance of different forecasting models and selecting the best
    one based on various criteria and strategies.
    
    Attributes:
        config: Configuration settings
        models (list): List of available models ['arima', 'prophet', 'lstm']
        metrics (list): Performance metrics to evaluate ['mae', 'rmse', 'mape']
        performance_history (list): Historical performance data
        current_best_model (str): Currently selected best model
        
    Example:
        >>> selector = ModelSelector()
        >>> performance = selector.get_latest_model_performance()
        >>> best_model, scores = selector.select_best_model(performance)
    """
    
    def __init__(self, config_path='config.ini'):
        """
        Initialize the model selector with configuration.
        
        Args:
            config_path (str): Path to configuration file
            
        Note:
            The selector evaluates three models: ARIMA, Prophet, and LSTM
            using three metrics: MAE, RMSE, and MAPE for comprehensive evaluation.
        """
        self.config = load_config(config_path)
        self.models = ['arima', 'prophet', 'lstm']
        self.metrics = ['mae', 'rmse', 'mape']
        self.performance_history = []
        self.current_best_model = None
        
    def get_latest_model_performance(self, experiment_name: str = None) -> Dict:
        """
        Get latest performance metrics for all models from MLflow.
        
        This method retrieves the most recent performance metrics for each model
        from the MLflow tracking server, including MAE, RMSE, MAPE, and training time.
        
        Args:
            experiment_name (str, optional): MLflow experiment name. 
                                          Uses config default if None.
            
        Returns:
            dict: Dictionary with model names as keys and performance metrics as values.
                 Each model's metrics include: mae, rmse, mape, training_time, 
                 last_updated, and run_id.
                 
        Example:
            >>> performance = selector.get_latest_model_performance()
            >>> print(performance['arima']['mae'])
            >>> # Output: 12.5
        """
        if experiment_name is None:
            experiment_name = self.config['mlflow']['experiment_name']
        
        try:
            # Set up MLflow tracking
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                logger.warning(f"Experiment {experiment_name} not found")
                return {}
            
            performance_data = {}
            
            # Get latest performance for each model
            for model_name in self.models:
                # Get the latest run for each model
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    filter_string=f"tags.model_name = '{model_name}'",
                    order_by=["start_time DESC"],
                    max_results=1
                )
                
                if not runs.empty:
                    run = runs.iloc[0]
                    performance_data[model_name] = {
                        'mae': run.get('metrics.mae', float('inf')),
                        'rmse': run.get('metrics.rmse', float('inf')),
                        'mape': run.get('metrics.mape', float('inf')),
                        'training_time': run.get('metrics.training_time', 0),
                        'last_updated': run['start_time'],
                        'run_id': run['run_id']
                    }
                else:
                    logger.warning(f"No runs found for model {model_name}")
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
    
    def calculate_composite_score(self, metrics: Dict, weights: Dict = None) -> float:
        """Calculate composite performance score"""
        if weights is None:
            weights = {'mae': 0.4, 'rmse': 0.4, 'mape': 0.2}
        
        # Normalize metrics (lower is better for all)
        normalized_scores = {}
        for metric in self.metrics:
            if metric in metrics and metrics[metric] != float('inf'):
                # Simple normalization - could be improved with historical data
                normalized_scores[metric] = 1.0 / (1.0 + metrics[metric])
            else:
                normalized_scores[metric] = 0.0
        
        # Calculate weighted score
        composite_score = sum(
            weights[metric] * normalized_scores[metric] 
            for metric in self.metrics 
            if metric in weights
        )
        
        return composite_score
    
    def select_best_model(self, performance_data: Dict, strategy: str = 'composite') -> Tuple[str, Dict]:
        """Select the best model based on specified strategy"""
        
        if not performance_data:
            logger.warning("No performance data available")
            return None, {}
        
        if strategy == 'composite':
            # Use composite score
            best_model = None
            best_score = -1
            scores = {}
            
            for model_name, metrics in performance_data.items():
                score = self.calculate_composite_score(metrics)
                scores[model_name] = score
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
            
            logger.info(f"Model scores: {scores}")
            logger.info(f"Selected model: {best_model} (score: {best_score:.4f})")
            
            return best_model, {
                'strategy': strategy,
                'scores': scores,
                'selected_model': best_model,
                'selected_score': best_score
            }
        
        elif strategy == 'mae':
            # Select based on MAE only
            best_model = min(performance_data.keys(), 
                           key=lambda x: performance_data[x].get('mae', float('inf')))
            return best_model, {'strategy': strategy, 'selected_model': best_model}
        
        elif strategy == 'rmse':
            # Select based on RMSE only
            best_model = min(performance_data.keys(), 
                           key=lambda x: performance_data[x].get('rmse', float('inf')))
            return best_model, {'strategy': strategy, 'selected_model': best_model}
        
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
    
    def check_model_drift(self, performance_data: Dict, threshold: float = 0.1) -> Dict:
        """Check for model performance drift"""
        drift_alerts = {}
        
        for model_name, metrics in performance_data.items():
            # Get historical performance for comparison
            historical_mae = self._get_historical_metric(model_name, 'mae')
            
            if historical_mae is not None:
                current_mae = metrics.get('mae', float('inf'))
                if current_mae != float('inf'):
                    drift_ratio = abs(current_mae - historical_mae) / historical_mae
                    
                    if drift_ratio > threshold:
                        drift_alerts[model_name] = {
                            'metric': 'mae',
                            'current': current_mae,
                            'historical': historical_mae,
                            'drift_ratio': drift_ratio,
                            'threshold': threshold
                        }
                        
                        # Send alert
                        alert_manager.performance_alert(
                            model_name, 'MAE', current_mae, historical_mae * (1 + threshold)
                        )
        
        return drift_alerts
    
    def _get_historical_metric(self, model_name: str, metric: str, days_back: int = 30) -> Optional[float]:
        """Get historical average metric for drift detection"""
        try:
            mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
            experiment = mlflow.get_experiment_by_name(self.config['mlflow']['experiment_name'])
            
            if experiment is None:
                return None
            
            # Get runs from last N days - without timestamp filter to avoid parsing issues
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.model_name = '{model_name}'",
                order_by=["start_time DESC"],
                max_results=10
            )
            
            if not runs.empty:
                metric_values = runs[f'metrics.{metric}'].dropna()
                if len(metric_values) > 0:
                    return metric_values.mean()
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting historical metric: {e}")
            return None
    
    def update_model_selection(self, force_recalculation: bool = False) -> Dict:
        """Update model selection and return results"""
        logger.info("Updating model selection...")
        
        # Get latest performance data
        performance_data = self.get_latest_model_performance()
        
        if not performance_data:
            logger.warning("No performance data available for model selection")
            return {}
        
        # Check for model drift
        drift_alerts = self.check_model_drift(performance_data)
        
        # Select best model
        best_model, selection_info = self.select_best_model(performance_data)
        
        # Update current best model if changed
        if best_model != self.current_best_model or force_recalculation:
            old_model = self.current_best_model
            self.current_best_model = best_model
            
            # Log the change
            if old_model is not None:
                logger.info(f"Model selection changed: {old_model} -> {best_model}")
                
                # Send notification about model change
                alert_manager.send_slack_alert(
                    f"🔄 Model selection updated: {old_model} → {best_model}\n"
                    f"Reason: Better performance based on {selection_info.get('strategy', 'composite')} strategy",
                    priority='normal'
                )
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'performance_data': performance_data,
            'selected_model': best_model,
            'selection_info': selection_info,
            'drift_alerts': drift_alerts
        })
        
        # Keep only last 100 entries
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        return {
            'selected_model': best_model,
            'performance_data': performance_data,
            'selection_info': selection_info,
            'drift_alerts': drift_alerts,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_model_recommendation(self, use_case: str = 'general') -> Dict:
        """Get model recommendation based on use case"""
        recommendations = {
            'general': {
                'primary': 'composite',
                'fallback': 'mae'
            },
            'accuracy_focused': {
                'primary': 'mae',
                'fallback': 'rmse'
            },
            'robustness_focused': {
                'primary': 'rmse',
                'fallback': 'mae'
            },
            'interpretability_focused': {
                'primary': 'arima',  # ARIMA is most interpretable
                'fallback': 'prophet'
            }
        }
        
        strategy = recommendations.get(use_case, recommendations['general'])['primary']
        
        if strategy in ['arima', 'prophet', 'lstm']:
            # Direct model recommendation
            return {
                'recommended_model': strategy,
                'reason': f'Direct recommendation for {use_case} use case',
                'confidence': 'high'
            }
        else:
            # Performance-based recommendation
            selection_result = self.update_model_selection()
            return {
                'recommended_model': selection_result.get('selected_model'),
                'reason': f'Performance-based selection using {strategy} strategy',
                'confidence': 'medium',
                'performance_data': selection_result.get('performance_data', {})
            }

# Global model selector instance
model_selector = ModelSelector() 