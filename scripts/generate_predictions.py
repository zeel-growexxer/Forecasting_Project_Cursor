#!/usr/bin/env python3
"""
Generate predictions from all trained models and save them for dashboard display
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_data, load_config
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel

def generate_predictions():
    """Generate predictions from all models and save them"""
    print("🔮 Generating Model Predictions for Dashboard...")
    print("=" * 60)
    
    # Load data
    df = load_data(processed=True)
    config = load_config()
    
    # Prepare data for predictions
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Get unique product categories
    product_categories = df['product_id'].unique()
    
    # Generate predictions for each product category
    all_predictions = []
    
    for product_id in product_categories:
        print(f"📊 Generating predictions for {product_id}...")
        
        # Filter data for this product
        product_data = df[df['product_id'] == product_id].copy()
        product_data = product_data.sort_values('date')
        
        if len(product_data) < 30:  # Need minimum data
            print(f"⚠️  Skipping {product_id} - insufficient data")
            continue
        
        # Split data for training and prediction
        train_size = int(len(product_data) * 0.8)
        train_data = product_data.iloc[:train_size]
        test_data = product_data.iloc[train_size:]
        
        # Generate predictions for the test period
        predictions = {}
        
        # ARIMA predictions
        try:
            arima_model = ARIMAModel()
            arima_model.train(train_data)
            arima_pred = arima_model.predict(len(test_data))
            predictions['arima'] = arima_pred
            print(f"   ✅ ARIMA: {len(arima_pred)} predictions")
        except Exception as e:
            print(f"   ❌ ARIMA failed: {e}")
            predictions['arima'] = None
        
        # Prophet predictions
        try:
            prophet_model = ProphetModel()
            prophet_model.train(train_data)
            prophet_pred = prophet_model.predict(len(test_data))
            predictions['prophet'] = prophet_pred
            print(f"   ✅ Prophet: {len(prophet_pred)} predictions")
        except Exception as e:
            print(f"   ❌ Prophet failed: {e}")
            predictions['prophet'] = None
        
        # LSTM predictions
        try:
            lstm_model = LSTMModel()
            lstm_model.train(train_data)
            lstm_pred = lstm_model.predict(len(test_data))
            predictions['lstm'] = lstm_pred
            print(f"   ✅ LSTM: {len(lstm_pred)} predictions")
        except Exception as e:
            print(f"   ❌ LSTM failed: {e}")
            predictions['lstm'] = None
        
        # Create prediction dataframe
        for model_name, pred in predictions.items():
            if pred is not None and len(pred) > 0:
                # Ensure we have the right number of predictions
                pred_length = min(len(pred), len(test_data))
                
                for i in range(pred_length):
                    prediction_record = {
                        'date': test_data.iloc[i]['date'],
                        'product_id': product_id,
                        'model': model_name,
                        'predicted_sales': float(pred[i]) if isinstance(pred[i], (np.ndarray, list)) else float(pred[i]),
                        'actual_sales': float(test_data.iloc[i]['sales']),
                        'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    all_predictions.append(prediction_record)
    
    # Create predictions dataframe
    if all_predictions:
        predictions_df = pd.DataFrame(all_predictions)
        
        # Save predictions in multiple formats for dashboard
        predictions_dir = "predictions"
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(predictions_dir, "model_predictions.csv")
        predictions_df.to_csv(csv_path, index=False)
        
        # Save as JSON for easy web consumption
        json_path = os.path.join(predictions_dir, "model_predictions.json")
        predictions_df.to_json(json_path, orient='records', date_format='iso')
        
        # Save summary statistics
        summary_stats = {}
        for model in predictions_df['model'].unique():
            model_data = predictions_df[predictions_df['model'] == model]
            mae = np.mean(np.abs(model_data['actual_sales'] - model_data['predicted_sales']))
            mape = np.mean(np.abs((model_data['actual_sales'] - model_data['predicted_sales']) / model_data['actual_sales'])) * 100
            
            summary_stats[model] = {
                'mae': float(mae),
                'mape': float(mape),
                'predictions_count': len(model_data)
            }
        
        # Save summary stats
        summary_path = os.path.join(predictions_dir, "prediction_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"\n✅ Predictions generated successfully!")
        print(f"📊 Total predictions: {len(predictions_df)}")
        print(f"📁 Saved to: {predictions_dir}/")
        print(f"📈 Models: {', '.join(predictions_df['model'].unique())}")
        
        # Print summary
        print(f"\n📊 Prediction Summary:")
        for model, stats in summary_stats.items():
            print(f"   {model.upper()}: MAE={stats['mae']:.2f}, MAPE={stats['mape']:.1f}%, Count={stats['predictions_count']}")
        
        return predictions_df
    else:
        print("❌ No predictions generated")
        return None

def create_sample_predictions():
    """Create sample predictions for demonstration"""
    print("🎭 Creating sample predictions for dashboard demo...")
    
    # Load data
    df = load_data(processed=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Create sample predictions for the last 30 days
    last_date = df['date'].max()
    sample_dates = pd.date_range(start=last_date - timedelta(days=29), end=last_date, freq='D')
    
    sample_predictions = []
    product_categories = df['product_id'].unique()
    
    for date in sample_dates:
        for product_id in product_categories:
            # Get actual sales for this date/product
            actual_data = df[(df['date'] == date) & (df['product_id'] == product_id)]
            actual_sales = actual_data['sales'].iloc[0] if not actual_data.empty else 100
            
            # Create realistic predictions with some noise
            for model in ['arima', 'prophet', 'lstm']:
                # Add some realistic variation
                noise = np.random.normal(0, actual_sales * 0.1)  # 10% noise
                predicted_sales = max(0, actual_sales + noise)
                
                sample_predictions.append({
                    'date': date,
                    'product_id': product_id,
                    'model': model,
                    'predicted_sales': round(predicted_sales, 2),
                    'actual_sales': round(actual_sales, 2),
                    'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
    
    # Save sample predictions
    predictions_dir = "predictions"
    os.makedirs(predictions_dir, exist_ok=True)
    
    sample_df = pd.DataFrame(sample_predictions)
    sample_df.to_csv(os.path.join(predictions_dir, "sample_predictions.csv"), index=False)
    sample_df.to_json(os.path.join(predictions_dir, "sample_predictions.json"), orient='records', date_format='iso')
    
    print(f"✅ Sample predictions created: {len(sample_predictions)} records")
    return sample_df

if __name__ == "__main__":
    try:
        # Try to generate real predictions first
        predictions_df = generate_predictions()
        
        if predictions_df is None or len(predictions_df) == 0:
            print("⚠️  Real predictions failed, creating sample predictions...")
            create_sample_predictions()
        else:
            print("🎉 Real predictions generated successfully!")
            
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        print("🎭 Creating sample predictions instead...")
        create_sample_predictions() 