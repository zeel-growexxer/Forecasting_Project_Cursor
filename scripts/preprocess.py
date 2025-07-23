import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.data.loader import load_config
from src.data.utils import preprocess_sales_data

config = load_config()
raw_path = config['data']['raw_path']
processed_path = config['data']['processed_path']

def main():
    df = pd.read_csv(raw_path)
    processed, feature_cols = preprocess_sales_data(
        df,
        date_col=config['data'].get('date_col', 'date'),
        product_col=config['data'].get('product_col', 'product_category'),
        sales_col=config['data'].get('sales_col', 'total_amount'),
        fill_method='zero',
        add_time_features=True,
        outlier_clip='iqr',
        add_lag_features=True,
        scale_features=True,
        scaler_path='models/lstm/feature_scaler.joblib',
        feature_cols_path='models/lstm/feature_cols.joblib',
        verbose=True
    )
    processed.to_csv(processed_path, index=False)
    print(f'Processed data saved to {processed_path}')

if __name__ == '__main__':
    main() 