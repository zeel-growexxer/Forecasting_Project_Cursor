#!/usr/bin/env python3
"""
Generate realistic sales data with different patterns for each product category.

This script creates sales data where each product category has:
- Different base sales levels
- Different seasonal patterns
- Different day-of-week effects
- Realistic variability
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_realistic_sales_data(start_date='2024-01-01', end_date='2024-12-31'):
    """
    Generate realistic sales data with different patterns for each product category.
    
    Args:
        start_date (str): Start date for the data
        end_date (str): End date for the data
        
    Returns:
        pd.DataFrame: Sales data with realistic patterns
    """
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define product categories with different characteristics
    products = {
        'Books': {
            'base_sales': 80,
            'seasonal_boost': 1.3,  # Higher sales during holidays
            'weekend_boost': 1.2,   # Higher sales on weekends
            'volatility': 0.15      # 15% standard deviation
        },
        'Clothing': {
            'base_sales': 120,
            'seasonal_boost': 1.5,  # Strong seasonal patterns
            'weekend_boost': 1.4,   # Much higher on weekends
            'volatility': 0.25      # 25% standard deviation
        },
        'Electronics': {
            'base_sales': 150,
            'seasonal_boost': 1.8,  # Very strong seasonal patterns
            'weekend_boost': 1.3,   # Higher on weekends
            'volatility': 0.30      # 30% standard deviation
        },
        'Home & Garden': {
            'base_sales': 90,
            'seasonal_boost': 1.6,  # Strong spring/summer boost
            'weekend_boost': 1.5,   # Much higher on weekends
            'volatility': 0.20      # 20% standard deviation
        }
    }
    
    # Generate data for each product
    all_data = []
    
    for product_name, config in products.items():
        print(f"Generating data for {product_name}...")
        
        for date in dates:
            # Base sales
            base = config['base_sales']
            
            # Day of week effect (0=Monday, 6=Sunday)
            day_of_week = date.weekday()
            if day_of_week >= 5:  # Weekend
                day_multiplier = config['weekend_boost']
            else:
                day_multiplier = 1.0
            
            # Seasonal effect (higher in certain months)
            month = date.month
            if product_name == 'Books':
                # Books: Higher in December (holidays), September (back to school)
                if month in [12, 9]:
                    seasonal_multiplier = config['seasonal_boost']
                else:
                    seasonal_multiplier = 1.0
            elif product_name == 'Clothing':
                # Clothing: Higher in December, March (spring), September (fall)
                if month in [12, 3, 9]:
                    seasonal_multiplier = config['seasonal_boost']
                else:
                    seasonal_multiplier = 1.0
            elif product_name == 'Electronics':
                # Electronics: Higher in December, November (Black Friday)
                if month in [12, 11]:
                    seasonal_multiplier = config['seasonal_boost']
                else:
                    seasonal_multiplier = 1.0
            elif product_name == 'Home & Garden':
                # Home & Garden: Higher in spring/summer months
                if month in [4, 5, 6, 7, 8]:
                    seasonal_multiplier = config['seasonal_boost']
                else:
                    seasonal_multiplier = 1.0
            
            # Calculate sales with all effects
            sales = base * day_multiplier * seasonal_multiplier
            
            # Add random variability
            noise = np.random.normal(0, config['volatility'] * sales)
            sales += noise
            
            # Ensure sales are positive
            sales = max(0, sales)
            
            # Round to 2 decimal places
            sales = round(sales, 2)
            
            # Generate unique product ID
            product_id = f"{product_name[:3].upper()}{len(all_data):03d}"
            
            all_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Product Category': product_name,
                'Total Amount': sales,
                'Product ID': product_id
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    return df

def main():
    """Main function to generate and save realistic sales data"""
    print("🚀 Generating Realistic Sales Data")
    print("=" * 50)
    
    # Generate data
    df = generate_realistic_sales_data()
    
    # Show summary statistics
    print(f"\n📊 Generated {len(df)} records")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Show sales by product category
    print("\n💰 Sales Summary by Product Category:")
    summary = df.groupby('Product Category')['Total Amount'].agg(['sum', 'mean', 'count']).round(2)
    summary.columns = ['Total Sales', 'Average Sales', 'Records']
    print(summary)
    
    # Show day-of-week patterns
    print("\n📅 Day-of-Week Sales Patterns:")
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.day_name()
    day_patterns = df.groupby(['Product Category', 'DayOfWeek'])['Total Amount'].mean().round(2)
    print(day_patterns)
    
    # Show seasonal patterns
    print("\n🌤️ Seasonal Sales Patterns (by month):")
    df['Month'] = df['Date'].dt.month
    seasonal_patterns = df.groupby(['Product Category', 'Month'])['Total Amount'].mean().round(2)
    print(seasonal_patterns)
    
    # Save to file
    output_path = 'data/raw/retail_sales_dataset.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n✅ Data saved to: {output_path}")
    print(f"📁 File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    
    # Verify the data is different
    print("\n🔍 Verifying Data Differences:")
    for product in df['Product Category'].unique():
        product_data = df[df['Product Category'] == product]
        total_sales = product_data['Total Amount'].sum()
        avg_sales = product_data['Total Amount'].mean()
        print(f"🏷️ {product}: ${total_sales:,.2f} total, ${avg_sales:.2f} average")
    
    print("\n🎉 Realistic sales data generated successfully!")
    print("Each product category now has different sales patterns:")
    print("- 📚 Books: Moderate sales, higher during holidays and back-to-school")
    print("- 👕 Clothing: Higher sales, strong seasonal patterns, weekend boost")
    print("- 📱 Electronics: Highest sales, very strong holiday season")
    print("- 🏡 Home & Garden: Moderate sales, strong spring/summer season")

if __name__ == "__main__":
    main() 