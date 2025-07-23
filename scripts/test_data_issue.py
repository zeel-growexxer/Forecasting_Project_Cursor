#!/usr/bin/env python3
"""
Test script to investigate why all product categories show the same sales values.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_data
import pandas as pd

def test_data_structure():
    """Test the data structure to understand the issue"""
    print("🔍 Investigating Data Structure Issue")
    print("=" * 50)
    
    # Load data
    df = load_data(processed=True)
    print(f"📊 Total records: {len(df)}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏷️ Product categories: {sorted(df['product_id'].unique())}")
    
    # Check for duplicate data
    print("\n🔍 Checking for data duplication...")
    
    # Group by product and date to see if there are duplicates
    daily_sales = df.groupby(['date', 'product_id'])['sales'].sum().reset_index()
    print(f"📈 Unique date-product combinations: {len(daily_sales)}")
    print(f"📊 Original records: {len(df)}")
    
    if len(daily_sales) != len(df):
        print("⚠️  WARNING: Data has duplicate date-product combinations!")
        
        # Check what's causing duplicates
        duplicates = df.groupby(['date', 'product_id']).size().reset_index(name='count')
        duplicates = duplicates[duplicates['count'] > 1]
        print(f"🔍 Found {len(duplicates)} duplicate combinations:")
        print(duplicates.head())
    
    # Check sales distribution by product
    print("\n📊 Sales by Product Category (Original Data):")
    product_sales = df.groupby('product_id').agg({
        'sales': ['sum', 'mean', 'count']
    }).round(2)
    product_sales.columns = ['Total Sales', 'Average Sales', 'Records']
    print(product_sales)
    
    # Check if all products have the same number of records
    record_counts = df.groupby('product_id').size()
    print(f"\n📋 Records per product: {record_counts.to_dict()}")
    
    # Check if all products have the same total sales
    total_sales = df.groupby('product_id')['sales'].sum()
    print(f"\n💰 Total sales per product: {total_sales.to_dict()}")
    
    # Check if the issue is in the data itself or in the dashboard calculation
    print("\n🔍 Testing Dashboard Calculation Logic...")
    
    # Simulate the dashboard filtering logic
    all_categories = sorted(df['product_id'].unique())
    selected_categories = all_categories  # All categories selected
    
    # Apply filter (like dashboard does)
    df_filtered = df[df['product_id'].isin(selected_categories)].copy()
    
    print(f"📊 Original data shape: {df.shape}")
    print(f"📊 Filtered data shape: {df_filtered.shape}")
    
    # Calculate summary stats on filtered data (like dashboard does)
    summary_stats = df_filtered.groupby('product_id').agg({
        'sales': ['sum', 'mean', 'count']
    }).round(2)
    summary_stats.columns = ['Total Sales', 'Average Sales', 'Records']
    summary_stats = summary_stats.sort_values('Total Sales', ascending=False)
    
    print("\n📊 Summary Stats (Filtered Data - Dashboard Logic):")
    print(summary_stats)
    
    # Check if the issue is that all products have identical data
    print("\n🔍 Checking for identical data across products...")
    
    # Get unique sales values for each product
    for product in df['product_id'].unique():
        product_data = df[df['product_id'] == product]
        unique_sales = product_data['sales'].unique()
        print(f"🏷️ {product}: {len(unique_sales)} unique sales values")
        if len(unique_sales) <= 5:  # Show first few if small number
            print(f"   Values: {sorted(unique_sales)[:5]}")
    
    # Check if all products have the same daily sales pattern
    print("\n🔍 Checking daily sales patterns...")
    daily_by_product = df.groupby(['date', 'product_id'])['sales'].sum().reset_index()
    
    # Pivot to compare products
    pivot_data = daily_by_product.pivot(index='date', columns='product_id', values='sales')
    print(f"📊 Pivot shape: {pivot_data.shape}")
    
    # Check if all columns are identical
    columns = pivot_data.columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            if pivot_data[columns[i]].equals(pivot_data[columns[j]]):
                print(f"⚠️  WARNING: {columns[i]} and {columns[j]} have identical sales data!")
    
    # Show sample of pivot data
    print("\n📊 Sample of daily sales by product (first 10 days):")
    print(pivot_data.head(10))
    
    return df, product_sales

def test_data_generation():
    """Test if the data generation is creating identical data for all products"""
    print("\n🔍 Testing Data Generation Process...")
    print("=" * 50)
    
    # Load raw data to see original structure
    try:
        from src.data.loader import load_config
        config = load_config()
        raw_path = config['data']['raw_path']
        raw_df = pd.read_csv(raw_path)
        
        print(f"📊 Raw data shape: {raw_df.shape}")
        print(f"📊 Raw data columns: {raw_df.columns.tolist()}")
        
        # Check original product distribution
        if 'Product Category' in raw_df.columns:
            product_counts = raw_df['Product Category'].value_counts()
            print(f"\n🏷️ Original product distribution:")
            print(product_counts)
        
        # Check if raw data has different values per product
        if 'Total Amount' in raw_df.columns and 'Product Category' in raw_df.columns:
            raw_sales = raw_df.groupby('Product Category')['Total Amount'].agg(['sum', 'mean', 'count'])
            print(f"\n💰 Raw sales by product:")
            print(raw_sales)
    
    except Exception as e:
        print(f"❌ Error loading raw data: {e}")

def main():
    """Main test function"""
    print("🚀 Testing Data Structure Issue")
    print("=" * 50)
    
    try:
        # Test data structure
        df, product_sales = test_data_structure()
        
        # Test data generation
        test_data_generation()
        
        print("\n✅ Data investigation completed!")
        print("\n🎯 Potential Issues Found:")
        print("1. Check if data preprocessing is creating identical data for all products")
        print("2. Check if the raw data has different values per product")
        print("3. Check if the dashboard filtering logic is working correctly")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 