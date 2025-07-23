#!/usr/bin/env python3
"""
Demo script for the enhanced sales dashboard.

This script demonstrates the new sales data visualization features
that show actual sales amounts by date and product category, making
the dashboard much more practical and informative than just MAE metrics.

Key Features Demonstrated:
- Sales amounts by date and product category
- Interactive time series analysis
- Product performance comparison
- Forecast vs actual comparisons
- Downloadable data tables
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_data
import pandas as pd
import streamlit as st

def main():
    """Demo the sales dashboard features"""
    st.title("🛍️ Sales Dashboard Demo")
    st.write("""
    This demo showcases the enhanced dashboard features that display actual sales amounts
    by date and product category, making it much more practical than just MAE metrics.
    """)
    
    # Load sample data
    try:
        df = load_data(processed=True)
        
        if df.empty:
            st.error("No data available. Please run preprocessing first.")
            return
        
        st.success(f"✅ Loaded {len(df)} sales records")
        
        # Show data overview
        st.header("📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sales", f"${df['sales'].sum():,.2f}")
        with col2:
            st.metric("Products", df['product_id'].nunique())
        with col3:
            st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        with col4:
            st.metric("Avg Daily Sales", f"${df.groupby('date')['sales'].sum().mean():,.2f}")
        
        # Show sample data
        st.header("📋 Sample Data")
        st.write("Here's what the sales data looks like:")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show product breakdown
        st.header("🏷️ Product Sales Breakdown")
        product_sales = df.groupby('product_id')['sales'].sum().sort_values(ascending=False)
        st.bar_chart(product_sales)
        
        st.write("""
        ## 🎯 Key Benefits of Sales Amount Display:
        
        1. **📈 Actual Business Impact**: See real sales amounts, not just error metrics
        2. **📅 Time-based Analysis**: Track sales trends over time
        3. **🏷️ Product Performance**: Compare sales across product categories
        4. **🔮 Forecast Comparison**: Compare predicted vs actual sales amounts
        5. **📊 Interactive Filtering**: Filter by date ranges and products
        6. **📥 Data Export**: Download filtered data for further analysis
        
        ## 🚀 Next Steps:
        
        Run the full dashboard to see all these features in action:
        ```bash
        python scripts/run_dashboard.py
        ```
        
        Then navigate to the "📈 Sales Data" tab to explore:
        - Sales Overview with key metrics
        - Time Series Analysis with date filters
        - Product Analysis with performance metrics
        - Forecast Comparison (when models are trained)
        - Interactive Data Table with filtering
        """)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.write("Please ensure you have run the preprocessing step first.")

if __name__ == "__main__":
    main() 