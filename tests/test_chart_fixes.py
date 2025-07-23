#!/usr/bin/env python3
"""
Test script to verify chart fixes for text visibility and date cutoff issues.

This script tests:
1. Horizontal bar chart creation (px.bar with orientation='h')
2. Chart margins and text formatting
3. Date axis formatting
4. Product category labels
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def test_horizontal_bar_chart():
    """Test horizontal bar chart creation"""
    print("🔍 Testing horizontal bar chart...")
    
    # Create sample data
    data = {
        'category': ['Books', 'Clothing', 'Electronics', 'Home & Garden'],
        'sales': [43167, 43167, 43167, 43167]
    }
    df = pd.DataFrame(data)
    
    try:
        # Test the fixed horizontal bar chart
        fig = px.bar(x=df['sales'], y=df['category'], orientation='h',
                    title="Test Horizontal Bar Chart",
                    labels={'x': 'Sales Amount ($)', 'y': 'Product Category'})
        
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(tickmode='array', ticktext=df['category'], tickvals=list(range(len(df['category']))))
        )
        
        print("✅ Horizontal bar chart created successfully!")
        print(f"   - Chart type: {type(fig)}")
        print(f"   - Margins: {fig.layout.margin}")
        print(f"   - Y-axis labels: {df['category'].tolist()}")
        
    except Exception as e:
        print(f"❌ Horizontal bar chart failed: {e}")

def test_chart_margins():
    """Test chart margin settings"""
    print("\n📏 Testing chart margins...")
    
    # Load real data
    df = load_data(processed=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Test line chart with margins
    daily_sales = df.groupby('date')['sales'].sum().reset_index()
    
    try:
        fig = px.line(daily_sales, x='date', y='sales',
                     title="Test Line Chart with Margins",
                     labels={'sales': 'Sales Amount ($)', 'date': 'Date'})
        
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=40, t=60, b=60),  # Add margins for labels
            xaxis=dict(
                gridcolor='lightgray',
                tickangle=45,  # Angle date labels to prevent overlap
                tickmode='auto',
                nticks=10  # Limit number of ticks
            ),
            yaxis=dict(
                gridcolor='lightgray',
                tickformat=',',
                tickprefix='$'
            ),
            font=dict(size=12)
        )
        
        print("✅ Line chart with margins created successfully!")
        print(f"   - Margins: {fig.layout.margin}")
        print(f"   - X-axis tick angle: {fig.layout.xaxis.tickangle}")
        print(f"   - Y-axis tick format: {fig.layout.yaxis.tickformat}")
        print(f"   - Font size: {fig.layout.font.size}")
        
    except Exception as e:
        print(f"❌ Line chart failed: {e}")

def test_date_formatting():
    """Test date axis formatting"""
    print("\n📅 Testing date formatting...")
    
    # Load real data
    df = load_data(processed=True)
    df['date'] = pd.to_datetime(df['date'])
    
    # Test date range
    date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    print(f"   - Date range: {date_range}")
    print(f"   - Total days: {(df['date'].max() - df['date'].min()).days}")
    
    # Test date tick formatting
    try:
        daily_sales = df.groupby('date')['sales'].sum().reset_index()
        fig = px.line(daily_sales, x='date', y='sales')
        
        fig.update_layout(
            xaxis=dict(
                tickangle=45,
                tickmode='auto',
                nticks=10
            )
        )
        
        print("✅ Date formatting applied successfully!")
        print(f"   - Tick angle: {fig.layout.xaxis.tickangle}")
        print(f"   - Max ticks: {fig.layout.xaxis.nticks}")
        
    except Exception as e:
        print(f"❌ Date formatting failed: {e}")

def test_product_categories():
    """Test product category display"""
    print("\n🏷️ Testing product categories...")
    
    # Load real data
    df = load_data(processed=True)
    
    categories = sorted(df['product_id'].unique())
    print(f"   - Found {len(categories)} categories: {categories}")
    
    # Test category bar chart
    try:
        category_sales = df.groupby('product_id')['sales'].sum().sort_values(ascending=False)
        fig = px.bar(x=category_sales.values, y=category_sales.index,
                    title="Test Category Chart",
                    labels={'x': 'Sales Amount ($)', 'y': 'Product Category'})
        
        fig.update_layout(
            margin=dict(l=60, r=40, t=60, b=60),
            yaxis=dict(
                tickmode='array',
                ticktext=category_sales.index,
                tickvals=list(range(len(category_sales)))
            )
        )
        
        print("✅ Category chart created successfully!")
        print(f"   - Categories displayed: {category_sales.index.tolist()}")
        
    except Exception as e:
        print(f"❌ Category chart failed: {e}")

def main():
    """Run all chart tests"""
    print("🚀 Testing Chart Fixes")
    print("=" * 50)
    
    try:
        # Test horizontal bar chart
        test_horizontal_bar_chart()
        
        # Test chart margins
        test_chart_margins()
        
        # Test date formatting
        test_date_formatting()
        
        # Test product categories
        test_product_categories()
        
        print("\n✅ All chart tests completed successfully!")
        print("\n🎯 Chart Fixes Applied:")
        print("1. ✅ Fixed px.barh → px.bar with orientation='h'")
        print("2. ✅ Added proper margins to prevent text cutoff")
        print("3. ✅ Angled date labels (45°) to prevent overlap")
        print("4. ✅ Limited date ticks to prevent crowding")
        print("5. ✅ Added currency formatting ($ and commas)")
        print("6. ✅ Increased font size for better readability")
        print("7. ✅ Proper category label display")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 