import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List
import os
import sys

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Forecasting Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Try to import optional dependencies
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from src.data.loader import load_config, load_data
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    DATA_LOADER_AVAILABLE = False
    st.error(f"Data loader not available: {e}")

try:
    from src.models.model_selector import model_selector
    MODEL_SELECTOR_AVAILABLE = True
except ImportError:
    MODEL_SELECTOR_AVAILABLE = False

try:
    from src.notifications.alert_manager import alert_manager
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False

# Show warnings for missing dependencies
if not MLFLOW_AVAILABLE:
    st.warning("MLflow not available. Some features will be disabled.")
if not MODEL_SELECTOR_AVAILABLE:
    st.warning("Model selector not available. Some features will be disabled.")
if not NOTIFICATIONS_AVAILABLE:
    st.warning("Notifications not available. Some features will be disabled.")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success { color: #28a745; }
    .status-warning { color: #ffc107; }
    .status-error { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class SimpleDashboard:
    def __init__(self):
        self.config = None
        self.experiment = None
        if DATA_LOADER_AVAILABLE:
            try:
                self.config = load_config()
                if MLFLOW_AVAILABLE:
                    self.setup_mlflow()
            except Exception as e:
                st.error(f"Failed to initialize dashboard: {e}")
    
    def setup_mlflow(self):
        """Setup MLflow connection"""
        try:
            if self.config and 'mlflow' in self.config:
                mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
                self.experiment = mlflow.get_experiment_by_name(self.config['mlflow']['experiment_name'])
        except Exception as e:
            st.warning(f"Failed to connect to MLflow: {e}")
            self.experiment = None
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 Forecasting Pipeline Dashboard</h1>', unsafe_allow_html=True)
        
        # Last updated timestamp
        st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    def render_sales_data(self):
        """Render sales data by date and product category"""
        st.header("📈 Sales Data Analysis")
        
        if not DATA_LOADER_AVAILABLE:
            st.error("Data loader not available. Cannot display sales data.")
            return
        
        try:
            # Load processed data
            df = load_data(processed=True)
            
            if df.empty:
                st.warning("No sales data available")
                return
            
            # Convert date column to datetime if needed
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Fix day_of_week to show actual day names
            df = self.fix_day_of_week(df)
            
            self.render_sales_data_with_df(df)
                
        except Exception as e:
            st.error(f"Error loading sales data: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def fix_day_of_week(self, df):
        """Convert decimal day_of_week to actual day names"""
        # Map decimal values to day names with proper rounding
        def map_day_name(day_value):
            # Round to handle floating point precision issues
            rounded = round(day_value, 6)
            if rounded == 0.0:
                return 'Monday'
            elif rounded == 0.166667:
                return 'Tuesday'
            elif rounded == 0.333333:
                return 'Wednesday'
            elif rounded == 0.5:
                return 'Thursday'
            elif rounded == 0.666667:
                return 'Friday'
            elif rounded == 0.833333:
                return 'Saturday'
            elif rounded == 1.0:
                return 'Sunday'
            else:
                # Fallback: calculate day number and map
                day_num = int(round(day_value * 6))
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                return days[day_num] if 0 <= day_num < 7 else 'Unknown'
        
        # Create a new column with actual day names
        df['day_name'] = df['day_of_week'].apply(map_day_name)
        
        # Also create a proper day_of_week number (0-6) for sorting
        df['day_number'] = (df['day_of_week'] * 6).round().astype(int)
        
        return df
    
    def render_sales_data_with_df(self, df):
        """Render sales data with the provided dataframe"""
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales Overview", "📅 Time Series", "🏷️ Product Analysis", "📋 Data Table"])
        
        with tab1:
            self.render_sales_overview(df)
        
        with tab2:
            self.render_time_series(df)
        
        with tab3:
            self.render_product_analysis(df)
        
        with tab4:
            self.render_data_table(df)
    
    def render_sales_overview(self, df):
        """Render sales overview metrics and charts"""
        st.subheader("Sales Overview")
        
        # Calculate key metrics
        total_sales = df['sales'].sum()
        avg_daily_sales = df.groupby('date')['sales'].sum().mean()
        total_categories = df['product_id'].nunique()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sales", f"${total_sales:,.2f}")
        with col2:
            st.metric("Avg Daily Sales", f"${avg_daily_sales:,.2f}")
        with col3:
            st.metric("Product Categories", total_categories)
        with col4:
            # Display date range with more space
            start_date = df['date'].min().strftime('%Y-%m-%d')
            end_date = df['date'].max().strftime('%Y-%m-%d')
            st.write("**Date Range:**")
            st.write(f"{start_date}")
            st.write(f"to {end_date}")
        
        # Sales trend chart with better styling
        daily_sales = df.groupby('date')['sales'].sum().reset_index()
        fig = px.line(daily_sales, x='date', y='sales', 
                     title="📈 Daily Sales Trend",
                     labels={'sales': 'Sales Amount ($)', 'date': 'Date'},
                     line_shape='linear',
                     render_mode='svg')
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
                tickformat=',',  # Add commas to numbers
                tickprefix='$'
            ),
            font=dict(size=12)  # Increase font size
        )
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by product category
        category_sales = df.groupby('product_id')['sales'].sum().sort_values(ascending=False)
        fig = px.bar(x=category_sales.values, y=category_sales.index,
                    title="🏷️ Sales by Product Category",
                    labels={'x': 'Sales Amount ($)', 'y': 'Product Category'},
                    color=category_sales.values,
                    color_continuous_scale='viridis')
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            margin=dict(l=60, r=40, t=60, b=60),  # Add margins for labels
            xaxis=dict(
                tickformat=',',
                tickprefix='$',
                tickangle=0
            ),
            yaxis=dict(
                tickmode='array',
                ticktext=category_sales.index,
                tickvals=list(range(len(category_sales)))
            ),
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by day of week
        day_sales = df.groupby('day_name')['sales'].sum().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        fig = px.bar(x=day_sales.index, y=day_sales.values,
                    title="📅 Sales by Day of Week",
                    labels={'x': 'Day of Week', 'y': 'Sales Amount ($)'},
                    color=day_sales.values,
                    color_continuous_scale='plasma')
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            margin=dict(l=60, r=40, t=60, b=60),  # Add margins for labels
            xaxis=dict(
                tickangle=0,
                tickmode='array',
                ticktext=day_sales.index,
                tickvals=list(range(len(day_sales)))
            ),
            yaxis=dict(
                tickformat=',',
                tickprefix='$'
            ),
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_time_series(self, df):
        """Render time series analysis"""
        st.subheader("📅 Time Series Analysis")
        
        # Product category dropdown filter
        st.write("**Filter by Product Category:**")
        categories = sorted(df['product_id'].unique())
        selected_categories = st.multiselect(
            "Select Product Categories", 
            categories, 
            default=categories,
            help="Choose which product categories to display in the charts"
        )
        
        if not selected_categories:
            st.warning("Please select at least one product category")
            return
        
        # Filter data by selected categories
        filtered_df = df[df['product_id'].isin(selected_categories)]
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=filtered_df['date'].min().date())
        with col2:
            end_date = st.date_input("End Date", value=filtered_df['date'].max().date())
        
        # Filter data by date range
        mask = (filtered_df['date'].dt.date >= start_date) & (filtered_df['date'].dt.date <= end_date)
        date_filtered_df = filtered_df[mask]
        
        if date_filtered_df.empty:
            st.warning("No data available for selected date range")
            return
        
        # 1. Overall Sales Trend Line Chart
        st.subheader("📈 Overall Sales Trend")
        daily_sales = date_filtered_df.groupby('date')['sales'].sum().reset_index()
        fig = px.line(daily_sales, x='date', y='sales',
                     title="Daily Total Sales Trend",
                     labels={'sales': 'Sales Amount ($)', 'date': 'Date'},
                     line_shape='linear',
                     render_mode='svg')
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
        fig.update_traces(line=dict(width=3, color='#1f77b4'))
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Sales by Category Line Chart
        st.subheader("🏷️ Sales by Product Category")
        category_sales = date_filtered_df.groupby(['date', 'product_id'])['sales'].sum().reset_index()
        fig = px.line(category_sales, x='date', y='sales', color='product_id',
                     title="Sales Trend by Product Category",
                     labels={'sales': 'Sales Amount ($)', 'date': 'Date', 'product_id': 'Product Category'},
                     line_shape='linear',
                     render_mode='svg')
        fig.update_layout(
            height=500,
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
        fig.update_traces(line=dict(width=2))
        st.plotly_chart(fig, use_container_width=True)
    
    def render_product_analysis(self, df):
        """Render product category analysis"""
        st.subheader("🏷️ Product Category Analysis")
        
        # Product performance metrics
        product_metrics = df.groupby('product_id').agg({
            'sales': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'date': ['min', 'max']
        }).round(2)
        
        product_metrics.columns = ['Total Sales', 'Avg Sales', 'Std Dev', 'Min Sales', 'Max Sales', 'Data Points', 'First Sale', 'Last Sale']
        product_metrics = product_metrics.sort_values('Total Sales', ascending=False)
        
        # Format date columns to prevent cutoff
        product_metrics['First Sale'] = product_metrics['First Sale'].dt.strftime('%Y-%m-%d')
        product_metrics['Last Sale'] = product_metrics['Last Sale'].dt.strftime('%Y-%m-%d')
        
        # Display product metrics
        st.write("**📊 Product Category Performance:**")
        st.dataframe(product_metrics, use_container_width=True)
        
        # Product comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Total sales by category
            fig = px.pie(values=product_metrics['Total Sales'], 
                        names=product_metrics.index,
                        title="📊 Sales Distribution by Category")
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average daily sales by category
            avg_sales = product_metrics['Avg Sales'].sort_values(ascending=True)
            fig = px.bar(x=avg_sales.values, y=avg_sales.index, orientation='h',
                        title="📈 Average Daily Sales by Category",
                        labels={'x': 'Average Sales ($)', 'y': 'Product Category'})
            fig.update_layout(
                height=400, 
                plot_bgcolor='white', 
                paper_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20),  # Add margins to prevent cutoff
                yaxis=dict(tickmode='array', ticktext=avg_sales.index, tickvals=list(range(len(avg_sales))))
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_data_table(self, df):
        """Render interactive data table"""
        st.subheader("📋 Sales Data Table")
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("Date Range", 
                                     value=(df['date'].min().date(), df['date'].max().date()),
                                     key="date_range_filter")
        with col2:
            categories = st.multiselect("Filter by Product Categories", 
                                      sorted(df['product_id'].unique()),
                                      default=sorted(df['product_id'].unique()),
                                      key="category_filter")
        
        # Apply filters
        filtered_df = df.copy()
        if len(date_range) == 2:
            mask = (df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])
            filtered_df = filtered_df[mask]
        
        if categories:
            filtered_df = filtered_df[filtered_df['product_id'].isin(categories)]
        
        # Select meaningful columns for display
        display_columns = ['date', 'product_id', 'sales', 'day_name', 'month', 'is_weekend']
        display_df = filtered_df[display_columns].copy()
        
        # Rename columns for better display
        display_df.columns = ['Date', 'Product Category', 'Sales ($)', 'Day of Week', 'Month', 'Weekend']
        
        # Fix month names
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        display_df['Month'] = display_df['Month'].apply(lambda x: month_names.get(int(x), f'Month {int(x)}'))
        
        # Format the data for better readability
        display_df['Sales ($)'] = display_df['Sales ($)'].apply(lambda x: f"${x:,.2f}")
        display_df['Weekend'] = display_df['Weekend'].apply(lambda x: "Yes" if x == 1.0 else "No")
        
        # Display data
        st.write(f"**Showing {len(display_df)} records**")
        st.dataframe(display_df, use_container_width=True)
        
        # Summary statistics for filtered data
        if not filtered_df.empty:
            st.subheader("📊 Filtered Data Summary")
            summary_stats = filtered_df.groupby('product_id').agg({
                'sales': ['sum', 'mean', 'count']
            }).round(2)
            summary_stats.columns = ['Total Sales', 'Average Sales', 'Records']
            summary_stats = summary_stats.sort_values('Total Sales', ascending=False)
            st.dataframe(summary_stats, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def render_model_performance(self):
        """Render model performance section"""
        st.header("📊 Model Performance")
        
        if not MLFLOW_AVAILABLE:
            st.warning("MLflow not available. Model performance data cannot be displayed.")
            return
        
        if not self.experiment:
            st.warning("No MLflow experiment available. Train some models first.")
            return
        
        st.info("Model performance tracking requires trained models. Use the training scripts to generate model data.")
    
    def render_system_status(self):
        """Render system status"""
        st.header("🔧 System Status")
        
        # Check various system components
        health_checks = {
            "Data Directory": os.path.exists("data"),
            "Models Directory": os.path.exists("models"),
            "Config File": os.path.exists("config.ini"),
            "Processed Data": os.path.exists("data/processed/processed_retail_sales_data.csv")
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Health")
            for check, status in health_checks.items():
                icon = "✅" if status else "❌"
                color = "status-success" if status else "status-error"
                st.markdown(f'<span class="{color}">{icon} {check}</span>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Recent Activity")
            models_dir = "models"
            if os.path.exists(models_dir):
                recent_files = []
                for root, dirs, files in os.walk(models_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if mtime > datetime.now() - timedelta(hours=24):
                            recent_files.append({
                                'file': file,
                                'path': os.path.relpath(file_path, models_dir),
                                'modified': mtime
                            })
                
                if recent_files:
                    recent_files.sort(key=lambda x: x['modified'], reverse=True)
                    for file_info in recent_files[:5]:
                        st.write(f"📄 {file_info['path']} ({file_info['modified'].strftime('%H:%M')})")
                else:
                    st.write("No recent activity")
            else:
                st.write("Models directory not found")
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Create main tabs
        tab1, tab2, tab3 = st.tabs(["📈 Sales Data", "📊 Model Performance", "🔧 System Status"])
        
        with tab1:
            # Sales data content
            self.render_sales_data()
        
        with tab2:
            # Model performance content
            self.render_model_performance()
        
        with tab3:
            # System status content
            self.render_system_status()

# Create and run the dashboard
if __name__ == "__main__":
    dashboard = SimpleDashboard()
    dashboard.run() 