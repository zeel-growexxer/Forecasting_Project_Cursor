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
    st.warning("MLflow not available. Some features will be disabled.")

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
    st.warning("Model selector not available. Some features will be disabled.")

try:
    from src.notifications.alert_manager import alert_manager
    NOTIFICATIONS_AVAILABLE = True
except ImportError:
    NOTIFICATIONS_AVAILABLE = False
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

class Dashboard:
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
    
    def get_model_performance_data(self, days_back: int = 30) -> pd.DataFrame:
        """Get model performance data from MLflow"""
        if not MLFLOW_AVAILABLE or self.experiment is None:
            return pd.DataFrame()
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Remove timestamp filter to avoid parsing issues
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            
            if runs.empty:
                return pd.DataFrame()
            
            # Extract model name from tags
            runs['model_name'] = runs['tags.model_name'].fillna('unknown')
            
            # Check which columns are available
            available_cols = []
            required_cols = ['run_id', 'start_time', 'end_time', 'model_name', 
                           'metrics.mae', 'metrics.rmse', 'metrics.mape', 'metrics.training_time']
            
            for col in required_cols:
                if col in runs.columns:
                    available_cols.append(col)
                else:
                    # Add missing columns with default values
                    if col.startswith('metrics.'):
                        runs[col] = None
                        available_cols.append(col)
            
            df = runs[available_cols].copy()
            
            # Rename columns
            column_mapping = {
                'run_id': 'run_id',
                'start_time': 'start_time', 
                'end_time': 'end_time',
                'model_name': 'model_name',
                'metrics.mae': 'mae',
                'metrics.rmse': 'rmse', 
                'metrics.mape': 'mape',
                'metrics.training_time': 'training_time'
            }
            
            df.columns = [column_mapping.get(col, col) for col in df.columns]
            
            # Convert timestamps - handle different formats
            try:
                df['start_time'] = pd.to_datetime(df['start_time'])
                df['end_time'] = pd.to_datetime(df['end_time'])
            except Exception as e:
                st.error(f"Error converting timestamps: {e}")
                # Try alternative conversion
                df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
                df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching performance data: {e}")
            return pd.DataFrame()
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown('<h1 class="main-header">📊 Forecasting Pipeline Dashboard</h1>', unsafe_allow_html=True)
        
        # Last updated timestamp
        st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    def render_overview_metrics(self, days_back=30):
        """Render overview metrics"""
        st.header("📈 Overview Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate fresh MAE values and determine best model
        fresh_mae_data = self.calculate_fresh_mae_values(days_back)
        
        with col1:
            if fresh_mae_data:
                # Determine best model based on fresh MAE values
                best_model = None
                best_mae = float('inf')
                
                for model_name, mae in fresh_mae_data.items():
                    if mae is not None and mae < best_mae:
                        best_mae = mae
                        best_model = model_name
                
                if best_model:
                    st.metric("Current Best Model", best_model.upper())
                else:
                    st.metric("Current Best Model", "NONE")
            else:
                st.metric("Current Best Model", "N/A")
        
        with col2:
            # Calculate average MAE from fresh values
            if fresh_mae_data:
                avg_mae = np.mean(list(fresh_mae_data.values()))
                st.metric("Average MAE", f"{avg_mae:.2f}")
            else:
                st.metric("Average MAE", "N/A")
        
        with col3:
            if fresh_mae_data:
                total_models = len(fresh_mae_data)
                st.metric("Active Models", total_models)
            else:
                st.metric("Active Models", 0)
        
        with col4:
            # Check pipeline status
            pipeline_status = self.check_pipeline_status()
            status_color = "🟢" if pipeline_status == "Healthy" else "🔴"
            st.metric("Pipeline Status", f"{status_color} {pipeline_status}")
    
    def calculate_fresh_mae_values(self, days_back=30):
        """Calculate fresh MAE values for all models without printing"""
        try:
            from src.data.loader import load_data
            from scripts.evaluate import evaluate_model
            
            # Load data for the selected time period
            df = load_data(processed=True)
            
            # Filter data by date range - use the most recent data
            df_sorted = df.sort_values('date')
            if len(df_sorted) > days_back:
                df_filtered = df_sorted.tail(days_back)
            else:
                df_filtered = df_sorted  # Use all data if not enough
            
            if len(df_filtered) < 10:  # Need minimum data points
                return None
            
            # Calculate test split for this time period
            test_size = 0.2
            split_idx = int(len(df_filtered) * (1 - test_size))
            train_data = df_filtered.iloc[:split_idx]
            test_data = df_filtered.iloc[split_idx:]
            
            if len(test_data) < 5:  # Need minimum test data
                return None
            
            # Evaluate models and collect MAE values
            mae_data = {}
            
            # ARIMA evaluation
            try:
                arima_mae = evaluate_model('arima', train_data, test_data)
                if arima_mae is not None:
                    mae_data['arima'] = arima_mae
            except Exception as e:
                pass  # Silently handle errors
            
            # Prophet evaluation
            try:
                prophet_mae = evaluate_model('prophet', train_data, test_data)
                if prophet_mae is not None:
                    mae_data['prophet'] = prophet_mae
            except Exception as e:
                pass  # Silently handle errors
            
            # LSTM evaluation
            try:
                lstm_mae = evaluate_model('lstm', train_data, test_data)
                if lstm_mae is not None:
                    mae_data['lstm'] = lstm_mae
            except Exception as e:
                pass  # Silently handle errors
            
            return mae_data
            
        except Exception as e:
            return None

    def calculate_dynamic_mae(self, days_back=30):
        """Calculate MAE dynamically based on selected date range"""
        try:
            from src.data.loader import load_data
            from scripts.evaluate import evaluate_model
            
            # Load data for the selected time period
            df = load_data(processed=True)
            
            # Filter data by date range - use the most recent data
            # Since our data is from 2024, we'll take the last N days from the dataset
            df_sorted = df.sort_values('date')
            if len(df_sorted) > days_back:
                df_filtered = df_sorted.tail(days_back)
            else:
                df_filtered = df_sorted  # Use all data if not enough
            
            if len(df_filtered) < 10:  # Need minimum data points
                st.warning(f"Not enough data for {days_back} days: {len(df_filtered)} points")
                return None
            
            # Calculate test split for this time period
            test_size = 0.2
            split_idx = int(len(df_filtered) * (1 - test_size))
            train_data = df_filtered.iloc[:split_idx]
            test_data = df_filtered.iloc[split_idx:]
            
            if len(test_data) < 5:  # Need minimum test data
                st.warning(f"Not enough test data: {len(test_data)} points")
                return None
            
            # Evaluate models on this time period
            mae_scores = []
            
            # ARIMA evaluation
            try:
                arima_mae = evaluate_model('arima', train_data, test_data)
                if arima_mae is not None:
                    mae_scores.append(arima_mae)
                    st.info(f"ARIMA MAE: {arima_mae:.2f}")
            except Exception as e:
                st.error(f"ARIMA evaluation failed: {e}")
            
            # Prophet evaluation
            try:
                prophet_mae = evaluate_model('prophet', train_data, test_data)
                if prophet_mae is not None:
                    mae_scores.append(prophet_mae)
                    st.info(f"Prophet MAE: {prophet_mae:.2f}")
            except Exception as e:
                st.error(f"Prophet evaluation failed: {e}")
            
            # LSTM evaluation
            try:
                lstm_mae = evaluate_model('lstm', train_data, test_data)
                if lstm_mae is not None:
                    mae_scores.append(lstm_mae)
                    st.info(f"LSTM MAE: {lstm_mae:.2f}")
            except Exception as e:
                st.error(f"LSTM evaluation failed: {e}")
            
            if mae_scores:
                avg_mae = np.mean(mae_scores)
                st.success(f"Average MAE calculated: {avg_mae:.2f}")
                return avg_mae
            else:
                st.warning("No valid MAE scores calculated")
                return None
                
        except Exception as e:
            st.error(f"Error calculating dynamic MAE: {e}")
            return None
    
    def check_pipeline_status(self) -> str:
        """Check overall pipeline health"""
        try:
            # Check if models directory exists and has recent files
            models_dir = "models"
            if not os.path.exists(models_dir):
                return "No Models"
            
            # Check for recent model files (within last 24 hours)
            recent_files = 0
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) > (datetime.now() - timedelta(hours=24)).timestamp():
                        recent_files += 1
            
            if recent_files > 0:
                return "Healthy"
            else:
                return "Stale"
                
        except Exception:
            return "Unknown"
    
    def render_model_performance_chart(self):
        """Render model performance comparison chart"""
        st.header("📊 Model Performance Comparison")
        
        # Get performance data
        df = self.get_model_performance_data(days_back=30)
        
        if df.empty:
            st.warning("No performance data available")
            return
        
        # Create performance comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE Comparison', 'RMSE Comparison', 'MAPE Comparison', 'Training Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        metrics = ['mae', 'rmse', 'mape', 'training_time']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for metric, pos in zip(metrics, positions):
            for model in df['model_name'].unique():
                model_data = df[df['model_name'] == model]
                if not model_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=model_data['start_time'],
                            y=model_data[metric],
                            mode='lines+markers',
                            name=f"{model.upper()} - {metric.upper()}",
                            showlegend=True
                        ),
                        row=pos[0], col=pos[1]
                    )
        
        fig.update_layout(height=600, title_text="Model Performance Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    def render_model_selection_info(self):
        """Render model selection information"""
        st.header("🎯 Model Selection")
        
        if not MODEL_SELECTOR_AVAILABLE:
            st.warning("Model selector not available. Model selection features are disabled.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Current Selection")
            
            # Get current model selection
            try:
                selection_result = model_selector.update_model_selection()
                
                if selection_result:
                    selected_model = selection_result.get('selected_model', 'None')
                    st.metric("Selected Model", selected_model.upper())
                    
                    performance_data = selection_result.get('performance_data', {})
                    if selected_model in performance_data:
                        metrics = performance_data[selected_model]
                        st.write("**Performance Metrics:**")
                        st.write(f"- MAE: {metrics.get('mae', 'N/A'):.4f}")
                        st.write(f"- RMSE: {metrics.get('rmse', 'N/A'):.4f}")
                        st.write(f"- MAPE: {metrics.get('mape', 'N/A'):.4f}")
                else:
                    st.warning("No model selection data available")
            except Exception as e:
                st.error(f"Error getting model selection: {e}")
        
        with col2:
            st.subheader("Model Recommendations")
            
            use_cases = ['general', 'accuracy_focused', 'robustness_focused', 'interpretability_focused']
            selected_use_case = st.selectbox("Select Use Case", use_cases)
            
            if st.button("Get Recommendation"):
                try:
                    recommendation = model_selector.get_model_recommendation(selected_use_case)
                    
                    if recommendation:
                        st.write(f"**Recommended Model:** {recommendation.get('recommended_model', 'N/A')}")
                        st.write(f"**Reason:** {recommendation.get('reason', 'N/A')}")
                        st.write(f"**Confidence:** {recommendation.get('confidence', 'N/A')}")
                except Exception as e:
                    st.error(f"Error getting recommendation: {e}")
    
    def render_pipeline_status(self):
        """Render pipeline status and recent activity"""
        st.header("⚙️ Pipeline Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Activity")
            
            # Check for recent model files
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
        
        with col2:
            st.subheader("System Health")
            
            # Check various system components
            health_checks = {
                "MLflow Connection": self.experiment is not None,
                "Models Directory": os.path.exists("models"),
                "Config File": os.path.exists("config.ini"),
                "Data Directory": os.path.exists("data")
            }
            
            for check, status in health_checks.items():
                icon = "✅" if status else "❌"
                color = "status-success" if status else "status-error"
                st.markdown(f'<span class="{color}">{icon} {check}</span>', unsafe_allow_html=True)
    
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
    
    def render_sales_data_with_df(self, df):
        """Render sales data with the provided dataframe"""
        # Fix day of week names and month names
        df = self.fix_day_of_week(df)
        df = self.fix_month_names(df)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Sales Overview", "📅 Time Series", "🏷️ Product Analysis", "🔮 Forecast Comparison", "📋 Data Table"])
        
        with tab1:
            self.render_sales_overview(df)
        
        with tab2:
            self.render_time_series(df)
        
        with tab3:
            self.render_product_analysis(df)
        
        with tab4:
            self.render_forecast_comparison(df)
        
        with tab5:
            self.render_data_table(df)
    
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
    
    def fix_month_names(self, df):
        """Convert month numbers to actual month names"""
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }
        
        if 'month' in df.columns:
            df['month_name'] = df['month'].apply(lambda x: month_names.get(int(x), f'Month {int(x)}'))
        
        return df
    
    def render_sales_overview(self, df):
        """Render sales overview metrics and charts"""
        st.subheader("Sales Overview")
        
        # Calculate key metrics
        total_sales = df['sales'].sum()
        avg_daily_sales = df.groupby('date')['sales'].sum().mean()
        total_categories = df['product_id'].nunique()
        date_range = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        
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
        
        # 3. Weekly Sales Pattern
        st.subheader("📅 Weekly Sales Pattern")
        weekly_sales = date_filtered_df.groupby(['day_name', 'product_id'])['sales'].mean().reset_index()
        weekly_sales = weekly_sales.sort_values('day_name', key=lambda x: pd.Categorical(x, [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]))
        
        fig = px.line(weekly_sales, x='day_name', y='sales', color='product_id',
                     title="Average Sales by Day of Week",
                     labels={'sales': 'Average Sales ($)', 'day_name': 'Day of Week', 'product_id': 'Product Category'},
                     markers=True)
        fig.update_layout(
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=40, t=60, b=60),  # Add margins for labels
            xaxis=dict(
                gridcolor='lightgray',
                tickangle=0,
                tickmode='array',
                ticktext=weekly_sales['day_name'].unique(),
                tickvals=list(range(len(weekly_sales['day_name'].unique())))
            ),
            yaxis=dict(
                gridcolor='lightgray',
                tickformat=',',
                tickprefix='$'
            ),
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Summary Statistics
        st.subheader("📊 Summary Statistics")
        summary = date_filtered_df.groupby('product_id').agg({
            'sales': ['sum', 'mean', 'std', 'min', 'max', 'count']
        }).round(2)
        summary.columns = ['Total Sales', 'Mean Sales', 'Std Dev', 'Min Sales', 'Max Sales', 'Data Points']
        summary = summary.sort_values('Total Sales', ascending=False)
        st.dataframe(summary, use_container_width=True)
    
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
        
        # Sales distribution by category
        fig = px.box(df, x='product_id', y='sales',
                    title="📦 Sales Distribution by Product Category",
                    labels={'sales': 'Sales Amount ($)', 'product_id': 'Product Category'})
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            margin=dict(l=60, r=40, t=60, b=60),  # Add margins for labels
            xaxis=dict(
                tickangle=0,
                tickmode='array',
                ticktext=df['product_id'].unique(),
                tickvals=list(range(len(df['product_id'].unique())))
            ),
            yaxis=dict(
                tickformat=',',
                tickprefix='$'
            ),
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by day of week for each category
        day_category_sales = df.groupby(['day_name', 'product_id'])['sales'].mean().reset_index()
        day_category_sales = day_category_sales.sort_values('day_name', key=lambda x: pd.Categorical(x, [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]))
        
        fig = px.bar(day_category_sales, x='day_name', y='sales', color='product_id',
                    title="📅 Average Sales by Day and Category",
                    labels={'sales': 'Average Sales ($)', 'day_name': 'Day of Week', 'product_id': 'Product Category'},
                    barmode='group')
        fig.update_layout(
            height=400, 
            plot_bgcolor='white', 
            paper_bgcolor='white',
            margin=dict(l=60, r=40, t=60, b=60),  # Add margins for labels
            xaxis=dict(
                tickangle=0,
                tickmode='array',
                ticktext=day_category_sales['day_name'].unique(),
                tickvals=list(range(len(day_category_sales['day_name'].unique())))
            ),
            yaxis=dict(
                tickformat=',',
                tickprefix='$'
            ),
            font=dict(size=12)
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
        
        # Fix month names and day names
        filtered_df = self.fix_month_names(filtered_df)
        
        # Select meaningful columns for display
        display_columns = ['date', 'product_id', 'sales', 'day_name', 'month_name', 'is_weekend']
        display_df = filtered_df[display_columns].copy()
        
        # Rename columns for better display
        display_df.columns = ['Date', 'Product Category', 'Sales ($)', 'Day of Week', 'Month', 'Weekend']
        
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
    
    def render_forecast_comparison(self, df):
        """Render forecast vs actual comparison"""
        st.subheader("🔮 Forecast vs Actual Sales")
        
        try:
            # Load model predictions if available
            predictions = self.load_model_predictions()
            
            if predictions is not None and not predictions.empty:
                # Create a proper comparison dataframe
                # The predictions file has: date, product_id, model, predicted_sales, actual_sales
                # We need to pivot this to have one row per date/product with columns for each model
                
                # Pivot predictions to have models as columns
                pivot_predictions = predictions.pivot_table(
                    index=['date', 'product_id'], 
                    columns='model', 
                    values='predicted_sales',
                    aggfunc='first'
                ).reset_index()
                
                # Add actual sales column
                actual_sales = predictions.groupby(['date', 'product_id'])['actual_sales'].first().reset_index()
                comparison_df = pivot_predictions.merge(actual_sales, on=['date', 'product_id'], how='left')
                
                # Rename columns for consistency
                comparison_df = comparison_df.rename(columns={'actual_sales': 'actual'})
                
                if not comparison_df.empty:
                    # Calculate forecast accuracy metrics for each model
                    model_metrics = {}
                    available_models = [col for col in comparison_df.columns if col not in ['date', 'product_id', 'actual']]
                    
                    for model in available_models:
                        if model in comparison_df.columns:
                            model_data = comparison_df.dropna(subset=[model])
                            if not model_data.empty:
                                mae = np.mean(np.abs(model_data['actual'] - model_data[model]))
                                mape = np.mean(np.abs((model_data['actual'] - model_data[model]) / model_data['actual'])) * 100
                                model_metrics[model] = {'mae': mae, 'mape': mape, 'count': len(model_data)}
                    
                    # Display metrics
                    if model_metrics:
                        st.write("**📊 Forecast Accuracy Metrics:**")
                        metric_cols = st.columns(len(model_metrics))
                        for i, (model, metrics) in enumerate(model_metrics.items()):
                            with metric_cols[i]:
                                st.metric(f"{model.upper()} MAE", f"${metrics['mae']:.2f}")
                                st.metric(f"{model.upper()} MAPE", f"{metrics['mape']:.1f}%")
                                st.metric(f"{model.upper()} Periods", metrics['count'])
                    
                    # Model comparison chart
                    st.subheader("📈 Model Performance Comparison")
                    
                    # Prepare data for plotting
                    plot_data = []
                    for model in available_models:
                        if model in comparison_df.columns:
                            model_data = comparison_df.dropna(subset=[model])
                            if not model_data.empty:
                                for _, row in model_data.iterrows():
                                    plot_data.append({
                                        'date': row['date'],
                                        'product_id': row['product_id'],
                                        'model': model.upper(),
                                        'actual': row['actual'],
                                        'predicted': row[model]
                                    })
                    
                    if plot_data:
                        plot_df = pd.DataFrame(plot_data)
                        
                        # Scatter plot comparing all models
                        fig = px.scatter(plot_df, x='actual', y='predicted', 
                                       color='model', title="Forecast vs Actual Sales by Model",
                                       labels={'actual': 'Actual Sales ($)', 'predicted': 'Predicted Sales ($)'})
                        
                        # Add perfect prediction line
                        max_val = max(plot_df['actual'].max(), plot_df['predicted'].max())
                        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', 
                                               name='Perfect Prediction', line=dict(dash='dash', color='red')))
                        
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Time series comparison
                        st.subheader("📅 Time Series Comparison")
                        selected_product = st.selectbox("Select Product for Time Series", 
                                                      sorted(plot_df['product_id'].unique()))
                        
                        if selected_product:
                            product_data = plot_df[plot_df['product_id'] == selected_product]
                            
                            fig = go.Figure()
                            
                            # Add actual line
                            actual_data = product_data.groupby('date')['actual'].first().reset_index()
                            fig.add_trace(go.Scatter(x=actual_data['date'], y=actual_data['actual'], 
                                                   mode='lines+markers', name='Actual', 
                                                   line=dict(color='black', width=3)))
                            
                            # Add predicted lines for each model
                            colors = ['red', 'blue', 'green', 'orange', 'purple']
                            for i, model in enumerate(available_models):
                                if model in comparison_df.columns:
                                    model_data = product_data[product_data['model'] == model.upper()]
                                    if not model_data.empty:
                                        fig.add_trace(go.Scatter(x=model_data['date'], y=model_data['predicted'], 
                                                               mode='lines+markers', name=f'{model.upper()} Predicted',
                                                               line=dict(color=colors[i % len(colors)])))
                            
                            fig.update_layout(title=f"Forecast vs Actual: {selected_product}",
                                            xaxis_title="Date", yaxis_title="Sales Amount ($)",
                                            height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No forecast data available for plotting")
                else:
                    st.warning("No forecast data available for comparison")
            else:
                st.info("No model predictions available. Train models first to see forecast comparisons.")
                
        except Exception as e:
            st.error(f"Error loading forecast data: {e}")
    
    def load_model_predictions(self):
        """Load model predictions from saved files"""
        try:
            # Try to load predictions from multiple possible locations
            prediction_files = [
                "predictions/model_predictions.csv",
                "predictions/sample_predictions.csv",
                "predictions/predictions.csv"
            ]
            
            for file_path in prediction_files:
                if os.path.exists(file_path):
                    predictions_df = pd.read_csv(file_path)
                    
                    # Convert date column
                    if 'date' in predictions_df.columns:
                        predictions_df['date'] = pd.to_datetime(predictions_df['date'])
                    
                    # Ensure we have the required columns
                    required_cols = ['date', 'product_id', 'model', 'predicted_sales', 'actual_sales']
                    if all(col in predictions_df.columns for col in required_cols):
                        st.success(f"✅ Loaded predictions from {file_path}")
                        return predictions_df
            
            # If no prediction files found, return None
            return None
            
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            return None
    
    def render_notifications(self):
        """Render notification history"""
        st.header("🔔 Recent Notifications")
        
        if not NOTIFICATIONS_AVAILABLE:
            st.warning("Notifications not available. Notification history cannot be displayed.")
            return
        
        if hasattr(alert_manager, 'notification_history'):
            notifications = alert_manager.notification_history[-10:]  # Last 10 notifications
            
            if notifications:
                for notification in reversed(notifications):
                    timestamp = notification['timestamp']
                    notification_type = notification['type']
                    message = notification['message']
                    success = notification['success']
                    
                    status_icon = "✅" if success else "❌"
                    st.write(f"{status_icon} **{timestamp}** - {notification_type}: {message}")
            else:
                st.write("No recent notifications")
        else:
            st.write("Notification history not available")
    
    def render_sidebar(self, df=None):
        """Render sidebar controls"""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Date range selector
        st.sidebar.subheader("Date Range")
        days_back = st.sidebar.slider("Days to look back", 1, 90, 30)
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.rerun()
        
        # Model selection strategy
        st.sidebar.subheader("Model Selection")
        strategy = st.sidebar.selectbox(
            "Selection Strategy",
            ['composite', 'mae', 'rmse'],
            help="Strategy for selecting the best model"
        )
        
        # Display preferences
        st.sidebar.subheader("Display Preferences")
        show_sales_amounts = st.sidebar.checkbox(
            "Show Sales Amounts", 
            value=True,
            help="Display actual sales amounts in addition to error metrics"
        )
        
        # Global filters
        st.sidebar.subheader("🔍 Global Filters")
        if df is not None:
            all_categories = sorted(df['product_id'].unique())
            selected_categories = st.sidebar.multiselect(
                "Product Categories",
                all_categories,
                default=all_categories,
                help="Filter data by product categories across all sections"
            )
        else:
            selected_categories = []
        
        if st.sidebar.button("Update Model Selection"):
            if MODEL_SELECTOR_AVAILABLE:
                try:
                    performance_data = model_selector.get_latest_model_performance()
                    best_model, info = model_selector.select_best_model(performance_data, strategy)
                    st.sidebar.success(f"Selected: {best_model}")
                except Exception as e:
                    st.sidebar.error(f"Error updating model selection: {e}")
            else:
                st.sidebar.warning("Model selector not available")
        
        # Export data
        st.sidebar.subheader("Export")
        if st.sidebar.button("📊 Export Performance Data"):
            df = self.get_model_performance_data(days_back)
            if not df.empty:
                csv = df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"model_performance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        return days_back, show_sales_amounts, selected_categories
    
    def run(self):
        """Run the dashboard"""
        self.render_header()
        
        # Load data first for sidebar filters
        df = None
        if DATA_LOADER_AVAILABLE:
            try:
                df = load_data(processed=True)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = self.fix_day_of_week(df)
            except Exception as e:
                st.error(f"Error loading data: {e}")
        else:
            st.error("Data loader not available. Cannot load sales data.")
        
        days_back, show_sales_amounts, selected_categories = self.render_sidebar(df)
        
        # Create main tabs - Sales Data first, then Model Performance
        tab1, tab2, tab3 = st.tabs(["📈 Sales Data", "📊 Model Performance", "🔧 System Status"])
        
        with tab1:
            # Sales data content (now first)
            if df is not None:
                # Apply global category filter
                if selected_categories:
                    df_filtered = df[df['product_id'].isin(selected_categories)].copy()
                else:
                    df_filtered = df.copy()
                self.render_sales_data_with_df(df_filtered)
            else:
                st.error("No data available")
        
        with tab2:
            # Model performance content (now second)
            self.render_overview_metrics(days_back)
            self.render_model_performance_chart()
            self.render_model_selection_info()
        
        with tab3:
            # System status content
            self.render_pipeline_status()
            self.render_notifications()

# Create and run the dashboard
if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run() 