# 🚀 Retail Sales Forecasting Project

## 📊 Overview

This project implements a **production-ready forecasting pipeline** for daily product sales using multiple time series models:

- **📈 ARIMA** - Traditional statistical forecasting
- **🔮 Prophet** - Facebook's seasonal forecasting model  
- **🧠 LSTM** - Deep learning neural network approach

## ✨ Key Features

- **🤖 Automated Pipeline** - Scheduled retraining and data monitoring
- **📊 Interactive Dashboard** - Real-time model performance monitoring with **actual sales amounts**
- **📈 Sales Data Visualization** - Date and product category-based sales analysis
- **🔬 Experiment Tracking** - MLflow integration for model versioning
- **📧 Smart Notifications** - Email/Slack alerts for pipeline events
- **🎯 Model Selection** - Automatic best model identification
- **🔮 Forecast Comparison** - Predicted vs actual sales amounts
- **☁️ Cloud Ready** - Docker, Kubernetes, and cloud deployment support

## 🏗️ Architecture

The system uses **Prefect** for workflow orchestration, **MLflow** for experiment tracking, and **Streamlit** for the monitoring dashboard. It's designed to run automatically with minimal human intervention while providing comprehensive monitoring and alerting capabilities.

## Project Structure
```
forecasting_project/
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── scripts/
│   ├── preprocess.py
│   ├── train_arima.py
│   ├── train_prophet.py
│   ├── train_lstm.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── run_dashboard.py
│   ├── test_pipeline.py
│   └── start_automated_pipeline.py
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   ├── utils.py
│   │   └── monitor.py
│   ├── models/
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── lstm_model.py
│   │   ├── lstm_dataloader.py
│   │   └── model_selector.py
│   ├── notifications/
│   │   └── alert_manager.py
│   ├── pipeline/
│   │   └── retrain_flow.py
│   ├── tracking/
│   │   └── mlflow_utils.py
│   └── dashboard/
│       └── dashboard.py
├── config.example.ini
├── requirements.txt
├── README.md
├── DEPLOYMENT.md
└── .gitignore
```

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**Linux/Mac:**
```bash
git clone https://github.com/yourusername/forecasting_project.git
cd forecasting_project
chmod +x setup.sh
./setup.sh
```

**Windows:**
```bash
git clone https://github.com/yourusername/forecasting_project.git
cd forecasting_project
setup.bat
```

### Option 2: Manual Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/forecasting_project.git
   cd forecasting_project
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # OR
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup configuration**
   ```bash
   cp config.example.ini config.ini
   ```

5. **Add your data**
   ```bash
   # Place your CSV file in data/raw/
   cp your_sales_data.csv data/raw/retail_sales_dataset.csv
   ```

6. **Run the pipeline**
   ```bash
   python scripts/preprocess.py
   python scripts/train_prophet.py
   python scripts/train_arima.py
   python scripts/train_lstm.py
   ```

7. **Start the dashboard**
   ```bash
   python scripts/run_dashboard.py
   ```

8. **Open dashboard**
   ```
   http://localhost:8501
   ```

## Setup (Legacy)

## Usage

### 🚀 Automated Pipeline (Recommended)

The pipeline is designed to run automatically without manual intervention:

1. **Start the automated pipeline:**
   ```bash
   python scripts/start_automated_pipeline.py
   ```

   This will:
   - ✅ Start Prefect server and agent
   - ✅ Create scheduled retraining deployment (daily at 2 AM UTC)
   - ✅ Start data monitoring (checks for new data every 5 minutes)
   - ✅ Automatically trigger retraining when new data is detected
   - ✅ Send email notifications for pipeline success/failure

2. **Monitor the pipeline:**
   - **Dashboard:** http://localhost:8501
   - **MLflow UI:** http://localhost:5000
   - **Prefect UI:** http://localhost:4200

### Manual Execution
- Training scripts: `scripts/train_arima.py`, `scripts/train_prophet.py`, `scripts/train_lstm.py`
- Inference: `scripts/inference.py`
- Evaluation: `scripts/evaluate.py`
- Preprocessing: `scripts/preprocess.py`
- Dashboard: `python scripts/run_dashboard.py`
- Sales Demo: `python scripts/demo_sales_dashboard.py`

### Pipeline Components
- **Scheduled retraining**: Prefect flow in `src/pipeline/retrain_flow.py` and deployment in `retrain_deployment.yaml`
- **Data monitoring**: `src/data/monitor.py` for detecting new data
- **Pipeline testing**: `scripts/test_pipeline.py` to validate pipeline functionality
- **Notifications**: `src/notifications/alert_manager.py` for email/Slack alerts
- **Model selection**: `src/models/model_selector.py` for automatic best model selection
- **Monitoring dashboard**: `src/dashboard/dashboard.py` for real-time monitoring

## 📊 Enhanced Dashboard Features

The dashboard now includes comprehensive sales data visualization beyond just MAE metrics:

### 📈 Sales Data Tab
- **Sales Overview**: Total sales, average daily sales, product count, date range
- **Time Series Analysis**: Interactive charts with date range and product filters
- **Product Analysis**: Performance metrics, sales distribution, heatmaps
- **Forecast Comparison**: Predicted vs actual sales amounts (when models are trained)
- **Data Table**: Interactive table with filtering and export capabilities

### 🎯 Key Benefits
1. **📊 Actual Business Impact**: See real sales amounts, not just error metrics
2. **📅 Time-based Analysis**: Track sales trends over specific date ranges
3. **🏷️ Product Performance**: Compare sales across different product categories
4. **🔮 Forecast Comparison**: Compare predicted vs actual sales amounts
5. **📊 Interactive Filtering**: Filter by date ranges and specific products
6. **📥 Data Export**: Download filtered data for further analysis

### 🚀 Dashboard Usage
```bash
# Start the dashboard
python scripts/run_dashboard.py

# Demo sales features
python scripts/demo_sales_dashboard.py
```

**Dashboard URL:** http://localhost:8501
- **Model Performance Tab**: Traditional MAE/RMSE metrics
- **Sales Data Tab**: Actual sales amounts and business insights
- **System Status Tab**: Pipeline health and notifications

### MLflow Tracking
- Experiment tracking: see `src/tracking/mlflow_utils.py`
- Model registry and versioning
- Performance metrics logging

## ☁️ Cloud Deployment

For cloud deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

### Quick Cloud Setup

1. **Copy configuration:**
   ```bash
   cp config.example.ini config.ini
   ```

2. **Update with your cloud settings:**
   - Database credentials
   - Cloud storage (S3/GCS/Azure)
   - Email/Slack notifications
   - MLflow/Prefect server URLs

3. **Deploy using your preferred method:**
   - Docker: `docker-compose up -d`
   - Kubernetes: `kubectl apply -f k8s/`
   - Cloud platforms: See DEPLOYMENT.md

## Notebooks
- EDA: `notebooks/eda.ipynb`
- Model comparison: `notebooks/model_comparison.ipynb`

## Configuration
- Edit `config.ini` for model and pipeline parameters.

## Production Pipeline

### Automatic Data Handling
The pipeline automatically:
1. **Detects new data** - Monitors raw data file for changes
2. **Validates data** - Checks format, columns, and data quality
3. **Preprocesses data** - Handles missing values, outliers, feature engineering
4. **Retrains models** - Updates all models with latest data
5. **Runs inference** - Generates new forecasts
6. **Logs everything** - Tracks all steps in MLflow

### Error Handling & Monitoring
- **Retry logic** - Failed tasks retry up to 3 times
- **Timeout protection** - 1-hour timeout per pipeline run
- **Comprehensive logging** - All steps logged with timestamps
- **Data validation** - Ensures data quality before processing

### Deployment
```bash
# Deploy the pipeline
prefect deploy -f retrain_deployment.yaml

# Start the agent
prefect agent start

# Test the pipeline
python scripts/test_pipeline.py
```
