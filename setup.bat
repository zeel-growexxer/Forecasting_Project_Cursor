@echo off
REM =============================================================================
REM FORECASTING PROJECT - AUTOMATED SETUP SCRIPT (Windows)
REM =============================================================================
REM This script automates the complete setup process for new users on Windows.
REM It handles environment creation, dependency installation, and configuration.
REM
REM USAGE:
REM   setup.bat
REM
REM WHAT THIS SCRIPT DOES:
REM 1. Checks for Python installation
REM 2. Creates a virtual environment
REM 3. Installs all required dependencies
REM 4. Sets up configuration files
REM 5. Provides next steps for data and training
REM =============================================================================

echo 🚀 Setting up Forecasting Pipeline Dashboard...

REM Check if Python 3 is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Copy config file
echo ⚙️ Setting up configuration...
if not exist "config.ini" (
    copy config.example.ini config.ini
    echo ✅ Created config.ini from template
) else (
    echo ✅ config.ini already exists
)

REM Check for data file
echo 📊 Checking for data file...
if not exist "data\raw\retail_sales_dataset.csv" (
    echo ⚠️  No data file found at data\raw\retail_sales_dataset.csv
    echo    Please add your CSV file to data\raw\retail_sales_dataset.csv
    echo    Then run: python scripts\preprocess.py
) else (
    echo ✅ Data file found
)

echo.
echo 🎉 Setup complete!
echo.
echo 📋 Next steps:
echo 1. Add your CSV data to data\raw\retail_sales_dataset.csv
echo 2. Run: python scripts\preprocess.py
echo 3. Run: python scripts\train_prophet.py
echo 4. Run: python scripts\train_arima.py
echo 5. Run: python scripts\train_lstm.py
echo 6. Run: python scripts\run_dashboard.py
echo 7. Open: http://localhost:8501
echo.
echo 🔧 Or run the automated pipeline:
echo    python scripts\start_automated_pipeline.py
pause 