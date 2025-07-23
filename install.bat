@echo off
REM =============================================================================
REM FORECASTING PROJECT - WINDOWS INSTALLATION SCRIPT
REM =============================================================================
REM This script sets up the forecasting project environment on Windows.
REM 
REM USAGE:
REM   install.bat
REM =============================================================================

echo 🚀 Setting up Forecasting Project...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [INFO] Found Python %PYTHON_VERSION%

REM Create virtual environment
echo [INFO] Creating virtual environment...
if exist venv (
    echo [WARNING] Virtual environment already exists. Removing...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment created

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment activated

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip
    pause
    exit /b 1
)
echo [SUCCESS] pip upgraded

REM Install requirements
echo [INFO] Installing main requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements
    pause
    exit /b 1
)
echo [SUCCESS] Main requirements installed

REM Ask about development dependencies
set /p INSTALL_DEV="Do you want to install development dependencies? (y/n): "
if /i "%INSTALL_DEV%"=="y" (
    echo [INFO] Installing development requirements...
    pip install -r requirements-dev.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install development requirements
        pause
        exit /b 1
    )
    echo [SUCCESS] Development requirements installed
)

REM Create necessary directories
echo [INFO] Creating project directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist models mkdir models
if not exist logs mkdir logs
echo. > data\raw\.gitkeep
echo. > data\processed\.gitkeep
echo. > models\.gitkeep
echo [SUCCESS] Project directories created

REM Copy example configuration
echo [INFO] Setting up configuration...
if not exist config.ini (
    if exist config.example.ini (
        copy config.example.ini config.ini
        echo [SUCCESS] Configuration file created from example
        echo [WARNING] Please edit config.ini with your settings
    ) else (
        echo [WARNING] No config.example.ini found. Please create config.ini manually
    )
) else (
    echo [SUCCESS] Configuration file already exists
)

REM Test installation
echo [INFO] Testing installation...
python -c "import pandas as pd; import numpy as np; import streamlit as st; import plotly.express as px; import mlflow; import prefect; print('✅ All core packages imported successfully')"
if errorlevel 1 (
    echo [ERROR] Installation test failed
    pause
    exit /b 1
)
echo [SUCCESS] Installation test passed

echo.
echo ==============================================================================
echo [SUCCESS] Installation completed successfully!
echo ==============================================================================
echo.
echo Next steps:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Edit config.ini with your settings
echo 3. Run the dashboard: streamlit run src\dashboard\dashboard.py
echo 4. Or run the pipeline: python scripts\test_pipeline.py
echo.
echo For more information, see the README.md file
echo.
pause 