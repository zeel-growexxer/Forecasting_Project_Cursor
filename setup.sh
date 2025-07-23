#!/bin/bash
# =============================================================================
# FORECASTING PROJECT - AUTOMATED SETUP SCRIPT (Linux/Mac)
# =============================================================================
# This script automates the complete setup process for new users.
# It handles environment creation, dependency installation, and configuration.
#
# USAGE:
#   chmod +x setup.sh
#   ./setup.sh
#
# WHAT THIS SCRIPT DOES:
# 1. Checks for Python 3 and pip installation
# 2. Creates a virtual environment
# 3. Installs all required dependencies
# 4. Sets up configuration files
# 5. Provides next steps for data and training
# =============================================================================

echo "🚀 Setting up Forecasting Pipeline Dashboard..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip first."
    exit 1
fi

echo "✅ Python and pip found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Copy config file
echo "⚙️ Setting up configuration..."
if [ ! -f "config.ini" ]; then
    cp config.example.ini config.ini
    echo "✅ Created config.ini from template"
else
    echo "✅ config.ini already exists"
fi

# Check for data file
echo "📊 Checking for data file..."
if [ ! -f "data/raw/retail_sales_dataset.csv" ]; then
    echo "⚠️  No data file found at data/raw/retail_sales_dataset.csv"
    echo "   Please add your CSV file to data/raw/retail_sales_dataset.csv"
    echo "   Then run: python scripts/preprocess.py"
else
    echo "✅ Data file found"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add your CSV data to data/raw/retail_sales_dataset.csv"
echo "2. Run: python scripts/preprocess.py"
echo "3. Run: python scripts/train_prophet.py"
echo "4. Run: python scripts/train_arima.py"
echo "5. Run: python scripts/train_lstm.py"
echo "6. Run: python scripts/run_dashboard.py"
echo "7. Open: http://localhost:8501"
echo ""
echo "🔧 Or run the automated pipeline:"
echo "   python scripts/start_automated_pipeline.py" 