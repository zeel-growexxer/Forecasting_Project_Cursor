#!/bin/bash
# =============================================================================
# FORECASTING PROJECT - INSTALLATION SCRIPT
# =============================================================================
# This script sets up the forecasting project environment.
# 
# USAGE:
#   chmod +x install.sh
#   ./install.sh
# =============================================================================

set -e  # Exit on any error

echo "🚀 Setting up Forecasting Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python version..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python3"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            print_success "Python $PYTHON_VERSION found"
            PYTHON_CMD="python"
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip..."
    if command -v pip3 &> /dev/null; then
        PIP_CMD="pip3"
        print_success "pip3 found"
    elif command -v pip &> /dev/null; then
        PIP_CMD="pip"
        print_success "pip found"
    else
        print_error "pip not found. Please install pip"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    $PIP_CMD install --upgrade pip
    print_success "pip upgraded"
}

# Install requirements
install_requirements() {
    print_status "Installing main requirements..."
    $PIP_CMD install -r requirements.txt
    print_success "Main requirements installed"
}

# Install development requirements (optional)
install_dev_requirements() {
    print_status "Installing development requirements..."
    $PIP_CMD install -r requirements-dev.txt
    print_success "Development requirements installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    mkdir -p data/raw data/processed models logs
    touch data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep
    print_success "Project directories created"
}

# Copy example configuration
setup_config() {
    print_status "Setting up configuration..."
    if [ ! -f "config.ini" ]; then
        if [ -f "config.example.ini" ]; then
            cp config.example.ini config.ini
            print_success "Configuration file created from example"
            print_warning "Please edit config.ini with your settings"
        else
            print_warning "No config.example.ini found. Please create config.ini manually"
        fi
    else
        print_success "Configuration file already exists"
    fi
}

# Test installation
test_installation() {
    print_status "Testing installation..."
    
    # Test Python imports
    $PYTHON_CMD -c "
import sys
sys.path.append('.')
try:
    import pandas as pd
    import numpy as np
    import streamlit as st
    import plotly.express as px
    import mlflow
    import prefect
    print('✅ All core packages imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation test passed"
    else
        print_error "Installation test failed"
        exit 1
    fi
}

# Main installation process
main() {
    echo "=============================================================================="
    echo "FORECASTING PROJECT - INSTALLATION"
    echo "=============================================================================="
    
    check_python
    check_pip
    create_venv
    activate_venv
    upgrade_pip
    install_requirements
    
    # Ask if user wants development dependencies
    echo ""
    read -p "Do you want to install development dependencies? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_dev_requirements
    fi
    
    create_directories
    setup_config
    test_installation
    
    echo ""
    echo "=============================================================================="
    print_success "Installation completed successfully!"
    echo "=============================================================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Edit config.ini with your settings"
    echo "3. Run the dashboard: streamlit run src/dashboard/dashboard.py"
    echo "4. Or run the pipeline: python scripts/test_pipeline.py"
    echo ""
    echo "For more information, see the README.md file"
    echo ""
}

# Run main function
main "$@" 