#!/bin/bash

echo "🧹 Cleaning up repository for GitHub push..."
echo "=============================================="

# Remove large data files from Git tracking
echo "📁 Removing data files from Git tracking..."
echo "   Note: Users will generate their own data using scripts/generate_realistic_data.py"
git rm --cached data/raw/retail_sales_dataset.csv 2>/dev/null || echo "Data file not tracked"
git rm --cached data/processed/processed_retail_sales_data.csv 2>/dev/null || echo "Processed data not tracked"

# Remove prediction files
echo "🔮 Removing prediction files from Git tracking..."
git rm --cached predictions/sample_predictions.csv 2>/dev/null || echo "Predictions CSV not tracked"
git rm --cached predictions/sample_predictions.json 2>/dev/null || echo "Predictions JSON not tracked"

# Remove config file with potential secrets
echo "🔐 Removing config file from Git tracking..."
git rm --cached config.ini 2>/dev/null || echo "Config file not tracked"

# Remove virtual environment if tracked
echo "🐍 Removing virtual environment from Git tracking..."
git rm -r --cached venv/ 2>/dev/null || echo "Virtual environment not tracked"

# Remove MLflow runs if tracked
echo "📊 Removing MLflow runs from Git tracking..."
git rm -r --cached mlruns/ 2>/dev/null || echo "MLflow runs not tracked"

# Remove model files if tracked
echo "🤖 Removing model files from Git tracking..."
git rm -r --cached models/*/ 2>/dev/null || echo "Model files not tracked"

# Remove any Python cache files
echo "🗑️ Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || echo "No cache files found"
find . -name "*.pyc" -delete 2>/dev/null || echo "No .pyc files found"

# Remove any temporary files
echo "🧽 Removing temporary files..."
find . -name "*.tmp" -delete 2>/dev/null || echo "No .tmp files found"
find . -name "*.temp" -delete 2>/dev/null || echo "No .temp files found"

echo ""
echo "✅ Cleanup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Review the changes: git status"
echo "2. Add the changes: git add ."
echo "3. Commit: git commit -m 'Clean up repository for GitHub'"
echo "4. Push: git push origin main"
echo ""
echo "📁 Files that will be ignored (good!):"
echo "   - data/raw/retail_sales_dataset.csv (users generate their own)"
echo "   - data/processed/processed_retail_sales_data.csv (generated from raw)"
echo "   - predictions/sample_predictions.* (generated during training)"
echo "   - config.ini (contains secrets)"
echo "   - venv/ (virtual environment)"
echo "   - mlruns/ (MLflow runs)"
echo "   - models/*/ (trained models)"
echo ""
echo "📁 Files that will be included (good!):"
echo "   - README.md (project documentation)"
echo "   - requirements.txt (dependencies)"
echo "   - install.sh & install.bat (setup scripts)"
echo "   - config.example.ini (example configuration)"
echo "   - src/ (all source code)"
echo "   - scripts/ (including data generation script)"
echo "   - .gitignore (comprehensive ignore rules)" 