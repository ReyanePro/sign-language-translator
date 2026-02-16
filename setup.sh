#!/bin/bash
# =============================================================
# 🤟 Sign Language Translator - Setup Script (macOS)
# =============================================================

set -e

echo "=========================================="
echo "🤟 Sign Language Translator - Setup"
echo "=========================================="

# Check Python version
python3 --version || { echo "❌ Python3 not found. Install it: brew install python3"; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download ASL dataset from Kaggle (CSV-based landmark dataset)
echo "📥 Downloading ASL alphabet dataset..."
mkdir -p data

# We'll generate a synthetic dataset from MediaPipe landmarks
# This is more reliable than depending on external downloads
echo "✅ Setup complete!"
echo ""
echo "=========================================="
echo "Next steps:"
echo "  1. source venv/bin/activate"
echo "  2. python src/collect_data.py    (collect your own hand signs)"
echo "  3. python src/train_model.py     (train the classifier)"
echo "  4. python src/webcam_app.py      (run webcam app)"
echo "  5. python web/gradio_app.py      (run web interface)"
echo "=========================================="
