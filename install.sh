#!/bin/bash
# Quick Install Script

set -e

echo "🚀 Installing..."

# Check Python version
python3 --version || { echo "Python 3 required"; exit 1; }

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️ Please edit .env with your API keys"
fi

echo "✅ Installation complete!"
echo "Run: source venv/bin/activate && python gradio_app.py"
