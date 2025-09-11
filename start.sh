#!/bin/bash
echo "🚀 Starting Q-Learning Tic Tac Toe App..."
echo "=================================================="

# Make sure we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this from the project directory."
    exit 1
fi

# Install dependencies
echo "📦 Installing/updating dependencies..."
pip install -r requirements.txt

echo ""
echo "🎮 Starting the app..."
echo "📍 Game will be available at: http://localhost:8000"
echo "📖 API docs will be available at: http://localhost:8000/docs"
echo "🛑 Press Ctrl+C to stop"
echo "=================================================="

# Run the app
python main.py
