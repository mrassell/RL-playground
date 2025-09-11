#!/usr/bin/env python3
"""
Simple script to run the Q-Learning Tic Tac Toe app
"""
import subprocess
import sys
import os

def main():
    print("🚀 Starting Q-Learning Tic Tac Toe App...")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found. Please run this from the project directory.")
        sys.exit(1)
    
    # Install dependencies if needed
    try:
        import fastapi
        import uvicorn
        import numpy
        print("✅ Dependencies already installed")
    except ImportError:
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print("\n🎮 Starting the app...")
    print("📍 Game will be available at: http://localhost:8000")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Run the app
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 App stopped. Thanks for playing!")

if __name__ == "__main__":
    main()
