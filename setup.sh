#!/bin/bash
# Run this once when you first download the project.
# It creates a virtual environment and installs dependencies.

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up git (run once, then commit away)
git init
git add -A
git commit -m "Initial project structure — step 1: data pipeline"

echo ""
echo "Setup complete! To run the app:"
echo "  source venv/bin/activate"
echo "  python main.py"
