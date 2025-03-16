#!/bin/bash

# Set script to exit immediately if any command exits with non-zero status
set -e

echo "🚀 Launching Nova Documentation Server"

# Navigate to the website directory if not already there
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if dependencies are installed
if ! command -v mkdocs &> /dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "✅ MkDocs is already installed"
fi

# Start the MkDocs server
echo "🌐 Starting documentation server at http://127.0.0.1:8000/"
echo "Press Ctrl+C to stop the server"
echo ""
mkdocs serve

# This line will only be reached if mkdocs serve exits without error
echo "👋 Server stopped"