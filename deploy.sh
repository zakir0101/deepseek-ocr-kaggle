#!/bin/bash

# DeepSeek OCR Deployment Script
# This script deploys the updated code to the Vast.ai server
# Usage: ./deploy.sh [-m "commit message"]

COMMIT_MESSAGE=""

# Parse command line arguments
while getopts "m:" opt; do
  case $opt in
    m)
      COMMIT_MESSAGE="$OPTARG"
      ;;
    \?)
      echo "Usage: $0 [-m \"commit message\"]"
      exit 1
      ;;
  esac
done

echo "🚀 Starting DeepSeek OCR deployment..."

# Step 1: Add, commit and push changes (only if there are changes)
echo "📝 Checking for changes to commit..."
git status

if [[ -n "$COMMIT_MESSAGE" ]]; then
    echo "📦 Committing changes with message: $COMMIT_MESSAGE"
    git add .
    git commit -m "$COMMIT_MESSAGE"
    git push origin master
    echo "✅ Changes committed and pushed"
elif [[ -z "$COMMIT_MESSAGE" ]]; then
    read -p "Do you want to commit and push changes? (y/n): " commit_choice

    if [[ $commit_choice == "y" || $commit_choice == "Y" ]]; then
        echo "📦 Committing changes..."
        git add .
        git commit -m "Fix: OCR server improvements - frontend display and boxes image generation"
        git push origin master
        echo "✅ Changes committed and pushed"
    else
        echo "⏭️ Skipping commit step"
    fi
fi

echo ""
echo "🔗 SSH connection instructions:"
echo "================================"
echo "Run this command to connect to the server:"
echo "ssh -p 40032 zakir@223.166.245.194 -L 8080:localhost:8080 -L 5000:localhost:5000"
echo ""
echo "📋 Once connected, run these commands:"
echo "===================================="
echo "1. Stop current server: pkill -9 python3"
echo "2. Navigate to project: cd /home/zakir/deepseek-ocr-kaggle"
echo "3. Pull latest changes: git pull origin master"
echo "4. Start server: python3 vast_server.py"
echo ""
echo "🌐 After starting the server, you can access it at:"
echo "- Health check: http://localhost:5000/health"
echo "- Frontend: http://localhost:8080"
echo ""
echo "✅ Deployment script completed!"
echo "📋 Follow the instructions above to deploy to the server."