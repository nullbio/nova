#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Build the site
mkdocs build

# Deploy to GitHub Pages (if this is a GitHub project)
# Uncomment the following line to deploy
# mkdocs gh-deploy --force

# Deploy to a custom server (as an alternative)
# rsync -avz --delete site/ user@server:/path/to/website/