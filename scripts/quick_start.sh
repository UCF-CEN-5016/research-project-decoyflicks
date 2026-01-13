#!/bin/bash

###############################################################################
# RepGen Quick Start - Ultra-simplified replication
#
# This is the absolute minimal way to run RepGen experiments.
# Just set your API key and run this script.
#
# Usage: 
#   bash quick_start.sh [BUG_START] [BUG_END] [MAX_ATTEMPTS]
#
# Example:
#   bash quick_start.sh 1 10 5
#   bash quick_start.sh 1 106 5   (Full paper replication)
#
###############################################################################

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default arguments
BUG_START=${1:-1}
BUG_END=${2:-10}
MAX_ATTEMPTS=${3:-5}

# Check if running full paper (all 106 bugs)
if [ "$BUG_END" -eq 106 ]; then
    echo "🚀 Starting FULL PAPER REPLICATION (bugs 1-106)"
    echo "   This may take several hours depending on your API rate limits"
else
    echo "🚀 Starting Quick Test Run (bugs $BUG_START-$BUG_END)"
fi

echo ""
echo "Prerequisites:"
echo "  ✓ Python 3.12 installed"
echo "  ✓ OPENAI_API_KEY exported (if using GPT-4.1)"
echo ""

# Verify API key is set (if using OpenAI)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   Set it with: export OPENAI_API_KEY='sk-...'"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "Starting replication..."
echo ""

# Run the main replication script with provided arguments
bash "$PROJECT_DIR/scripts/replicate.sh" \
    --bug-start "$BUG_START" \
    --bug-end "$BUG_END" \
    --max-attempts "$MAX_ATTEMPTS"

exit $?
