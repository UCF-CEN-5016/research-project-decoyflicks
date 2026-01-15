#!/bin/bash

###############################################################################
# RepGen Quick Start
#
# Simplest way to run the pipeline with dynamic progress indicators.
#
# Usage: 
#   bash quick_start.sh [BUGS] [MAX_ATTEMPTS] [OPTIONS]
#
# Examples:
#   bash quick_start.sh 1-10 5
#   bash quick_start.sh 80-82 1
#   bash quick_start.sh 1-106 5      # Full paper replication
#   bash quick_start.sh 1-10 5 --quiet  # Minimal output
#
# Works on: macOS, Linux, Windows (Git Bash / WSL)
#
###############################################################################

# Get script directory first
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set defaults
BUGS="${1:-1-10}"
MAX_ATTEMPTS="${2:-5}"
shift 2 2>/dev/null || shift $# 2>/dev/null  # Remove first two args if they exist

# Run pipeline with dynamic logging (pass through any extra args like --quiet)
bash "$SCRIPT_DIR/../pipeline/cloud.sh" \
    --bugs "$BUGS" \
    --dataset dataset_cloud \
    --setup \
    --run \
    --max-attempts "$MAX_ATTEMPTS" \
    "$@"

exit $?