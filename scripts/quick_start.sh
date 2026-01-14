#!/bin/bash

###############################################################################
# RepGen Quick Start
#
# Simplest way to run the pipeline.
#
# Usage: 
#   bash quick_start.sh [BUGS] [MAX_ATTEMPTS]
#
# Examples:
#   bash quick_start.sh 1-10 5
#   bash quick_start.sh 80-82 1
#   bash quick_start.sh 1-106 5      # Full paper replication
#
# Works on: macOS, Linux, Windows (Git Bash / WSL)
#
fi

# Run pipeline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$SCRIPT_DIR/pipeline.sh" \
    --bugs "$BUGS" \
    --dataset dataset \
    --setup \
    --run \
    --max-attempts "$MAX_ATTEMPTS"

exit $?
