#!/bin/bash

###############################################################################
# RepGen Quick Start with Ollama (Local Inference)
#
# Simplest way to run the pipeline with local Ollama models.
# No cloud API needed - runs completely locally.
#
# Prerequisites:
#   1. Install Ollama: https://ollama.ai/download
#   2. Pull models: ollama pull qwen2.5:7b && ollama pull qwen2.5-coder:7b
#   3. Start Ollama: ollama serve (in another terminal)
#   4. Run this script
#
# Usage: 
#   bash ollama_quick_start.sh [BUGS] [MAX_ATTEMPTS]
#
# Examples:
#   bash ollama_quick_start.sh 1-10 3        # Bugs 1-10, max 3 attempts
#   bash ollama_quick_start.sh 80-82 1       # Bugs 80-82, quick test
#   bash ollama_quick_start.sh 1-106 5       # Full paper replication
#
# Works on: macOS, Linux, Windows (Git Bash / WSL)
#
###############################################################################

set -e

# Color output (disable on Windows CMD)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    NC=""
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Parse arguments
BUGS="${1:-80-82}"
MAX_ATTEMPTS="${2:-1}"

# Validate arguments
if [ -z "$BUGS" ]; then
    echo -e "${RED}Error: BUGS not specified${NC}"
    echo "Usage: bash ollama_quick_start.sh [BUGS] [MAX_ATTEMPTS]"
    echo "Examples:"
    echo "  bash ollama_quick_start.sh 80-82 1"
    echo "  bash ollama_quick_start.sh 1-10 5"
    exit 1
fi

if ! [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]] || [ "$MAX_ATTEMPTS" -lt 1 ]; then
    echo -e "${RED}Error: MAX_ATTEMPTS must be a positive integer${NC}"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      RepGen Quick Start with Ollama (Local Inference)         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Bugs: $BUGS"
echo "  Max Attempts: $MAX_ATTEMPTS"
echo "  Dataset: dataset"
echo "  Models: qwen2.5:7b, qwen2.5-coder:7b"
echo ""

# Check if Ollama is available
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Error: Ollama is not installed${NC}"
    echo "Download from: https://ollama.ai/download"
    exit 1
fi

echo -e "${YELLOW}✓ Ollama is installed${NC}"

# Check if Ollama service is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Ollama service is not running${NC}"
    echo "Start it in another terminal with: ollama serve"
    exit 1
fi

echo -e "${YELLOW}✓ Ollama service is running${NC}"

# Check if required models are available
echo ""
echo -e "${YELLOW}Checking required models...${NC}"

MODELS=$(ollama list 2>/dev/null || echo "")

if ! echo "$MODELS" | grep -q "qwen2.5:7b"; then
    echo -e "${RED}❌ Model qwen2.5:7b not found${NC}"
    echo "Pull it with: ollama pull qwen2.5:7b"
    exit 1
fi

if ! echo "$MODELS" | grep -q "qwen2.5-coder:7b"; then
    echo -e "${RED}❌ Model qwen2.5-coder:7b not found${NC}"
    echo "Pull it with: ollama pull qwen2.5-coder:7b"
    exit 1
fi

echo -e "${YELLOW}✓ All required models are available${NC}"

# Navigate to project root
cd "$PROJECT_ROOT"

# Run pipeline with ollama_pipeline.sh
echo ""
echo -e "${BLUE}Running pipeline...${NC}"
echo ""

bash "$SCRIPT_DIR/ollama_pipeline.sh" \
    --bugs "$BUGS" \
    --dataset dataset \
    --setup \
    --run \
    --max-attempts "$MAX_ATTEMPTS"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   ✅ Pipeline Completed!                       ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                   ❌ Pipeline Failed!                          ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
fi

exit $EXIT_CODE
