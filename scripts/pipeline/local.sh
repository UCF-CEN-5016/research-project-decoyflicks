#!/bin/bash

###############################################################################
# RepGen Pipeline with Ollama (Local Inference)
#
# Unified entry point for bug reproduction using local Ollama models.
# No OpenAI API needed - runs completely locally on your machine.
#
# Prerequisites:
#   1. Install Ollama: https://ollama.ai/download
#   2. Pull models: ollama pull qwen2.5:7b && ollama pull qwen2.5-coder:7b
#   3. Start Ollama: ollama serve
#
# Features:
#  - Setup code from repositories (clone at specific commits)
#  - Copy bug reports and context files
#  - Run the reproduction pipeline with local Ollama models
#  - Works with dataset/ or custom ae_dataset/ folder
#  - Highly configurable via command-line parameters
#
# Usage:
#   bash scripts/pipeline/local.sh [OPTIONS]
#
# Examples:
#   # Setup and run on original dataset, bugs 1-3
#   bash scripts/pipeline/local.sh --bugs 1-3 --setup --run
#
#   # Setup ae_dataset only, bugs 80-82
#   bash scripts/pipeline/local.sh --bugs 80-82 --dataset ae_dataset --setup
#
#   # Run pipeline only on existing ae_dataset, bugs 80-82
#   bash scripts/pipeline/local.sh --bugs 80-82 --dataset ae_dataset --run --skip-code
#
#   # Setup and run with custom config
#   bash scripts/pipeline/local.sh --bugs 80-82 --dataset ae_dataset --setup --run \
#                          --max-attempts 3 --retrieval full_system --generation all_steps
#
###############################################################################

set -e

# Color output
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
    NC='\033[0m'
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
BUGS=""
DATASET="dataset"
SETUP=false
RUN=false
SKIP_CODE=false
FORCE_CLONE=false
MAX_ATTEMPTS=1
RETRIEVAL_ABLATION="full_system"
GENERATION_ABLATION="all_steps"
TOOL_SCRIPT="tool.py"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bugs)
            BUGS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --setup)
            SETUP=true
            shift
            ;;
        --run)
            RUN=true
            shift
            ;;
        --skip-code)
            SKIP_CODE=true
            shift
            ;;
        --force-clone)
            FORCE_CLONE=true
            shift
            ;;
        --max-attempts)
            MAX_ATTEMPTS="$2"
            shift 2
            ;;
        --retrieval)
            RETRIEVAL_ABLATION="$2"
            shift 2
            ;;
        --generation)
            GENERATION_ABLATION="$2"
            shift 2
            ;;
        --help)
            cat << 'EOF'
RepGen Ollama Pipeline - Local Inference

USAGE:
    bash ollama_pipeline.sh [OPTIONS]

OPTIONS:
    --bugs RANGE              Bug IDs to process (e.g., 1-10, 80-82, 80,81,82)
    --dataset PATH            Dataset path (default: dataset)
    --setup                   Setup phase: clone code and copy files
    --run                     Run phase: execute pipeline
    --skip-code               Skip code cloning if already setup
    --force-clone             Force fresh repository clones
    --max-attempts N          Max generation attempts per bug (default: 1)
    --retrieval ABLATION      Retrieval ablation config (default: full_system)
    --generation ABLATION     Generation ablation config (default: all_steps)
    --help                    Show this help message

EXAMPLES:
    # Quick test: setup and run bugs 80-82
    bash ollama_pipeline.sh --bugs 80-82 --setup --run

    # Setup only (no run)
    bash ollama_pipeline.sh --bugs 80-82 --dataset ae_dataset --setup

    # Run only (skip setup)
    bash ollama_pipeline.sh --bugs 80-82 --dataset ae_dataset --run --skip-code

    # Full replication: all 106 bugs
    bash ollama_pipeline.sh --bugs 1-106 --dataset dataset --setup --run --max-attempts 3

MODELS:
    - qwen2.5:7b (reasoning, planning)
    - qwen2.5-coder:7b (code generation)

REQUIREMENTS:
    1. Ollama installed: https://ollama.ai/download
    2. Models downloaded: ollama pull qwen2.5:7b && ollama pull qwen2.5-coder:7b
    3. Ollama service running: ollama serve

EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$BUGS" ]; then
    echo -e "${RED}Error: --bugs is required${NC}"
    echo "Usage: bash scripts/pipeline/local.sh --bugs 1-10 --setup --run"
    exit 1
fi

if [ "$SETUP" = false ] && [ "$RUN" = false ]; then
    echo -e "${RED}Error: At least one of --setup or --run is required${NC}"
    exit 1
fi

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      RepGen Pipeline with Ollama (Local Inference)            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Bugs: $BUGS"
echo "  Dataset: $DATASET"
echo "  Setup: $SETUP"
echo "  Run: $RUN"
echo "  Skip Code: $SKIP_CODE"
echo "  Max Attempts: $MAX_ATTEMPTS"
echo "  Retrieval Ablation: $RETRIEVAL_ABLATION"
echo "  Generation Ablation: $GENERATION_ABLATION"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Ollama
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}❌ Ollama is not installed${NC}"
    echo "Download from: https://ollama.ai/download"
    exit 1
fi

echo -e "${YELLOW}✓ Ollama is installed${NC}"

# Check Ollama service
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${RED}❌ Ollama service is not running${NC}"
    echo "Start it with: ollama serve"
    exit 1
fi

echo -e "${YELLOW}✓ Ollama service is running${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed${NC}"
    exit 1
fi

echo -e "${YELLOW}✓ Python3 is installed${NC}"

# Check virtual environment
if [ ! -f "$PROJECT_ROOT/venv/bin/python" ] && [ ! -f "$PROJECT_ROOT/venv/Scripts/python.exe" ]; then
    echo -e "${RED}❌ Virtual environment not found${NC}"
    echo "Create it with: python3 -m venv venv"
    exit 1
fi

echo -e "${YELLOW}✓ Virtual environment found${NC}"
echo ""

# Activate virtual environment
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/Scripts/activate" ]; then
    source "$PROJECT_ROOT/venv/Scripts/activate"
fi

# Navigate to project root
cd "$PROJECT_ROOT"

# Convert BUGS range to array
convert_range() {
    local range=$1
    local bugs=()
    
    if [[ $range == *"-"* ]]; then
        # Range like 1-10
        IFS='-' read -ra PARTS <<< "$range"
        for ((i=${PARTS[0]}; i<=${PARTS[1]}; i++)); do
            bugs+=("$(printf '%03d' "$i")")
        done
    else
        # Comma-separated like 1,2,3 or single bug
        IFS=',' read -ra PARTS <<< "$range"
        for part in "${PARTS[@]}"; do
            bugs+=("$(printf '%03d' "$part")")
        done
    fi
    
    echo "${bugs[@]}"
}

BUGS_ARRAY=($(convert_range "$BUGS"))
TOTAL_BUGS=${#BUGS_ARRAY[@]}

if [ "$SETUP" = true ]; then
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                      SETUP PHASE                              ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Processing $TOTAL_BUGS bugs...${NC}"
    echo ""
    
    # Run setup phase - this is identical to the standard pipeline
    # as setup doesn't involve LLM calls
    python3 src/dataset_creation.py \
        --bugs "${BUGS_ARRAY[@]}" \
        --dataset "$DATASET" \
        --force-clone "$FORCE_CLONE"
    
    echo ""
    echo -e "${GREEN}✓ Setup phase completed${NC}"
    echo ""
fi

if [ "$RUN" = true ]; then
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                      RUN PHASE                                ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Processing $TOTAL_BUGS bugs with Ollama...${NC}"
    echo "Models: qwen2.5:7b, qwen2.5-coder:7b"
    echo ""
    
    FAILED_BUGS=()
    SUCCESSFUL_BUGS=()
    
    for ((i=0; i<${#BUGS_ARRAY[@]}; i++)); do
        BUG_ID="${BUGS_ARRAY[$i]}"
        CURRENT=$((i+1))
        
        echo -e "${BLUE}─────────────────────────────────────────────────────────────────${NC}"
        echo -e "${YELLOW}[$CURRENT/$TOTAL_BUGS] Processing bug $BUG_ID${NC}"
        echo -e "${BLUE}─────────────────────────────────────────────────────────────────${NC}"
        
        # Run tool.py for this bug (with Ollama backend)
        if python3 src/tool.py \
            --bug_id "$BUG_ID" \
            --retrieval_ablation "$RETRIEVAL_ABLATION" \
            --generation_ablation "$GENERATION_ABLATION" \
            --max-attempts "$MAX_ATTEMPTS"; then
            
            SUCCESSFUL_BUGS+=("$BUG_ID")
            echo -e "${GREEN}✓ Bug $BUG_ID completed successfully${NC}"
        else
            FAILED_BUGS+=("$BUG_ID")
            echo -e "${RED}✗ Bug $BUG_ID failed${NC}"
        fi
        
        echo ""
    done
    
    # Print summary
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                      SUMMARY                                  ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    
    echo "Total bugs: $TOTAL_BUGS"
    echo -e "${GREEN}Successful: ${#SUCCESSFUL_BUGS[@]}${NC}"
    echo -e "${RED}Failed: ${#FAILED_BUGS[@]}${NC}"
    echo ""
    
    if [ ${#SUCCESSFUL_BUGS[@]} -gt 0 ]; then
        echo -e "${GREEN}Successful bugs:${NC}"
        for bug in "${SUCCESSFUL_BUGS[@]}"; do
            echo "  ✓ $bug"
        done
        echo ""
    fi
    
    if [ ${#FAILED_BUGS[@]} -gt 0 ]; then
        echo -e "${RED}Failed bugs:${NC}"
        for bug in "${FAILED_BUGS[@]}"; do
            echo "  ✗ $bug"
        done
        echo ""
    fi
    
    # Results location
    echo -e "${YELLOW}Results location:${NC}"
    echo "  $DATASET/<bug_id>/reproduction_code/reproduce_<bug_id>.py"
    echo ""
    
    # Exit status
    if [ ${#FAILED_BUGS[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ All bugs processed successfully!${NC}"
        exit 0
    else
        exit 1
    fi
fi
