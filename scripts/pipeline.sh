#!/bin/bash

###############################################################################
# RepGen Unified Pipeline - Single configurable entry point
#
# A single, parameterized script for the complete bug reproduction workflow.
#
# Features:
#  - Setup code from repositories (clone at specific commits)
#  - Copy bug reports and context files
#  - Run the reproduction pipeline
#  - Works with dataset/ or custom ae_dataset/ folder
#  - Highly configurable via command-line parameters
#
# Usage:
#   bash pipeline.sh [OPTIONS]
#
# Examples:
#   # Setup and run on original dataset, bugs 1-3
#   bash pipeline.sh --bugs 1-3 --setup --run
#
#   # Setup ae_dataset only, bugs 80-82
#   bash pipeline.sh --bugs 80-82 --dataset ae_dataset --setup
#
#   # Run pipeline only on existing ae_dataset, bugs 80-82
#   bash pipeline.sh --bugs 80-82 --dataset ae_dataset --run
#
#   # Setup and run with custom config
#   bash pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --run \
#                   --max-attempts 3 --retrieval full_system --generation all_steps
#
# Options:
#   --bugs RANGE               Bug IDs to process (e.g., "1-3", "80-82", or "80,81,82")
#   --dataset PATH             Dataset path (default: dataset, use ae_dataset for experimental)
#   --setup                    Setup: clone code and copy files (default: false)
#   --run                      Run: execute pipeline (default: false)
#   --skip-code                Skip code setup (use if already setup)
#   --force-clone              Force re-clone of repositories
#   --max-attempts N           Max generation attempts (default: 1)
#   --retrieval ABLATION       Retrieval ablation (default: full_system)
#   --generation ABLATION      Generation ablation (default: all_steps)
#   --help                     Show this help
#
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_PATH="$PROJECT_DIR/dataset"
CACHE_DIR="$PROJECT_DIR/.code_cache"
CSV_FILE="$PROJECT_DIR/dataset/Dataset.csv"

# Default parameters
BUGS=""
SETUP=false
RUN=false
SKIP_CODE=false
FORCE_CLONE=false
MAX_ATTEMPTS=1
RETRIEVAL="full_system"
GENERATION="all_steps"

# Color output (disabled on Windows CMD)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    GREEN=''
    BLUE=''
    YELLOW=''
    RED=''
    NC=''
else
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    NC='\033[0m'
fi

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bugs) BUGS="$2"; shift 2 ;;
        --dataset) DATASET_PATH="$2"; shift 2 ;;
        --setup) SETUP=true; shift ;;
        --run) RUN=true; shift ;;
        --skip-code) SKIP_CODE=true; shift ;;
        --force-clone) FORCE_CLONE=true; shift ;;
        --max-attempts) MAX_ATTEMPTS="$2"; shift 2 ;;
        --retrieval) RETRIEVAL="$2"; shift 2 ;;
        --generation) GENERATION="$2"; shift 2 ;;
        --help)
            grep "^#" "$0" | head -50
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$BUGS" ]; then
    log_error "Must specify --bugs (e.g., '1-3' or '80-82')"
    exit 1
fi

if [ "$SETUP" = false ] && [ "$RUN" = false ]; then
    log_error "Must specify at least --setup or --run"
    exit 1
fi

# Parse bug range
parse_bugs() {
    local range="$1"
    if [[ "$range" == *"-"* ]]; then
        local start=$(echo "$range" | cut -d'-' -f1)
        local end=$(echo "$range" | cut -d'-' -f2)
        seq "$start" "$end"
    else
        echo "$range" | tr ',' '\n'
    fi
}

BUG_IDS=($(parse_bugs "$BUGS"))
BUG_START=""
BUG_END=""
TOTAL_BUGS=${#BUG_IDS[@]}

# Get first and last safely
for i in "${!BUG_IDS[@]}"; do
    if [ "$i" -eq 0 ]; then
        BUG_START="${BUG_IDS[$i]}"
    fi
    BUG_END="${BUG_IDS[$i]}"
done

echo ""
echo "=========================================="
echo "  RepGen Unified Pipeline"
echo "=========================================="
echo ""
log_info "Dataset: $DATASET_PATH"
log_info "Bugs: $BUGS (${#BUG_IDS[@]} total)"
log_info "Setup: $SETUP | Run: $RUN"
if [ "$RUN" = true ]; then
    log_info "Max attempts: $MAX_ATTEMPTS"
    log_info "Retrieval ablation: $RETRIEVAL"
    log_info "Generation ablation: $GENERATION"
fi
echo ""

# Extract repo info from CSV (cross-platform compatible)
get_repo_info() {
    local bug_id=$1
    local csv_line=$((bug_id + 1))
    # Use sed safely - need to handle Windows differently
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows: use Python for line extraction
        local line=$(python -c "
with open('$CSV_FILE', 'r') as f:
    for i, l in enumerate(f):
        if i == $csv_line:
            print(l.strip())
            break
")
    else
        # Unix/Mac: use sed
        local line=$(sed -n "${csv_line}p" "$CSV_FILE")
    fi
    IFS=',' read -r url hash <<< "$line"
    local repo=$(echo $url | sed 's|/issues.*||')
    echo "${repo}.git|$(echo $hash | xargs)"
}

# Get cache key (cross-platform compatible)
get_cache_key() {
    echo "$1" | sed 's|.*github.com/||' | sed 's|/|__|g' | sed 's|\.git||'
}

# ============================================================================
# SETUP PHASE
# ============================================================================

if [ "$SETUP" = true ]; then
    echo "=========================================="
    echo "  Setup Phase"
    echo "=========================================="
    echo ""
    
    mkdir -p "$CACHE_DIR"
    
    # If dataset_path is not the default, create directories
    if [ "$DATASET_PATH" != "$PROJECT_DIR/dataset" ]; then
        log_info "Creating custom dataset structure at: $DATASET_PATH"
        mkdir -p "$DATASET_PATH"
    fi
    
    SETUP_SUCCESS=0
    SETUP_FAILED=0
    
    for bug_id in "${BUG_IDS[@]}"; do
        BID=$(printf "%03d" "$bug_id")
        ORIG_BDIR="$PROJECT_DIR/dataset/$BID"
        AE_BDIR="$DATASET_PATH/$BID"
        AE_CDIR="$AE_BDIR/code"
        AE_BRDIR="$AE_BDIR/bug_report"
        AE_CTXDIR="$AE_BDIR/context"
        AE_REPDIR="$AE_BDIR/reproduction_code"
        
        log_info "Bug $BID"
        
        # Validate original dataset
        if [ ! -d "$ORIG_BDIR" ]; then
            log_error "  Original bug directory not found: $ORIG_BDIR"
            SETUP_FAILED=$((SETUP_FAILED + 1))
            continue
        fi
        
        # Create directories
        mkdir -p "$AE_CDIR" "$AE_BRDIR" "$AE_CTXDIR" "$AE_REPDIR"
        
        # Copy bug report
        if [ -f "$ORIG_BDIR/bug_report/$BID.txt" ]; then
            cp "$ORIG_BDIR/bug_report/$BID.txt" "$AE_BRDIR/"
        else
            log_warning "  Bug report not found"
        fi
        
        # Copy context files
        if [ -d "$ORIG_BDIR/context" ] && [ "$(ls -A "$ORIG_BDIR/context" 2>/dev/null)" ]; then
            cp "$ORIG_BDIR/context"/* "$AE_CTXDIR/" 2>/dev/null || true
        fi
        
        if [ "$SKIP_CODE" = true ]; then
            log_info "  Skipping code setup"
            SETUP_SUCCESS=$((SETUP_SUCCESS + 1))
            continue
        fi
        
        # Get repo info
        INFO=$(get_repo_info "$bug_id")
        REPO=$(echo "$INFO" | cut -d'|' -f1)
        HASH=$(echo "$INFO" | cut -d'|' -f2)
        
        if [ -z "$REPO" ] || [ -z "$HASH" ]; then
            log_error "  Invalid repo info from CSV"
            SETUP_FAILED=$((SETUP_FAILED + 1))
            continue
        fi
        
        KEY=$(get_cache_key "$REPO")
        CACHED="$CACHE_DIR/$KEY"
        
        # Clone if needed
        if [ ! -d "$CACHED" ] || [ "$FORCE_CLONE" = true ]; then
            if [ -d "$CACHED" ]; then
                rm -rf "$CACHED"
            fi
            log_info "  Cloning repo..."
            if ! git clone --quiet "$REPO" "$CACHED" 2>/dev/null; then
                log_error "  Clone failed"
                SETUP_FAILED=$((SETUP_FAILED + 1))
                continue
            fi
        fi
        
        # Checkout commit
        (
            cd "$CACHED"
            git fetch --quiet origin "$HASH" 2>/dev/null || true
            git checkout --quiet "$HASH" 2>/dev/null || true
        ) || true
        
        # Copy code
        rm -rf "$AE_CDIR"/*
        cp -r "$CACHED"/* "$AE_CDIR/" 2>/dev/null || true
        (cd "$CACHED" && find . -maxdepth 1 -type f -name ".*" ! -name ".git*" -exec cp {} "$AE_CDIR/" \; 2>/dev/null) || true
        
        log_success "  Ready"
        SETUP_SUCCESS=$((SETUP_SUCCESS + 1))
    done
    
    echo ""
    log_success "Setup: $SETUP_SUCCESS/$TOTAL_BUGS"
    if [ $SETUP_FAILED -gt 0 ]; then
        log_warning "Failed: $SETUP_FAILED"
    fi
    echo ""
fi

# ============================================================================
# RUN PHASE
# ============================================================================

if [ "$RUN" = true ]; then
    echo "=========================================="
    echo "  Run Phase"
    echo "=========================================="
    echo ""
    
    # Activate venv (cross-platform)
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows
        if [ -f "$PROJECT_DIR/venv/Scripts/activate" ]; then
            source "$PROJECT_DIR/venv/Scripts/activate" 2>/dev/null || true
        fi
    else
        # Unix/Mac
        if [ -d "$PROJECT_DIR/venv" ]; then
            source "$PROJECT_DIR/venv/bin/activate" 2>/dev/null || true
        fi
    fi
    
    # Check API key
    if [ -z "$OPENAI_API_KEY" ]; then
        log_error "OPENAI_API_KEY not set"
        exit 1
    fi
    
    log_success "Environment ready"
    echo ""
    
    RUN_SUCCESS=0
    RUN_FAILED=0
    
    for bug_id in "${BUG_IDS[@]}"; do
        BID=$(printf "%03d" "$bug_id")
        BDIR="$DATASET_PATH/$BID"
        CDIR="$BDIR/code"
        
        if [ ! -d "$CDIR" ]; then
            log_error "Code directory missing for bug $BID"
            RUN_FAILED=$((RUN_FAILED + 1))
            continue
        fi
        
        log_info "Bug $BID"
        
        if python "$PROJECT_DIR/src/tool_openai.py" \
            --bug_id="$BID" \
            --max-attempts="$MAX_ATTEMPTS" \
            --retrieval_ablation="$RETRIEVAL" \
            --generation_ablation="$GENERATION" \
            --ae_dataset_path="$DATASET_PATH" 2>&1 | tail -15; then
            
            log_success "  Completed"
            RUN_SUCCESS=$((RUN_SUCCESS + 1))
        else
            log_warning "  Failed"
            RUN_FAILED=$((RUN_FAILED + 1))
        fi
        echo ""
    done
    
    echo "=========================================="
    echo "  Run Summary"
    echo "=========================================="
    echo ""
    log_success "Successful: $RUN_SUCCESS/$TOTAL_BUGS"
    if [ $RUN_FAILED -gt 0 ]; then
        log_warning "Failed: $RUN_FAILED"
    fi
    echo ""
fi

log_success "Pipeline complete!"
echo ""

exit 0
