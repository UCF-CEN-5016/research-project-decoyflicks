#!/bin/bash

###############################################################################
# RepGen Unified Pipeline - Single configurable entry point
#
# Usage:
#   bash pipeline.sh [OPTIONS]
###############################################################################

# Ensure pipeline fails if any part of a pipe fails (catches Python errors)
set -o pipefail

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
QUIET=false

# Color output - Using printf-safe ANSI codes
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    GREEN=''
    BLUE=''
    YELLOW=''
    RED=''
    CYAN=''
    MAGENTA=''
    BOLD=''
    NC=''
    SPINNER_SUPPORT=false
else
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    RED='\033[0;31m'
    CYAN='\033[0;36m'
    MAGENTA='\033[0;35m'
    BOLD='\033[1m'
    NC='\033[0m'
    SPINNER_SUPPORT=true
fi

# Logging functions using printf for consistent cross-platform behavior
log_info() {
    if [ "$QUIET" = false ]; then
        printf "${BLUE}[INFO]${NC} %s\n" "$1"
    fi
}

log_success() { printf "${GREEN}[✓]${NC} %s\n" "$1"; }
log_warning() { printf "${YELLOW}[WARNING]${NC} %s\n" "$1"; }
log_error() { printf "${RED}[✗]${NC} %s\n" "$1"; }

log_step() {
    if [ "$QUIET" = false ]; then
        printf "${CYAN}➜${NC} %s\n" "$1"
    fi
}

log_substep() {
    if [ "$QUIET" = false ]; then
        printf "  ${MAGENTA}•${NC} %s\n" "$1"
    fi
}

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local label=$3
    
    if [ "$QUIET" = true ]; then
        return
    fi
    
    local percent=$((current * 100 / total))
    local filled=$((percent / 2))
    local empty=$((50 - filled))
    
    printf "\r${BOLD}${label}${NC} ["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%% (%d/%d)" "$percent" "$current" "$total"
}

# Spinner for long operations
spinner_pid=""
start_spinner() {
    if [ "$QUIET" = true ] || [ "$SPINNER_SUPPORT" = false ]; then
        return
    fi
    
    local message=$1
    (
        local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
        local i=0
        while true; do
            i=$(( (i+1) % 10 ))
            printf "\r${CYAN}${spin:$i:1}${NC} ${message}"
            sleep 0.1
        done
    ) &
    spinner_pid=$!
}

stop_spinner() {
    if [ -n "$spinner_pid" ]; then
        kill "$spinner_pid" 2>/dev/null || true
        wait "$spinner_pid" 2>/dev/null || true
        spinner_pid=""
        printf "\r"
    fi
}

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
        --quiet) QUIET=true; shift ;;
        --help)
            grep "^#" "$0" | head -60
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
printf "==========================================\n"
printf "  ${BOLD}RepGen Unified Pipeline${NC}\n"
printf "==========================================\n"
echo ""
log_info "Dataset: ${BOLD}$DATASET_PATH${NC}"
log_info "Bugs: ${BOLD}$BUGS${NC} (${TOTAL_BUGS} total)"
log_info "Setup: ${BOLD}$SETUP${NC} | Run: ${BOLD}$RUN${NC}"
if [ "$RUN" = true ]; then
    log_info "Max attempts: ${BOLD}$MAX_ATTEMPTS${NC}"
    log_info "Retrieval ablation: ${BOLD}$RETRIEVAL${NC}"
    log_info "Generation ablation: ${BOLD}$GENERATION${NC}"
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
    printf "==========================================\n"
    printf "  ${BOLD}Setup Phase${NC}\n"
    printf "==========================================\n"
    echo ""
    
    mkdir -p "$CACHE_DIR"
    
    # If dataset_path is not the default, create directories
    if [ "$DATASET_PATH" != "$PROJECT_DIR/dataset" ]; then
        log_step "Creating custom dataset structure at: $DATASET_PATH"
        mkdir -p "$DATASET_PATH"
    fi
    
    SETUP_SUCCESS=0
    SETUP_FAILED=0
    CURRENT_BUG=0
    
    for bug_id in "${BUG_IDS[@]}"; do
        CURRENT_BUG=$((CURRENT_BUG + 1))
        BID=$(printf "%03d" "$bug_id")
        ORIG_BDIR="$PROJECT_DIR/dataset/$BID"
        AE_BDIR="$DATASET_PATH/$BID"
        AE_CDIR="$AE_BDIR/code"
        AE_BRDIR="$AE_BDIR/bug_report"
        AE_CTXDIR="$AE_BDIR/context"
        AE_REPDIR="$AE_BDIR/reproduction_code"
        
        show_progress "$CURRENT_BUG" "$TOTAL_BUGS" "Setup Progress"
        echo ""
        log_step "Processing Bug ${BOLD}$BID${NC} (${CURRENT_BUG}/${TOTAL_BUGS})"
        
        # Validate original dataset
        if [ ! -d "$ORIG_BDIR" ]; then
            log_error "  Original bug directory not found: $ORIG_BDIR"
            SETUP_FAILED=$((SETUP_FAILED + 1))
            continue
        fi
        
        # Create directories
        log_substep "Creating directory structure..."
        mkdir -p "$AE_CDIR" "$AE_BRDIR" "$AE_CTXDIR" "$AE_REPDIR"
        
        # Copy bug report
        if [ -f "$ORIG_BDIR/bug_report/$BID.txt" ]; then
            log_substep "Copying bug report..."
            cp "$ORIG_BDIR/bug_report/$BID.txt" "$AE_BRDIR/"
        else
            log_warning "  Bug report not found"
        fi
        
        if [ "$SKIP_CODE" = true ]; then
            log_info "  Skipping code setup"
            SETUP_SUCCESS=$((SETUP_SUCCESS + 1))
            continue
        fi
        
        # Get repo info
        log_substep "Reading repository info..."
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
                log_substep "Removing old cache..."
                rm -rf "$CACHED"
            fi
            
            log_substep "Cloning repository: ${REPO##*/}"
            start_spinner "Cloning repository..."
            
            if git clone "$REPO" "$CACHED" > /tmp/git_clone_$$.log 2>&1; then
                stop_spinner
                log_success "  Repository cloned"
            else
                stop_spinner
                log_error "  Clone failed (see /tmp/git_clone_$$.log)"
                SETUP_FAILED=$((SETUP_FAILED + 1))
                continue
            fi
        else
            log_substep "Using cached repository"
        fi
        
        # Checkout commit
        log_substep "Checking out commit: ${HASH:0:8}..."
        (
            cd "$CACHED"
            git fetch origin "$HASH" > /tmp/git_fetch_$$.log 2>&1 || true
            git checkout "$HASH" > /tmp/git_checkout_$$.log 2>&1 || true
        ) || true
        
        # Copy code
        log_substep "Copying code to workspace..."
        rm -rf "$AE_CDIR"/*
        cp -r "$CACHED"/* "$AE_CDIR/" 2>/dev/null || true
        (cd "$CACHED" && find . -maxdepth 1 -type f -name ".*" ! -name ".git*" -exec cp {} "$AE_CDIR/" \; 2>/dev/null) || true
        
        log_success "  Bug $BID ready"
        SETUP_SUCCESS=$((SETUP_SUCCESS + 1))
        echo ""
    done
    
    show_progress "$TOTAL_BUGS" "$TOTAL_BUGS" "Setup Progress"
    echo ""
    echo ""
    log_success "Setup complete: ${BOLD}$SETUP_SUCCESS/$TOTAL_BUGS${NC} successful"
    if [ $SETUP_FAILED -gt 0 ]; then
        log_warning "Failed: ${BOLD}$SETUP_FAILED${NC}"
    fi
    echo ""
fi

# ============================================================================
# RUN PHASE
# ============================================================================

if [ "$RUN" = true ]; then
    printf "==========================================\n"
    printf "  ${BOLD}Run Phase${NC}\n"
    printf "==========================================\n"
    echo ""
    
    # Activate venv (cross-platform)
    log_step "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows
        if [ -f "$PROJECT_DIR/venv/Scripts/activate" ]; then
            source "$PROJECT_DIR/venv/Scripts/activate" 2>/dev/null || true
            log_success "  Virtual environment activated (Windows)"
        fi
    else
        # Unix/Mac
        if [ -d "$PROJECT_DIR/venv" ]; then
            source "$PROJECT_DIR/venv/bin/activate" 2>/dev/null || true
            log_success "  Virtual environment activated (Unix/Mac)"
        fi
    fi
    
    # Check API key
    if [ -z "$OPENAI_API_KEY" ]; then
        log_error "OPENAI_API_KEY not set"
        exit 1
    fi
    
    log_success "Environment configured"
    echo ""
    
    RUN_SUCCESS=0
    RUN_FAILED=0
    CURRENT_BUG=0
    
    for bug_id in "${BUG_IDS[@]}"; do
        CURRENT_BUG=$((CURRENT_BUG + 1))
        BID=$(printf "%03d" "$bug_id")
        BDIR="$DATASET_PATH/$BID"
        CDIR="$BDIR/code"
        
        if [ ! -d "$CDIR" ]; then
            log_error "Code directory missing for bug $BID"
            RUN_FAILED=$((RUN_FAILED + 1))
            continue
        fi
        
        show_progress "$CURRENT_BUG" "$TOTAL_BUGS" "Pipeline Progress"
        echo ""
        log_step "Running pipeline for Bug ${BOLD}$BID${NC} (${CURRENT_BUG}/${TOTAL_BUGS})"
        echo ""
        
        # Create a temp file for output
        TEMP_OUTPUT="/tmp/repgen_output_$BID.log"
        
        # Run Python with live output
        # Use PIPESTATUS to get the exit code of python, not the while loop
        if [ "$QUIET" = false ]; then
            python "$PROJECT_DIR/src/tool_openai.py" \
                --bug_id="$BID" \
                --max-attempts="$MAX_ATTEMPTS" \
                --retrieval_ablation="$RETRIEVAL" \
                --generation_ablation="$GENERATION" \
                --ae_dataset_path="$DATASET_PATH" 2>&1 | while IFS= read -r line; do
                    echo "  ${line}"
                done
            RESULT=${PIPESTATUS[0]} 
        else
            python "$PROJECT_DIR/src/tool_openai.py" \
                --bug_id="$BID" \
                --max-attempts="$MAX_ATTEMPTS" \
                --retrieval_ablation="$RETRIEVAL" \
                --generation_ablation="$GENERATION" \
                --ae_dataset_path="$DATASET_PATH" > "$TEMP_OUTPUT" 2>&1
            RESULT=$?
        fi
        
        echo ""
        
        if [ $RESULT -eq 0 ]; then
            log_success "Bug $BID completed successfully"
            RUN_SUCCESS=$((RUN_SUCCESS + 1))
        else
            log_error "Bug $BID failed (Exit Code: $RESULT)"
            if [ "$QUIET" = true ] && [ -f "$TEMP_OUTPUT" ]; then
                echo "  Last 10 lines of output:"
                tail -10 "$TEMP_OUTPUT" | sed 's/^/    /'
            fi
            RUN_FAILED=$((RUN_FAILED + 1))
        fi
        echo ""
    done
    
    show_progress "$TOTAL_BUGS" "$TOTAL_BUGS" "Pipeline Progress"
    echo ""
    echo ""
    printf "==========================================\n"
    printf "  ${BOLD}Run Summary${NC}\n"
    printf "==========================================\n"
    echo ""
    log_success "Successful: ${BOLD}$RUN_SUCCESS/$TOTAL_BUGS${NC}"
    if [ $RUN_FAILED -gt 0 ]; then
        log_warning "Failed: ${BOLD}$RUN_FAILED${NC}"
    fi
    echo ""
fi

log_success "${BOLD}Pipeline complete!${NC}"
echo ""

exit 0