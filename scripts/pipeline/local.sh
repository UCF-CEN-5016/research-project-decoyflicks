#!/bin/bash

###############################################################################
# RepGen Unified Pipeline - Local Inference (Ollama)
#
# Usage:
#   bash local.sh [OPTIONS]
###############################################################################

# Ensure pipeline fails if any part of a pipe fails
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Go up two levels: scripts/pipeline -> scripts -> Project Root
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET_PATH="$PROJECT_DIR/dataset_local"
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
LOG_FILE=""

# Detect terminal capabilities
USE_COLOR=false
USE_FANCY=false

if [ -t 1 ]; then
    if command -v tput >/dev/null 2>&1; then
        if [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
            USE_COLOR=true
            if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "cygwin" && "$OSTYPE" != "win32" ]]; then
                if [[ "${LANG}" =~ [Uu][Tt][Ff]-?8 ]] || [[ "${LC_ALL}" =~ [Uu][Tt][Ff]-?8 ]]; then
                    USE_FANCY=true
                fi
            fi
        fi
    fi
fi

# Color codes
if [ "$USE_COLOR" = true ]; then
    GREEN=$'\033[0;32m'
    BLUE=$'\033[0;34m'
    YELLOW=$'\033[1;33m'
    RED=$'\033[0;31m'
    CYAN=$'\033[0;36m'
    MAGENTA=$'\033[0;35m'
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    NC=$'\033[0m'
else
    GREEN='' BLUE='' YELLOW='' RED='' CYAN='' MAGENTA='' BOLD='' DIM='' NC=''
fi

# Safe symbols
if [ "$USE_FANCY" = true ]; then
    SYM_CHECK="✓" SYM_CROSS="✗" SYM_ARROW="➜" SYM_DOT="•"
else
    SYM_CHECK="OK" SYM_CROSS="X" SYM_ARROW=">" SYM_DOT="-"
fi

# Logging functions
log_info() {
    local msg="$1"
    if [ "$QUIET" = false ]; then
        echo -e "${BLUE}[INFO]${NC} $msg"
    fi
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $msg" >> "$LOG_FILE"
}

log_success() {
    local msg="$1"
    echo -e "${GREEN}[${SYM_CHECK}]${NC} $msg"
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $msg" >> "$LOG_FILE"
}

log_warning() {
    local msg="$1"
    echo -e "${YELLOW}[WARNING]${NC} $msg"
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARNING] $msg" >> "$LOG_FILE"
}

log_error() {
    local msg="$1"
    echo -e "${RED}[${SYM_CROSS}]${NC} $msg" >&2
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $msg" >> "$LOG_FILE"
}

log_step() {
    local msg="$1"
    if [ "$QUIET" = false ]; then
        echo -e "${CYAN}${SYM_ARROW}${NC} $msg"
    fi
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [STEP] $msg" >> "$LOG_FILE"
}

log_substep() {
    local msg="$1"
    if [ "$QUIET" = false ]; then
        echo -e "  ${MAGENTA}${SYM_DOT}${NC} $msg"
    fi
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUBSTEP] $msg" >> "$LOG_FILE"
}

log_debug() {
    local msg="$1"
    [ -n "$LOG_FILE" ] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEBUG] $msg" >> "$LOG_FILE"
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
    
    echo -ne "\r${BOLD}${label}${NC} ["
    if [ $filled -gt 0 ]; then
        printf "%${filled}s" | tr ' ' '#'
    fi
    if [ $empty -gt 0 ]; then
        printf "%${empty}s" | tr ' ' '-'
    fi
    echo -ne "] ${percent}% (${current}/${total})"
}

# Spinner for long operations
spinner_pid=""
start_spinner() {
    if [ "$QUIET" = true ] || [ "$USE_FANCY" = false ]; then
        return
    fi
    
    local message=$1
    (
        local spin='|/-\\'
        local i=0
        while true; do
            i=$(( (i+1) % 4 ))
            echo -ne "\r${CYAN}${spin:$i:1}${NC} ${message}"
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
        echo -ne "\r$(printf '%80s' '')\r"
    fi
}

# Cleanup function
cleanup() {
    stop_spinner
    log_warning "Pipeline interrupted by user"
    exit 130
}
trap cleanup INT TERM

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
        --log-file) LOG_FILE="$2"; shift 2 ;;
        --help)
            echo "Usage: bash scripts/pipeline/local.sh --bugs 1-10 --setup --run"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Initialize log file
if [ -n "$LOG_FILE" ]; then
    touch "$LOG_FILE" || { log_error "Cannot create log file: $LOG_FILE"; exit 1; }
    log_info "Logging to file: $LOG_FILE"
fi

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
TOTAL_BUGS=${#BUG_IDS[@]}

# Print banner
echo ""
echo "=========================================="
echo -e "  ${BOLD}RepGen Pipeline (Local/Ollama)${NC}"
echo "=========================================="
echo ""
log_info "Dataset: ${BOLD}${DATASET_PATH}${NC}"
log_info "Bugs: ${BOLD}${BUGS}${NC} (${TOTAL_BUGS} total)"
log_info "Setup: ${BOLD}${SETUP}${NC} | Run: ${BOLD}${RUN}${NC}"

# Extract repo info from CSV
get_repo_info() {
    local bug_id=$1
    local csv_line=$((bug_id + 1))
    
    # DETERMINE PYTHON COMMAND (Fix for macOS where 'python' might be missing)
    local python_cmd="python3"
    if ! command -v python3 &>/dev/null; then
        python_cmd="python"
    fi

    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows/Git Bash logic
        local line=$($python_cmd -c "
with open('$CSV_FILE', 'r') as f:
    for i, l in enumerate(f):
        if i == $csv_line:
            print(l.strip())
            break
")
    else
        # macOS/Linux logic (Using sed is faster/easier here)
        local line=$(sed -n "${csv_line}p" "$CSV_FILE")
    fi
    
    IFS=',' read -r url hash <<< "$line"
    local repo=$(echo $url | sed 's|/issues.*||')
    echo "${repo}.git|$(echo $hash | xargs)"
}

get_cache_key() {
    echo "$1" | sed 's|.*github.com/||' | sed 's|/|__|g' | sed 's|\.git||'
}

# ============================================================================
# SETUP PHASE
# ============================================================================

if [ "$SETUP" = true ]; then
    echo "=========================================="
    echo -e "  ${BOLD}Setup Phase${NC}"
    echo "=========================================="
    echo ""
    log_info "Starting setup phase"
    
    mkdir -p "$CACHE_DIR"
    
    if [ "$DATASET_PATH" != "$PROJECT_DIR/dataset" ]; then
        log_step "Creating custom dataset structure"
        mkdir -p "$DATASET_PATH"
        log_success "Custom dataset directory created"
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
        
        show_progress "$CURRENT_BUG" "$TOTAL_BUGS" "Setup Progress"
        echo ""
        log_step "Processing Bug ${BOLD}${BID}${NC} (${CURRENT_BUG}/${TOTAL_BUGS})"
        
        if [ ! -d "$ORIG_BDIR" ]; then
            log_error "  Original bug directory not found: $ORIG_BDIR"
            SETUP_FAILED=$((SETUP_FAILED + 1))
            continue
        fi
        
        mkdir -p "$AE_CDIR" "$AE_BRDIR" "$AE_BDIR/context" "$AE_BDIR/reproduction_code"
        
        if [ -f "$ORIG_BDIR/bug_report/$BID.txt" ]; then
            cp "$ORIG_BDIR/bug_report/$BID.txt" "$AE_BRDIR/" || {
                log_error "  Failed to copy bug report"
                SETUP_FAILED=$((SETUP_FAILED + 1))
                continue
            }
        fi
        
        if [ "$SKIP_CODE" = true ]; then
            log_info "  Skipping code setup (--skip-code enabled)"
            SETUP_SUCCESS=$((SETUP_SUCCESS + 1))
            continue
        fi
        
        INFO=$(get_repo_info "$bug_id")
        REPO=$(echo "$INFO" | cut -d'|' -f1)
        HASH=$(echo "$INFO" | cut -d'|' -f2)
        KEY=$(get_cache_key "$REPO")
        CACHED="$CACHE_DIR/$KEY"
        
        if [ ! -d "$CACHED" ] || [ "$FORCE_CLONE" = true ]; then
            rm -rf "$CACHED"
            start_spinner "Cloning repository..."
            if git clone --quiet "$REPO" "$CACHED" > /tmp/git_clone_$$.log 2>&1; then
                stop_spinner
                log_success "  Repository cloned successfully"
            else
                stop_spinner
                log_error "  Clone failed"
                SETUP_FAILED=$((SETUP_FAILED + 1))
                continue
            fi
        else
            log_substep "Using cached repository"
        fi
        
        log_substep "Checking out commit: ${HASH:0:8}"
        (cd "$CACHED" && git fetch --quiet origin "$HASH" > /dev/null 2>&1 && git checkout --quiet "$HASH") || true
        
        log_substep "Copying code to workspace"
        rm -rf "$AE_CDIR"/*
        cp -r "$CACHED"/* "$AE_CDIR/" 2>/dev/null
        (cd "$CACHED" && find . -maxdepth 1 -type f -name ".*" ! -name ".git*" -exec cp {} "$AE_CDIR/" \; 2>/dev/null) || true
        
        log_success "Bug $BID setup complete"
        SETUP_SUCCESS=$((SETUP_SUCCESS + 1))
        echo ""
    done
    
    show_progress "$TOTAL_BUGS" "$TOTAL_BUGS" "Setup Progress"
    echo ""
    log_success "Setup phase complete: ${BOLD}${SETUP_SUCCESS}/${TOTAL_BUGS}${NC} successful"
    echo ""
fi

# ============================================================================
# RUN PHASE (Local / Ollama Specific)
# ============================================================================

if [ "$RUN" = true ]; then
    echo "=========================================="
    echo -e "  ${BOLD}Run Phase (Local/Ollama)${NC}"
    echo "=========================================="
    echo ""
    log_info "Starting run phase"
    
    # Check Ollama Prerequisites
    if ! command -v ollama &> /dev/null; then
        log_error "Ollama is not installed. Please install from https://ollama.ai/download"
        exit 1
    fi
    
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_error "Ollama service is not running. Start with: ollama serve"
        exit 1
    fi
    log_success "Ollama service detected"

    # Activate Environment
    log_step "Activating virtual environment"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        [ -f "$PROJECT_DIR/venv/Scripts/activate" ] && source "$PROJECT_DIR/venv/Scripts/activate"
    else
        [ -d "$PROJECT_DIR/venv" ] && source "$PROJECT_DIR/venv/bin/activate"
    fi
    
    RUN_SUCCESS=0
    RUN_FAILED=0
    CURRENT_BUG=0
    
    for bug_id in "${BUG_IDS[@]}"; do
        CURRENT_BUG=$((CURRENT_BUG + 1))
        BID=$(printf "%03d" "$bug_id")
        BDIR="$DATASET_PATH/$BID"
        CDIR="$BDIR/code"
        
        if [ ! -d "$CDIR" ]; then
            log_error "Code directory missing for bug $BID: $CDIR"
            RUN_FAILED=$((RUN_FAILED + 1))
            continue
        fi
        
        show_progress "$CURRENT_BUG" "$TOTAL_BUGS" "Pipeline Progress"
        echo ""
        log_step "Running pipeline for Bug ${BOLD}${BID}${NC} (${CURRENT_BUG}/${TOTAL_BUGS})"
        
        TEMP_OUTPUT="/tmp/repgen_output_$BID.log"
        START_TIME=$(date +%s)
        
        # NOTE: Using src/tool.py for local inference
        PYTHON_ARGS=(
            "$PROJECT_DIR/src/tool.py"
            "--bug_id=$BID"
            "--max-attempts=$MAX_ATTEMPTS"
            "--retrieval_ablation=$RETRIEVAL"
            "--generation_ablation=$GENERATION"
            "--ae_dataset_path=$DATASET_PATH"
        )
        
        if [ "$QUIET" = false ]; then
            # Using python3 explicitly if possible, or python
            RUN_CMD="python"
            if command -v python3 &> /dev/null; then RUN_CMD="python3"; fi
            
            $RUN_CMD "${PYTHON_ARGS[@]}" 2>&1 | while IFS= read -r line; do echo "  ${line}"; done
            RESULT=${PIPESTATUS[0]}
        else
            RUN_CMD="python"
            if command -v python3 &> /dev/null; then RUN_CMD="python3"; fi

            $RUN_CMD "${PYTHON_ARGS[@]}" > "$TEMP_OUTPUT" 2>&1
            RESULT=$?
        fi
        
        DURATION=$(( $(date +%s) - START_TIME ))
        echo ""
        
        if [ $RESULT -eq 0 ]; then
            log_success "Bug $BID completed successfully (${DURATION}s)"
            RUN_SUCCESS=$((RUN_SUCCESS + 1))
        else
            log_error "Bug $BID failed (Exit Code: $RESULT, Duration: ${DURATION}s)"
            if [ "$QUIET" = true ]; then tail -15 "$TEMP_OUTPUT" | sed 's/^/    /'; fi
            RUN_FAILED=$((RUN_FAILED + 1))
        fi
        
        rm -f "$TEMP_OUTPUT"
        echo ""
    done
    
    show_progress "$TOTAL_BUGS" "$TOTAL_BUGS" "Pipeline Progress"
    echo ""
    log_success "Run Summary: ${BOLD}${RUN_SUCCESS}/${TOTAL_BUGS}${NC} Successful"
    echo ""
fi

# Final check
if [ "$RUN" = true ] && [ $RUN_FAILED -gt 0 ]; then exit 1; fi
exit 0