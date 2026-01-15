#!/bin/bash

###############################################################################
# RepGen Unified Pipeline - Single configurable entry point (Cloud)
#
# Usage:
#   bash scripts/pipeline/cloud.sh [OPTIONS]
###############################################################################

# Ensure pipeline fails if any part of a pipe fails (catches Python errors)
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET_PATH="$PROJECT_DIR/dataset_cloud"
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

# Cleanup function for interrupts
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
            grep "^#" "$0" | head -60
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
    touch "$LOG_FILE" || {
        log_error "Cannot create log file: $LOG_FILE"
        exit 1
    }
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
BUG_START="${BUG_IDS[0]}"
BUG_END="${BUG_IDS[$((TOTAL_BUGS-1))]}"

# Print banner
echo ""
echo "=========================================="
echo -e "  ${BOLD}RepGen Unified Pipeline${NC}"
echo "=========================================="
echo ""
log_info "Dataset: ${BOLD}${DATASET_PATH}${NC}"
log_info "Bugs: ${BOLD}${BUGS}${NC} (${TOTAL_BUGS} total)"
log_info "Setup: ${BOLD}${SETUP}${NC} | Run: ${BOLD}${RUN}${NC}"
if [ "$RUN" = true ]; then
    log_info "Max attempts: ${BOLD}${MAX_ATTEMPTS}${NC}"
    log_info "Retrieval: ${BOLD}${RETRIEVAL}${NC} | Generation: ${BOLD}${GENERATION}${NC}"
fi
if [ -n "$LOG_FILE" ]; then
    log_info "Log file: ${BOLD}${LOG_FILE}${NC}"
fi
echo ""

# Extract repo info from CSV
get_repo_info() {
    local bug_id=$1
    local csv_line=$((bug_id + 1))
    
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        local line=$(python -c "
with open('$CSV_FILE', 'r') as f:
    for i, l in enumerate(f):
        if i == $csv_line:
            print(l.strip())
            break
")
    else
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
        AE_CTXDIR="$AE_BDIR/context"
        AE_REPDIR="$AE_BDIR/reproduction_code"
        
        show_progress "$CURRENT_BUG" "$TOTAL_BUGS" "Setup Progress"
        echo ""
        log_step "Processing Bug ${BOLD}${BID}${NC} (${CURRENT_BUG}/${TOTAL_BUGS})"
        log_debug "Bug ID: $BID, Original dir: $ORIG_BDIR"
        
        if [ ! -d "$ORIG_BDIR" ]; then
            log_error "  Original bug directory not found: $ORIG_BDIR"
            SETUP_FAILED=$((SETUP_FAILED + 1))
            continue
        fi
        
        log_substep "Creating directory structure"
        mkdir -p "$AE_CDIR" "$AE_BRDIR" "$AE_CTXDIR" "$AE_REPDIR"
        
        if [ -f "$ORIG_BDIR/bug_report/$BID.txt" ]; then
            log_substep "Copying bug report"
            cp "$ORIG_BDIR/bug_report/$BID.txt" "$AE_BRDIR/" || {
                log_error "  Failed to copy bug report"
                SETUP_FAILED=$((SETUP_FAILED + 1))
                continue
            }
        else
            log_warning "  Bug report not found: $ORIG_BDIR/bug_report/$BID.txt"
        fi
        
        if [ "$SKIP_CODE" = true ]; then
            log_info "  Skipping code setup (--skip-code enabled)"
            SETUP_SUCCESS=$((SETUP_SUCCESS + 1))
            continue
        fi
        
        log_substep "Reading repository info from CSV"
        INFO=$(get_repo_info "$bug_id")
        REPO=$(echo "$INFO" | cut -d'|' -f1)
        HASH=$(echo "$INFO" | cut -d'|' -f2)
        
        if [ -z "$REPO" ] || [ -z "$HASH" ]; then
            log_error "  Invalid repo info from CSV (line $((bug_id + 1)))"
            log_debug "  Repo: '$REPO', Hash: '$HASH'"
            SETUP_FAILED=$((SETUP_FAILED + 1))
            continue
        fi
        
        log_debug "  Repository: $REPO"
        log_debug "  Commit: $HASH"
        
        KEY=$(get_cache_key "$REPO")
        CACHED="$CACHE_DIR/$KEY"
        
        if [ ! -d "$CACHED" ] || [ "$FORCE_CLONE" = true ]; then
            if [ -d "$CACHED" ]; then
                log_substep "Removing old cache (--force-clone enabled)"
                rm -rf "$CACHED" || {
                    log_error "  Failed to remove old cache"
                    SETUP_FAILED=$((SETUP_FAILED + 1))
                    continue
                }
            fi
            
            log_substep "Cloning repository: ${REPO##*/}"
            start_spinner "Cloning repository..."
            
            if git clone --quiet "$REPO" "$CACHED" > /tmp/git_clone_$$.log 2>&1; then
                stop_spinner
                log_success "  Repository cloned successfully"
                log_debug "  Clone log: /tmp/git_clone_$$.log"
            else
                stop_spinner
                log_error "  Clone failed (see /tmp/git_clone_$$.log)"
                [ -n "$LOG_FILE" ] && cat /tmp/git_clone_$$.log >> "$LOG_FILE"
                SETUP_FAILED=$((SETUP_FAILED + 1))
                continue
            fi
        else
            log_substep "Using cached repository"
        fi
        
        log_substep "Checking out commit: ${HASH:0:8}"
        (
            cd "$CACHED"
            git fetch --quiet origin "$HASH" > /tmp/git_fetch_$$.log 2>&1 || true
            if git checkout --quiet "$HASH" > /tmp/git_checkout_$$.log 2>&1; then
                log_debug "  Checkout successful"
            else
                log_warning "  Checkout encountered issues (see logs)"
                [ -n "$LOG_FILE" ] && cat /tmp/git_checkout_$$.log >> "$LOG_FILE"
            fi
        ) || true
        
        log_substep "Copying code to workspace"
        rm -rf "$AE_CDIR"/*
        if cp -r "$CACHED"/* "$AE_CDIR/" 2>/dev/null; then
            log_debug "  Code copied successfully"
        else
            log_warning "  Some files may not have been copied"
        fi
        
        (cd "$CACHED" && find . -maxdepth 1 -type f -name ".*" ! -name ".git*" -exec cp {} "$AE_CDIR/" \; 2>/dev/null) || true
        
        log_success "Bug $BID setup complete"
        SETUP_SUCCESS=$((SETUP_SUCCESS + 1))
        echo ""
    done
    
    show_progress "$TOTAL_BUGS" "$TOTAL_BUGS" "Setup Progress"
    echo ""
    echo ""
    log_success "Setup phase complete: ${BOLD}${SETUP_SUCCESS}/${TOTAL_BUGS}${NC} successful"
    if [ $SETUP_FAILED -gt 0 ]; then
        log_warning "Failed: ${BOLD}${SETUP_FAILED}${NC}"
    fi
    echo ""
fi

# ============================================================================
# RUN PHASE
# ============================================================================

if [ "$RUN" = true ]; then
    echo "=========================================="
    echo -e "  ${BOLD}Run Phase${NC}"
    echo "=========================================="
    echo ""
    log_info "Starting run phase"
    
    log_step "Activating virtual environment"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        if [ -f "$PROJECT_DIR/venv/Scripts/activate" ]; then
            source "$PROJECT_DIR/venv/Scripts/activate" 2>/dev/null || {
                log_error "Failed to activate virtual environment"
                exit 1
            }
            log_success "  Virtual environment activated (Windows)"
        else
            log_error "Virtual environment not found at $PROJECT_DIR/venv/Scripts/activate"
            exit 1
        fi
    else
        if [ -d "$PROJECT_DIR/venv" ]; then
            source "$PROJECT_DIR/venv/bin/activate" 2>/dev/null || {
                log_error "Failed to activate virtual environment"
                exit 1
            }
            log_success "Virtual environment activated (Unix/Mac)"
        else
            log_error "Virtual environment not found at $PROJECT_DIR/venv"
            exit 1
        fi
    fi
    
    if [ -z "$OPENAI_API_KEY" ]; then
        log_error "OPENAI_API_KEY environment variable not set"
        log_info "Please set it with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
    
    # Validate API key format (basic check)
    if [[ ! "$OPENAI_API_KEY" =~ ^sk- ]]; then
        log_warning "API key format looks unusual (should start with 'sk-')"
    fi
    
    log_success "Environment configured successfully"
    log_debug "Python: $(which python)"
    log_debug "API Key: ...${OPENAI_API_KEY: -4}"
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
            log_error "Code directory missing for bug $BID: $CDIR"
            RUN_FAILED=$((RUN_FAILED + 1))
            continue
        fi
        
        show_progress "$CURRENT_BUG" "$TOTAL_BUGS" "Pipeline Progress"
        echo ""
        log_step "Running pipeline for Bug ${BOLD}${BID}${NC} (${CURRENT_BUG}/${TOTAL_BUGS})"
        log_debug "Bug directory: $BDIR"
        log_debug "Code directory: $CDIR"
        echo ""
        
        TEMP_OUTPUT="/tmp/repgen_output_$BID.log"
        START_TIME=$(date +%s)
        
        PYTHON_ARGS=(
            "$PROJECT_DIR/src/tool_openai.py"
            "--bug_id=$BID"
            "--max-attempts=$MAX_ATTEMPTS"
            "--retrieval_ablation=$RETRIEVAL"
            "--generation_ablation=$GENERATION"
            "--ae_dataset_path=$DATASET_PATH"
        )
        
        # Add log file if specified
        if [ -n "$LOG_FILE" ]; then
            PYTHON_ARGS+=("--log-file=$LOG_FILE")
        fi
        
        if [ "$QUIET" = false ]; then
            python "${PYTHON_ARGS[@]}" 2>&1 | while IFS= read -r line; do
                echo "  ${line}"
            done
            RESULT=${PIPESTATUS[0]}
        else
            python "${PYTHON_ARGS[@]}" > "$TEMP_OUTPUT" 2>&1
            RESULT=$?
        fi
        
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        echo ""
        
        if [ $RESULT -eq 0 ]; then
            log_success "Bug $BID completed successfully (${DURATION}s)"
            RUN_SUCCESS=$((RUN_SUCCESS + 1))
        else
            log_error "Bug $BID failed (Exit Code: $RESULT, Duration: ${DURATION}s)"
            
            if [ "$QUIET" = true ] && [ -f "$TEMP_OUTPUT" ]; then
                echo "  ${DIM}Last 15 lines of output:${NC}"
                tail -15 "$TEMP_OUTPUT" | sed 's/^/    /'
            fi
            
            # Save error log
            if [ -n "$LOG_FILE" ]; then
                echo "=== Bug $BID Error Log ===" >> "$LOG_FILE"
                cat "$TEMP_OUTPUT" >> "$LOG_FILE" 2>/dev/null || true
                echo "" >> "$LOG_FILE"
            fi
            
            RUN_FAILED=$((RUN_FAILED + 1))
        fi
        
        # Clean up temp file
        rm -f "$TEMP_OUTPUT"
        echo ""
    done
    
    show_progress "$TOTAL_BUGS" "$TOTAL_BUGS" "Pipeline Progress"
    echo ""
    echo ""
    echo "=========================================="
    echo -e "  ${BOLD}Run Summary${NC}"
    echo "=========================================="
    echo ""
    log_success "Successful: ${BOLD}${RUN_SUCCESS}/${TOTAL_BUGS}${NC}"
    if [ $RUN_FAILED -gt 0 ]; then
        log_warning "Failed: ${BOLD}${RUN_FAILED}${NC}"
    fi
    
    SUCCESS_RATE=$((RUN_SUCCESS * 100 / TOTAL_BUGS))
    log_info "Success rate: ${BOLD}${SUCCESS_RATE}%${NC}"
    
    if [ -n "$LOG_FILE" ]; then
        log_info "Full logs saved to: ${BOLD}${LOG_FILE}${NC}"
    fi
    echo ""
fi

log_success "${BOLD}Pipeline complete!${NC}"
echo ""

# Exit with error if any runs failed
if [ "$RUN" = true ] && [ $RUN_FAILED -gt 0 ]; then
    exit 1
fi

exit 0