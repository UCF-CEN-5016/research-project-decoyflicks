#!/bin/bash

###############################################################################
# RepGen Setup Script - Initialize venv, install dependencies, and datasets
#
# Usage:
#   bash setup.sh --bugs <range> [OPTIONS]
#
# Examples:
#   bash setup.sh --bugs 1-10
#   bash setup.sh --bugs 1,3,5,10-15
#   bash setup.sh --bugs 80-82 --force-clone --log-file setup.log
#
# This script will:
#   1. Create and activate a Python virtual environment
#   2. Install all dependencies from requirements.txt
#   3. Set up dataset_local with the specified bugs
#   4. Set up dataset_cloud with the specified bugs
###############################################################################

# Ensure pipeline fails if any part of a pipe fails
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$PROJECT_DIR/.code_cache"
CSV_FILE="$PROJECT_DIR/dataset/Dataset.csv"

# Default parameters
BUGS=""
SKIP_CODE=false
FORCE_CLONE=false
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
    log_warning "Setup interrupted by user"
    exit 130
}
trap cleanup INT TERM

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bugs) BUGS="$2"; shift 2 ;;
        --skip-code) SKIP_CODE=true; shift ;;
        --force-clone) FORCE_CLONE=true; shift ;;
        --quiet) QUIET=true; shift ;;
        --log-file) LOG_FILE="$2"; shift 2 ;;
        --help)
            grep "^#" "$0" | head -20
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
echo -e "  ${BOLD}RepGen Setup Script${NC}"
echo "=========================================="
echo ""
log_info "Bugs: ${BOLD}${BUGS}${NC} (${TOTAL_BUGS} total)"
log_info "Skip Code: ${BOLD}${SKIP_CODE}${NC}"
log_info "Force Clone: ${BOLD}${FORCE_CLONE}${NC}"
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

# Setup a single dataset
setup_dataset() {
    local dataset_name=$1
    local dataset_path=$2
    
    echo "=========================================="
    echo -e "  ${BOLD}Setting up ${dataset_name}${NC}"
    echo "=========================================="
    echo ""
    log_info "Dataset path: ${BOLD}${dataset_path}${NC}"
    echo ""
    
    mkdir -p "$CACHE_DIR"
    mkdir -p "$dataset_path"
    
    SETUP_SUCCESS=0
    SETUP_FAILED=0
    CURRENT_BUG=0
    
    for bug_id in "${BUG_IDS[@]}"; do
        CURRENT_BUG=$((CURRENT_BUG + 1))
        BID=$(printf "%03d" "$bug_id")
        ORIG_BDIR="$PROJECT_DIR/dataset/$BID"
        AE_BDIR="$dataset_path/$BID"
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
    
    return $SETUP_FAILED
}

# ============================================================================
# SETUP VENV AND DEPENDENCIES
# ============================================================================

setup_venv() {
    echo "=========================================="
    echo -e "  ${BOLD}Setting up Virtual Environment${NC}"
    echo "=========================================="
    echo ""
    
    VENV_DIR="$PROJECT_DIR/venv"
    REQUIREMENTS_FILE="$PROJECT_DIR/requirements.txt"
    
    # Check if venv already exists
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists at: $VENV_DIR"
        log_step "Skipping venv creation"
    else
        log_step "Creating virtual environment"
        start_spinner "Creating venv..."
        
        # Detect Python executable (handle both python and python3)
        local python_cmd="python3"
        if ! command -v python3 &>/dev/null && command -v python &>/dev/null; then
            python_cmd="python"
        fi
        
        if $python_cmd -m venv "$VENV_DIR" > /tmp/venv_create_$$.log 2>&1; then
            stop_spinner
            log_success "Virtual environment created successfully"
        else
            stop_spinner
            log_error "Failed to create virtual environment"
            [ -n "$LOG_FILE" ] && cat /tmp/venv_create_$$.log >> "$LOG_FILE"
            return 1
        fi
    fi
    
    # Determine activation script based on OS
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        ACTIVATE_SCRIPT="$VENV_DIR/Scripts/activate"
    else
        ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
    fi
    
    if [ ! -f "$ACTIVATE_SCRIPT" ]; then
        log_error "Activation script not found: $ACTIVATE_SCRIPT"
        return 1
    fi
    
    log_step "Activating virtual environment"
    source "$ACTIVATE_SCRIPT" || {
        log_error "Failed to activate virtual environment"
        return 1
    }
    log_success "Virtual environment activated"
    
    # Upgrade pip
    log_step "Upgrading pip"
    start_spinner "Upgrading pip..."
    
    if pip install --quiet --upgrade pip > /tmp/pip_upgrade_$$.log 2>&1; then
        stop_spinner
        log_success "pip upgraded successfully"
    else
        stop_spinner
        log_warning "pip upgrade encountered issues"
        [ -n "$LOG_FILE" ] && cat /tmp/pip_upgrade_$$.log >> "$LOG_FILE"
    fi
    
    # Install dependencies
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        return 1
    fi
    
    log_step "Installing dependencies from requirements.txt"
    start_spinner "Installing packages..."
    
    if pip install --quiet -r "$REQUIREMENTS_FILE" > /tmp/pip_install_$$.log 2>&1; then
        stop_spinner
        log_success "All dependencies installed successfully"
    else
        stop_spinner
        log_warning "Some dependencies may have failed to install (see logs)"
        [ -n "$LOG_FILE" ] && cat /tmp/pip_install_$$.log >> "$LOG_FILE"
    fi
    
    echo ""
    return 0
}

# ============================================================================
# MAIN SETUP
# ============================================================================

TOTAL_FAILED=0

# Setup venv and install dependencies
setup_venv
VENV_SETUP=$?

# Setup dataset_local
setup_dataset "dataset_local" "$PROJECT_DIR/dataset_local"
TOTAL_FAILED=$((TOTAL_FAILED + $?))

# Setup dataset_cloud
setup_dataset "dataset_cloud" "$PROJECT_DIR/dataset_cloud"
TOTAL_FAILED=$((TOTAL_FAILED + $?))

# Final summary
echo "=========================================="
echo -e "  ${BOLD}Setup Complete${NC}"
echo "=========================================="
echo ""

if [ $VENV_SETUP -eq 0 ] && [ $TOTAL_FAILED -eq 0 ]; then
    log_success "All setup steps completed successfully!"
    echo ""
    log_info "To activate the virtual environment, run:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        echo "    source $PROJECT_DIR/venv/Scripts/activate"
    else
        echo "    source $PROJECT_DIR/venv/bin/activate"
    fi
    echo ""
    exit 0
elif [ $VENV_SETUP -ne 0 ]; then
    log_error "Virtual environment setup failed"
    echo ""
    exit 1
else
    log_error "Setup completed with errors in dataset configuration"
    echo ""
    exit 1
fi
