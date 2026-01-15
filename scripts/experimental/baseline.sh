#!/bin/bash

###############################################################################
# RepGen Baseline Experiments Pipeline
#
# Improvements:
# - Auto-detects backend based on model name
# - Generates summary.csv for easy analysis
# - Organized logging structure matching ablation script
# - Master log file with detailed experiment tracking
#
# Usage:
#   bash baselines.sh --bugs 1-5 [OPTIONS]
###############################################################################

set -o pipefail

# ==========================================
# 1. ENVIRONMENT & CONFIGURATION
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
BUGS=""
TOOL_SCRIPT="$PROJECT_DIR/src/baselines.py"
LOG_BASE_DIR="$PROJECT_DIR/logs/baselines_$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$LOG_BASE_DIR/summary.csv"
MASTER_LOG="$LOG_BASE_DIR/master.log"
DATASET_PATH="$PROJECT_DIR/dataset_local"
QUIET=false
EXAMPLES=3  # Default for few-shot
MAX_RUN_ATTEMPTS=1

# Global Counters
TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

# --- Models Configuration ---
MODELS=(
    "qwen2.5:7b"              # Ollama
    "qwen2.5-coder:7b"        # Ollama
    # Uncomment to add more models, make sure that you have the API keys, or the models installed locally.
    # "deepseek-r1:7b"          # Ollama
    # "llama3:8b"               # Ollama
    # "llama-3.3-70b-versatile" # Groq
    # "deepseek-reasoner"       # DeepSeek API
    # "gpt-4-turbo-2024-04-09"  # OpenAI
)

TECHNIQUES=("zero_shot" "few_shot" "cot")

# ==========================================
# 2. UI & LOGGING HELPERS
# ==========================================

# Colors - only set if terminal supports them
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    GRAY='\033[0;90m'
    BOLD='\033[1m'
    NC='\033[0m' # No Color
else
    GREEN=''
    RED=''
    BLUE=''
    YELLOW=''
    CYAN=''
    GRAY=''
    BOLD=''
    NC=''
fi

# Unified logging function - logs to both console and master log
log_to_master() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$MASTER_LOG"
}

log_header() {
    local msg="=== $1 ==="
    echo ""
    echo -e "${BOLD}${BLUE}${msg}${NC}"
    log_to_master "$msg"
}

log_info() {
    local msg="[INFO] $1"
    echo -e "${BLUE}${msg}${NC}"
    log_to_master "$msg"
}

log_warning() {
    local msg="[WARNING] $1"
    echo -e "${YELLOW}${msg}${NC}"
    log_to_master "$msg"
}

log_error() {
    local msg="[ERROR] $1"
    echo -e "${RED}${msg}${NC}" >&2
    log_to_master "$msg"
}

log_success() {
    local msg="[SUCCESS] $1"
    echo -e "${GREEN}${msg}${NC}"
    log_to_master "$msg"
}

# Spinner logic
spinner_pid=""
spinner_running=false

start_spinner() {
    [ "$QUIET" = true ] && return
    
    # Stop any existing spinner first
    stop_spinner
    
    local message="$1"
    spinner_running=true
    
    (
        local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
        local i=0
        while true; do
            i=$(( (i+1) % 10 ))
            printf "\r  %s %s" "${spin:$i:1}" "$message"
            sleep 0.1
        done
    ) &
    spinner_pid=$!
    disown
}

stop_spinner() {
    if [ "$spinner_running" = true ] && [ -n "$spinner_pid" ]; then
        kill "$spinner_pid" 2>/dev/null || true
        wait "$spinner_pid" 2>/dev/null || true
        spinner_pid=""
        spinner_running=false
        printf "\r\033[K"  # Clear the line
    fi
}

cleanup() {
    stop_spinner
    log_error "Script interrupted by user"
    echo ""
    echo -e "${RED}[ABORTED] Script interrupted.${NC}"
    exit 130
}
trap cleanup INT TERM EXIT

# Detect Python executable (handle both python and python3)
detect_python() {
    if command -v python3 &>/dev/null; then
        echo "python3"
    elif command -v python &>/dev/null; then
        echo "python"
    else
        echo "python3"  # Default fallback
    fi
}
PYTHON_CMD=$(detect_python)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

# Determine Backend Logic
get_backend() {
    local model_name=$1
    if [[ "$model_name" == *"gpt"* ]]; then
        echo "openai"
    elif [[ "$model_name" == *"versatile"* ]]; then
        echo "groq"
    elif [[ "$model_name" == *"deepseek-reasoner"* ]]; then
        echo "deepseek"
    else
        echo "ollama"
    fi
}

parse_bugs() {
    local range="$1"
    if [[ "$range" == *"-"* ]]; then
        seq "$(echo "$range" | cut -d'-' -f1)" "$(echo "$range" | cut -d'-' -f2)"
    else
        echo "$range" | tr ',' '\n'
    fi
}

# ==========================================
# 4. EXECUTION LOGIC
# ==========================================

run_baseline() {
    local bug_id="$1"
    local model="$2"
    local technique="$3"
    
    # Auto-detect backend
    local backend=$(get_backend "$model")
    
    # Sanitize model name for filename (replace : with -)
    local safe_model_name="${model//:/-}"
    local log_file="$LOG_BASE_DIR/bug_${bug_id}/${safe_model_name}_${technique}.log"
    
    local start_ts=$(date +%s)
    local status="FAIL"
    local attempt=1
    local exit_code=0
    
    # Log experiment start
    log_to_master "Starting baseline: Bug=$bug_id, Model=$model, Backend=$backend, Technique=$technique"
    log_to_master "Command: $PYTHON_CMD $TOOL_SCRIPT --bug_id=$bug_id --backend=$backend --model=$model --technique=$technique --examples=$EXAMPLES --dataset_path=$DATASET_PATH"
    log_to_master "Log file: $log_file"
    
    # Retry Loop
    for (( attempt=1; attempt<=MAX_RUN_ATTEMPTS; attempt++ )); do
        log_to_master "Attempt $attempt/$MAX_RUN_ATTEMPTS"
        
        local spinner_msg="Baseline: ${model} (${technique})"
        if [ $MAX_RUN_ATTEMPTS -gt 1 ]; then
            spinner_msg="${spinner_msg} (Attempt $attempt/$MAX_RUN_ATTEMPTS)"
        fi
        
        start_spinner "$spinner_msg"
        
        # Run Python Script with timestamped logging
        {
            echo "========================================" 
            echo "Baseline Start: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Bug ID: $bug_id"
            echo "Model: $model"
            echo "Backend: $backend"
            echo "Technique: $technique"
            echo "Examples: $EXAMPLES"
            echo "Attempt: $attempt/$MAX_RUN_ATTEMPTS"
            echo "Command: $PYTHON_CMD $TOOL_SCRIPT --bug_id=$bug_id --backend=$backend --model=$model --technique=$technique --examples=$EXAMPLES --dataset_path=$DATASET_PATH"
            echo "========================================" 
            echo ""
            $PYTHON_CMD "$TOOL_SCRIPT" \
                --bug_id "$bug_id" \
                --backend "$backend" \
                --model "$model" \
                --technique "$technique" \
                --examples "$EXAMPLES" \
                --dataset_path "$DATASET_PATH" 2>&1
            exit_code=$?
            echo ""
            echo "========================================" 
            echo "Baseline End: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Exit Code: $exit_code"
            echo "========================================" 
        } >> "$log_file" 2>&1
        
        stop_spinner

        if [ $exit_code -eq 0 ]; then
            status="PASS"
            log_to_master "Attempt $attempt: SUCCESS (exit code: $exit_code)"
            break
        else
            log_to_master "Attempt $attempt: FAILED (exit code: $exit_code)"
            if [ $attempt -lt $MAX_RUN_ATTEMPTS ]; then
                echo -e "  ${YELLOW}↻${NC} Baseline: ${model} (${technique}) - Retry $attempt"
                log_warning "Retrying in 2 seconds..."
                sleep 2
            fi
        fi
    done

    # Calculate Duration
    local end_ts=$(date +%s)
    local duration=$((end_ts - start_ts))

    # Log Result to Console and Master Log
    if [ "$status" == "PASS" ]; then
        echo -e "  ${GREEN}✓${NC} Baseline: ${model} (${technique}) ${GRAY}(${duration}s)${NC}"
        log_success "Bug $bug_id - $model/$technique: PASSED in ${duration}s"
        SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
    else
        echo -e "  ${RED}✗${NC} Baseline: ${model} (${technique}) ${GRAY}(${duration}s)${NC}"
        echo -e "    ${GRAY}→ Log: $log_file${NC}"
        log_error "Bug $bug_id - $model/$technique: FAILED after $attempt attempts (${duration}s)"
        log_error "Check log file: $log_file"
        FAILED_RUNS=$((FAILED_RUNS + 1))
    fi
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # Write to CSV
    echo "$bug_id,$model,$backend,$technique,$status,${duration}s,$exit_code,$attempt,$log_file" >> "$SUMMARY_CSV"
}

# ==========================================
# 5. ARGUMENT PARSING
# ==========================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --bugs) BUGS="$2"; shift 2 ;;
        --tool-script) TOOL_SCRIPT="$2"; shift 2 ;;
        --dataset) DATASET_PATH="$2"; shift 2 ;;
        --examples) EXAMPLES="$2"; shift 2 ;;
        --max-attempts) MAX_RUN_ATTEMPTS="$2"; shift 2 ;;
        --quiet) QUIET=true; shift ;;
        --help)
            echo "Usage: $0 --bugs 1-5 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --bugs RANGE          Bug IDs to process (e.g., '1-5' or '1,3,5')"
            echo "  --tool-script PATH    Path to baseline script (default: src/baselines.py)"
            echo "  --dataset PATH        Path to dataset (default: ae_dataset)"
            echo "  --examples NUM        Number of examples for few-shot (default: 3)"
            echo "  --max-attempts NUM    Maximum retry attempts (default: 1)"
            echo "  --quiet               Suppress spinner output"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}" >&2
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ -z "$BUGS" ]; then
    echo -e "${RED}Error: Must specify --bugs (e.g., '1-5')${NC}" >&2
    echo "Use --help for usage information"
    exit 1
fi

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

# Create log directory and initialize files
mkdir -p "$LOG_BASE_DIR"

# Initialize CSV with enhanced headers
echo "BugID,Model,Backend,Technique,Status,Duration,ExitCode,Attempts,LogFile" > "$SUMMARY_CSV"

# Initialize master log
{
    echo "========================================"
    echo "RepGen Baseline Experiments Pipeline"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
} > "$MASTER_LOG"

echo -e "${BOLD}Starting Baseline Experiments${NC}"
log_to_master "Configuration:"
echo "------------------------------------------------"
echo " Tool:         $TOOL_SCRIPT"
echo " Dataset:      $DATASET_PATH"
echo " Logs:         $LOG_BASE_DIR"
echo " Summary CSV:  $SUMMARY_CSV"
echo " Master Log:   $MASTER_LOG"
echo " Examples:     $EXAMPLES"
echo " Max Attempts: $MAX_RUN_ATTEMPTS"
echo " Models:       ${#MODELS[@]} configured"
echo "------------------------------------------------"

log_to_master "Tool: $TOOL_SCRIPT"
log_to_master "Dataset: $DATASET_PATH"
log_to_master "Examples: $EXAMPLES"
log_to_master "Max Attempts: $MAX_RUN_ATTEMPTS"
log_to_master "Models: ${MODELS[*]}"

BUG_IDS=($(parse_bugs "$BUGS"))
log_to_master "Processing bugs: ${BUG_IDS[*]}"

for bug_id in "${BUG_IDS[@]}"; do
    BID=$(printf "%03d" "$bug_id")
    BUG_LOG_DIR="$LOG_BASE_DIR/bug_$BID"
    mkdir -p "$BUG_LOG_DIR"
    
    log_header "Processing Bug $BID"
    
    for model in "${MODELS[@]}"; do
        log_info "Testing Model: $model (Backend: $(get_backend "$model"))"
        for technique in "${TECHNIQUES[@]}"; do
            run_baseline "$BID" "$model" "$technique"
        done
    done
    
    log_to_master "Completed Bug $BID"
done

# ==========================================
# 7. FINAL SUMMARY
# ==========================================

echo ""
echo "========================================"
echo -e "           ${BOLD}FINAL SUMMARY${NC}"
echo "========================================"
echo " Total Experiments: $TOTAL_RUNS"
echo -e " Successful:        ${GREEN}$SUCCESS_RUNS${NC}"
echo -e " Failed:            ${RED}$FAILED_RUNS${NC}"

echo ""
echo " CSV Report:        $SUMMARY_CSV"
echo " Master Log:        $MASTER_LOG"
echo ""

# Remove trap for normal exit
trap - EXIT

if [ $FAILED_RUNS -eq 0 ]; then
    echo -e "${GREEN}✓ All experiments completed successfully.${NC}"
    log_success "All experiments completed successfully"
    exit 0
else
    echo -e "${YELLOW}⚠ Some experiments failed. Check the logs:${NC}"
    echo "   Master Log: $MASTER_LOG"
    echo "   CSV Report: $SUMMARY_CSV"
    log_warning "Some experiments failed - check logs for details"
    exit 1
fi