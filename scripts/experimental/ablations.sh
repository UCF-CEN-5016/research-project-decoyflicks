#!/bin/bash

###############################################################################
# RepGen Ablation Study Pipeline
#
# Improvements:
# - Generates a persistent CSV summary (logs/ablation_.../summary.csv)
# - Tracks and displays duration for each experiment
# - Clear visual feedback (Pass/Fail/Retry)
# - Custom ablation selection via command-line arguments
#
# Usage:
#   bash ablations.sh --bugs 1-5 [OPTIONS]
#   bash ablations.sh --bugs 1-5 \
#     --retrieval-ablations "NO_TRAINING_LOOP_RANKING" \
#     --generation-ablations "no_relevance"
###############################################################################

set -o pipefail

# ==========================================
# 0. PATH HELPERS
# ==========================================

# Platform-independent path handling
normalize_path() {
    local path="$1"
    if command -v cygpath >/dev/null 2>&1; then
        # Convert '/c/Users' to 'C:/Users' (mixed mode)
        cygpath -m "$path"
    else
        echo "$path"
    fi
}

# ==========================================
# 1. ENVIRONMENT & CONFIGURATION
# ==========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_DIR="$(normalize_path "$RAW_PROJECT_DIR")"

# Default values
BUGS=""
TOOL_SCRIPT="$PROJECT_DIR/src/tool_openai.py"
DATASET_PATH="$PROJECT_DIR/dataset_cloud"
MAX_GEN_ATTEMPTS=1
MAX_RUN_ATTEMPTS=1
LOG_BASE_DIR="$PROJECT_DIR/logs/ablation_study_$(date +%Y%m%d_%H%M%S)"
SUMMARY_CSV="$LOG_BASE_DIR/summary.csv"
MASTER_LOG="$LOG_BASE_DIR/master.log"
QUIET=false
CUSTOM_RETRIEVAL_ABLATIONS=""
CUSTOM_GENERATION_ABLATIONS=""

# Global Counters
TOTAL_RUNS=0
SUCCESS_RUNS=0
FAILED_RUNS=0

# --- Define Retrieval Ablations (defaults) ---
RETRIEVAL_ABLATIONS=(
    "NO_BM25"
    "NO_ANN"
    "NO_RERANKER"
    "NO_TRAINING_LOOP_EXTRACTION"
    "NO_TRAINING_LOOP_RANKING"
    "NO_MODULE_PARTITIONING"
    "NO_DEPENDENCY_EXTRACTION"
)

# --- Define Generation Ablations (defaults) ---
GENERATION_ABLATIONS=(
    "no_refine"
    "no_plan"
    "no_compilation"
    "no_relevance"
    "no_static_analysis"
    "no_runtime_feedback"
)

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

# ==========================================
# 3. CORE LOGIC
# ==========================================

run_experiment() {
    local bug_id="$1"
    local type="$2"       # "Retrieval" or "Generation"
    local ablation="$3"
    local log_file="$4"
    local cmd_args=("${@:5}") # Remaining args are the command

    local start_ts=$(date +%s)
    local status="FAIL"
    local attempt=1
    local exit_code=0
    
    # Log experiment start
    log_to_master "Starting experiment: Bug=$bug_id, Type=$type, Ablation=$ablation"
    log_to_master "Command: python ${cmd_args[*]}"
    log_to_master "Log file: $log_file"
    
    # Retry Loop
    for (( attempt=1; attempt<=MAX_RUN_ATTEMPTS; attempt++ )); do
        log_to_master "Attempt $attempt/$MAX_RUN_ATTEMPTS"
        
        local spinner_msg="${type}: ${ablation}"
        if [ $MAX_RUN_ATTEMPTS -gt 1 ]; then
            spinner_msg="${spinner_msg} (Attempt $attempt/$MAX_RUN_ATTEMPTS)"
        fi
        
        start_spinner "$spinner_msg"
        
        # Run Python Script with timestamped logging
        {
            echo "========================================" 
            echo "Experiment Start: $(date '+%Y-%m-%d %H:%M:%S')"
            echo "Bug ID: $bug_id"
            echo "Type: $type"
            echo "Ablation: $ablation"
            echo "Attempt: $attempt/$MAX_RUN_ATTEMPTS"
            echo "Command: python ${cmd_args[*]}"
            echo "========================================" 
            echo ""
            python "${cmd_args[@]}" 2>&1
            exit_code=$?
            echo ""
            echo "========================================" 
            echo "Experiment End: $(date '+%Y-%m-%d %H:%M:%S')"
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
                echo -e "  ${YELLOW}↻${NC} ${type}: ${ablation} - Retry $attempt"
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
        echo -e "  ${GREEN}✓${NC} ${type}: ${ablation} ${GRAY}(${duration}s)${NC}"
        log_success "Bug $bug_id - $type/$ablation: PASSED in ${duration}s"
        SUCCESS_RUNS=$((SUCCESS_RUNS + 1))
    else
        echo -e "  ${RED}✗${NC} ${type}: ${ablation} ${GRAY}(${duration}s)${NC}"
        echo -e "    ${GRAY}→ Log: $log_file${NC}"
        log_error "Bug $bug_id - $type/$ablation: FAILED after $attempt attempts (${duration}s)"
        log_error "Check log file: $log_file"
        FAILED_RUNS=$((FAILED_RUNS + 1))
    fi
    
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # Log Result to CSV
    echo "$bug_id,$type,$ablation,$status,${duration}s,$exit_code,$attempt,$log_file" >> "$SUMMARY_CSV"
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
# 4. ARGUMENT PARSING
# ==========================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --bugs) BUGS="$2"; shift 2 ;;
        --tool-script) TOOL_SCRIPT="$2"; shift 2 ;;
        --dataset) DATASET_PATH="$(normalize_path "$2")"; shift 2 ;;
        --max-attempts) MAX_RUN_ATTEMPTS="$2"; shift 2 ;;
        --quiet) QUIET=true; shift ;;
        --retrieval-ablations) CUSTOM_RETRIEVAL_ABLATIONS="$2"; shift 2 ;;
        --generation-ablations) CUSTOM_GENERATION_ABLATIONS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 --bugs 1-5 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --bugs RANGE                    Bug IDs to process (e.g., '1-5' or '1,3,5')"
            echo "  --tool-script PATH              Path to tool script (default: src/tool_openai.py)"
            echo "  --dataset PATH                  Path to dataset (default: dataset_cloud)"
            echo "  --max-attempts NUM              Maximum retry attempts (default: 1)"
            echo "  --quiet                         Suppress spinner output"
            echo "  --retrieval-ablations LIST      Comma-separated retrieval ablations to run"
            echo "                                  (e.g., 'NO_BM25,NO_RERANKER')"
            echo "  --generation-ablations LIST     Comma-separated generation ablations to run"
            echo "                                  (e.g., 'no_refine,no_plan')"
            echo "  --help                          Show this help message"
            echo ""
            echo "Available Retrieval Ablations:"
            echo "  NO_BM25, NO_ANN, NO_RERANKER, NO_TRAINING_LOOP_EXTRACTION,"
            echo "  NO_TRAINING_LOOP_RANKING, NO_MODULE_PARTITIONING, NO_DEPENDENCY_EXTRACTION"
            echo ""
            echo "Available Generation Ablations:"
            echo "  no_refine, no_plan, no_compilation, no_relevance,"
            echo "  no_static_analysis, no_runtime_feedback"
            echo ""
            echo "Examples:"
            echo "  # Run all ablations for bugs 1-5"
            echo "  $0 --bugs 1-5"
            echo ""
            echo "  # Run specific ablations only"
            echo "  $0 --bugs 1-5 \\"
            echo "    --retrieval-ablations 'NO_TRAINING_LOOP_RANKING' \\"
            echo "    --generation-ablations 'no_relevance'"
            echo ""
            echo "  # Run multiple custom ablations"
            echo "  $0 --bugs 1-3 \\"
            echo "    --retrieval-ablations 'NO_BM25,NO_RERANKER' \\"
            echo "    --generation-ablations 'no_refine,no_plan,no_relevance'"
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
# 5. APPLY CUSTOM ABLATIONS IF PROVIDED
# ==========================================

if [ -n "$CUSTOM_RETRIEVAL_ABLATIONS" ]; then
    IFS=',' read -ra RETRIEVAL_ABLATIONS <<< "$CUSTOM_RETRIEVAL_ABLATIONS"
    log_info "Using custom retrieval ablations: ${RETRIEVAL_ABLATIONS[*]}"
fi

if [ -n "$CUSTOM_GENERATION_ABLATIONS" ]; then
    IFS=',' read -ra GENERATION_ABLATIONS <<< "$CUSTOM_GENERATION_ABLATIONS"
    log_info "Using custom generation ablations: ${GENERATION_ABLATIONS[*]}"
fi

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

# Create log directory and initialize files
mkdir -p "$LOG_BASE_DIR"

# Initialize CSV with enhanced headers
echo "BugID,Type,Ablation,Status,Duration,ExitCode,Attempts,LogFile" > "$SUMMARY_CSV"

# Initialize master log
{
    echo "========================================"
    echo "RepGen Ablation Study Pipeline"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
} > "$MASTER_LOG"

echo -e "${BOLD}Starting Ablation Study${NC}"
log_to_master "Configuration:"
echo "------------------------------------------------"
echo " Tool:         $TOOL_SCRIPT"
echo " Dataset:      $DATASET_PATH"
echo " Logs:         $LOG_BASE_DIR"
echo " Summary CSV:  $SUMMARY_CSV"
echo " Master Log:   $MASTER_LOG"
echo " Max Attempts: $MAX_RUN_ATTEMPTS"
echo "------------------------------------------------"
echo " Retrieval Ablations: ${RETRIEVAL_ABLATIONS[*]}"
echo " Generation Ablations: ${GENERATION_ABLATIONS[*]}"
echo "------------------------------------------------"

log_to_master "Tool: $TOOL_SCRIPT"
log_to_master "Dataset: $DATASET_PATH"
log_to_master "Max Attempts: $MAX_RUN_ATTEMPTS"
log_to_master "Retrieval Ablations: ${RETRIEVAL_ABLATIONS[*]}"
log_to_master "Generation Ablations: ${GENERATION_ABLATIONS[*]}"

BUG_IDS=($(parse_bugs "$BUGS"))
log_to_master "Processing bugs: ${BUG_IDS[*]}"

for bug_id in "${BUG_IDS[@]}"; do
    BID=$(printf "%03d" "$bug_id")
    BUG_LOG_DIR="$LOG_BASE_DIR/bug_$BID"
    mkdir -p "$BUG_LOG_DIR"
    
    log_header "Processing Bug $BID"

    # --- 1. RETRIEVAL ABLATIONS ---
    if [ ${#RETRIEVAL_ABLATIONS[@]} -gt 0 ]; then
        log_info "Running Retrieval Ablations"
        for abl in "${RETRIEVAL_ABLATIONS[@]}"; do
            log_file="$BUG_LOG_DIR/retrieval_${abl}.log"
            
            # Args for this run
            args=("$TOOL_SCRIPT" "--bug_id=$BID" "--max-attempts=$MAX_GEN_ATTEMPTS" "--retrieval_ablation=$abl" "--generation_ablation=all_steps")
            [ -n "$DATASET_PATH" ] && args+=("--ae_dataset_path=$DATASET_PATH")

            run_experiment "$BID" "Retrieval" "$abl" "$log_file" "${args[@]}"
        done
    else
        log_info "No retrieval ablations specified, skipping..."
    fi

    # --- 2. GENERATION ABLATIONS ---
    if [ ${#GENERATION_ABLATIONS[@]} -gt 0 ]; then
        echo ""
        log_info "Running Generation Ablations"
        for abl in "${GENERATION_ABLATIONS[@]}"; do
            log_file="$BUG_LOG_DIR/generation_${abl}.log"
            
            # Args for this run (Retrieval fixed to full_system)
            args=("$TOOL_SCRIPT" "--bug_id=$BID" "--max-attempts=$MAX_GEN_ATTEMPTS" "--retrieval_ablation=full_system" "--generation_ablation=$abl")
            [ -n "$DATASET_PATH" ] && args+=("--ae_dataset_path=$DATASET_PATH")

            run_experiment "$BID" "Generation" "$abl" "$log_file" "${args[@]}"
        done
    else
        log_info "No generation ablations specified, skipping..."
    fi
    
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