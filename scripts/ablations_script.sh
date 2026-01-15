#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================

# Default values
DEFAULT_TOOL_SCRIPT="tool_openai.py"
MAX_GEN_ATTEMPTS=5
MAX_RUN_ATTEMPTS=3

# --- 1. Define Retrieval Ablations ---
RETRIEVAL_ABLATIONS=(
    "NO_BM25"
    "NO_ANN"
    "NO_RERANKER"
    "NO_TRAINING_LOOP_EXTRACTION"
    "NO_TRAINING_LOOP_RANKING"
    "NO_MODULE_PARTITIONING"
    "NO_DEPENDENCY_EXTRACTION"
)

# --- 2. Define Generation Ablations ---
GENERATION_ABLATIONS=(
    "no_refine"
    "no_plan"
    "no_compilation"
    "no_relevance"
    "no_static_analysis"
    "no_runtime_feedback"
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

print_usage() {
    echo "Usage: $0 <start_bug_id> <end_bug_id> [--tool_script=SCRIPT] [--dataset_path=PATH]"
    echo "Example: $0 1 10 --tool_script=tool.py"
}

# Retry logic wrapper
# Usage: run_with_retry "log_file_path" command arg1 arg2 ...
run_with_retry() {
    local log_file="$1"
    shift # Shift arguments so $@ contains the command and its args
    local cmd=("$@")

    for (( attempt=1; attempt<=MAX_RUN_ATTEMPTS; attempt++ )); do
        echo "   [Attempt $attempt/$MAX_RUN_ATTEMPTS] Running: ${cmd[*]}"
        
        # Run command and redirect both stdout and stderr to log file
        "${cmd[@]}" > "$log_file" 2>&1
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            echo "   [SUCCESS] Log saved to: $log_file"
            return 0
        else
            echo "   [FAILED] Exit code $exit_code. Retrying in 2s..."
            sleep 2
        fi
    done
    
    echo "   [ERROR] Failed after $MAX_RUN_ATTEMPTS attempts. See log: $log_file"
    return 1
}

# ==========================================
# MAIN EXECUTION
# ==========================================

# 1. Parse Arguments
if [ "$#" -lt 2 ]; then
    print_usage
    exit 1
fi

START_ID=$1
END_ID=$2
shift 2

TOOL_SCRIPT="$DEFAULT_TOOL_SCRIPT"
DATASET_PATH=""

# Parse optional named arguments
for arg in "$@"; do
    case $arg in
        --tool_script=*)
        TOOL_SCRIPT="${arg#*=}"
        ;;
        --dataset_path=*)
        DATASET_PATH="${arg#*=}"
        ;;
        *)
        echo "Unknown argument: $arg"
        print_usage
        exit 1
        ;;
    esac
done

# Check if tool script exists
if [ ! -f "$TOOL_SCRIPT" ]; then
    echo "Error: Tool script '$TOOL_SCRIPT' not found."
    exit 1
fi

# 2. Main Loop
echo "Starting Ablation Study from Bug ID $START_ID to $END_ID..."
echo "Tool Script: $TOOL_SCRIPT"

for (( id=START_ID; id<=END_ID; id++ )); do
    # Format ID as 001, 002, etc.
    bug_id=$(printf "%03d" $id)
    
    echo ""
    echo "=================================================="
    echo " Processing Bug ID: $bug_id"
    echo "=================================================="

    # Create logs directory
    LOG_DIR="logs/bug_$bug_id"
    mkdir -p "$LOG_DIR"

    # --- STUDY 1: RETRIEVAL ABLATIONS ---
    echo ">>> Running Study 1: Retrieval Ablations"
    
    for abl in "${RETRIEVAL_ABLATIONS[@]}"; do
        # Handle special baseline naming if 'full_system' is used
        if [ "$abl" == "full_system" ]; then
            log_name="bug_${bug_id}_baseline_retrieval.log"
            echo " -> Running Baseline (full_system)"
        else
            log_name="bug_${bug_id}_${abl}.log"
            echo " -> Running Retrieval Ablation: $abl"
        fi
        
        log_file="$LOG_DIR/$log_name"
        
        # Build command array
        cmd=("python" "$TOOL_SCRIPT" "--bug_id=$bug_id" "--max-attempts=$MAX_GEN_ATTEMPTS" "--retrieval_ablation=$abl" "--generation_ablation=all_steps")
        
        if [ ! -z "$DATASET_PATH" ]; then
            cmd+=("--ae_dataset_path=$DATASET_PATH")
        fi

        run_with_retry "$log_file" "${cmd[@]}"
    done

    # --- STUDY 2: GENERATION ABLATIONS ---
    echo ">>> Running Study 2: Generation Ablations"
    
    # Iterate through array indices, starting from 1 to skip "all_steps" (index 0)
    for (( i=1; i<${#GENERATION_ABLATIONS[@]}; i++ )); do
        abl="${GENERATION_ABLATIONS[$i]}"
        log_name="bug_${bug_id}_${abl}.log"
        log_file="$LOG_DIR/$log_name"
        
        echo " -> Running Generation Ablation: $abl"
        
        # Build command array
        cmd=("python" "$TOOL_SCRIPT" "--bug_id=$bug_id" "--max-attempts=$MAX_GEN_ATTEMPTS" "--retrieval_ablation=full_system" "--generation_ablation=$abl")
        
        if [ ! -z "$DATASET_PATH" ]; then
            cmd+=("--ae_dataset_path=$DATASET_PATH")
        fi

        run_with_retry "$log_file" "${cmd[@]}"
    done

    # List generated logs
    echo "Logs for bug $bug_id saved in $LOG_DIR:"
    ls -lh "$LOG_DIR"
done

echo ""
echo "All processing completed."