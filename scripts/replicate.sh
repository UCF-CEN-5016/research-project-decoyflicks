#!/bin/bash

###############################################################################
# RepGen - ICSE'26: Automated Deep Learning Bug Reproduction
# Comprehensive Replication Script
# 
# This script sets up the environment and runs all experiments needed to 
# reproduce the results from the paper:
# "Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent"
#
# Usage: bash replicate.sh [OPTIONS]
# 
# Options:
#   --bug-start START_ID      First bug ID to process (default: 1)
#   --bug-end END_ID          Last bug ID to process (default: 106)
#   --max-attempts N          Max generation attempts per bug (default: 5)
#   --skip-setup              Skip environment setup (use if already configured)
#   --openai-api-key KEY      OpenAI API key (or set OPENAI_API_KEY env var)
#   --help                    Show this help message
#
###############################################################################

set -e  # Exit on error

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_VERSION="3.12"
BUG_START=1
BUG_END=106
MAX_ATTEMPTS=5
SKIP_SETUP=false
OPENAI_API_KEY=""

# --- Color Output ---
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    NC=''
else
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
fi

# --- Helper Functions ---
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --bug-start)
            BUG_START="$2"
            shift 2
            ;;
        --bug-end)
            BUG_END="$2"
            shift 2
            ;;
        --max-attempts)
            MAX_ATTEMPTS="$2"
            shift 2
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --openai-api-key)
            OPENAI_API_KEY="$2"
            shift 2
            ;;
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

# --- Validate Inputs ---
if ! [[ "$BUG_START" =~ ^[0-9]+$ ]] || ! [[ "$BUG_END" =~ ^[0-9]+$ ]]; then
    log_error "Bug IDs must be positive integers"
    exit 1
fi

if [[ $BUG_START -gt $BUG_END ]]; then
    log_error "BUG_START must be <= BUG_END"
    exit 1
fi

# --- Main Script ---

echo ""
echo "=========================================="
echo "  RepGen - Paper Replication Script"
echo "=========================================="
echo ""
log_info "Project directory: $PROJECT_DIR"
log_info "Processing bugs: $BUG_START to $BUG_END"
log_info "Max attempts per bug: $MAX_ATTEMPTS"
echo ""

# --- Step 1: Environment Setup ---
if [ "$SKIP_SETUP" = false ]; then
    log_info "Step 1/5: Setting up Python environment..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Please install Python $PYTHON_VERSION"
        exit 1
    fi
    
    PYTHON_VERSION_INSTALLED=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$PYTHON_VERSION_INSTALLED" != "$PYTHON_VERSION" ]]; then
        log_warning "Python version $PYTHON_VERSION_INSTALLED detected (expected $PYTHON_VERSION)"
    fi
    
    # Create virtual environment
    if [ ! -d "$PROJECT_DIR/venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv "$PROJECT_DIR/venv"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows
        if [ ! -f "$PROJECT_DIR/venv/Scripts/activate" ]; then
            log_error "Virtual environment activation script not found"
            exit 1
        fi
        source "$PROJECT_DIR/venv/Scripts/activate"
    else
        # Unix/Mac
        if [ ! -f "$PROJECT_DIR/venv/bin/activate" ]; then
            log_error "Virtual environment activation script not found"
            exit 1
        fi
        source "$PROJECT_DIR/venv/bin/activate"
    fi
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    
    # Install dependencies
    log_info "Installing dependencies from requirements.txt..."
    pip install -r "$PROJECT_DIR/requirements.txt" > /dev/null 2>&1
    
    log_success "Python environment setup complete"
else
    log_warning "Skipping environment setup"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows
        if [ -f "$PROJECT_DIR/venv/Scripts/activate" ]; then
            source "$PROJECT_DIR/venv/Scripts/activate"
        fi
    else
        # Unix/Mac
        if [ -d "$PROJECT_DIR/venv" ]; then
            source "$PROJECT_DIR/venv/bin/activate"
        fi
    fi
fi

echo ""

# --- Step 2: Environment Variables ---
log_info "Step 2/5: Configuring environment variables..."

# Set CUDA environment variables
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"

# Handle OpenAI API key
if [ -n "$OPENAI_API_KEY" ]; then
    export OPENAI_API_KEY="$OPENAI_API_KEY"
    log_success "OpenAI API key configured from argument"
elif [ -z "$OPENAI_API_KEY" ]; then
    if [ -z "${OPENAI_API_KEY+x}" ]; then
        log_warning "OPENAI_API_KEY not set. OpenAI models will not be available."
        log_info "Set it with: export OPENAI_API_KEY='your-key-here'"
    else
        log_success "Using OPENAI_API_KEY from environment"
    fi
fi

# Check for other API keys
if [ -z "${GROQ_API_KEY+x}" ]; then
    log_info "GROQ_API_KEY not set (optional for Llama models)"
else
    log_success "GROQ_API_KEY configured"
fi

if [ -z "${DEEPSEEK_API_KEY+x}" ]; then
    log_info "DEEPSEEK_API_KEY not set (optional for DeepSeek models)"
else
    log_success "DEEPSEEK_API_KEY configured"
fi

echo ""

# --- Step 3: Verify Dataset Structure & Setup Code ---
log_info "Step 3/5: Verifying dataset structure and setting up code..."

DATASET_DIR="$PROJECT_DIR/dataset"
if [ ! -d "$DATASET_DIR" ]; then
    log_error "Dataset directory not found at $DATASET_DIR"
    log_info "Please ensure the dataset directory exists"
    exit 1
fi

# Check for required dataset files and setup code directories
MISSING_BUGS=0
MISSING_CODE=0
for i in $(seq -f "%03g" "$BUG_START" "$BUG_END"); do
    BUG_DIR="$DATASET_DIR/$i"
    CODE_DIR="$BUG_DIR/code"
    
    if [ ! -d "$BUG_DIR" ]; then
        log_warning "Bug directory not found: $BUG_DIR"
        MISSING_BUGS=$((MISSING_BUGS + 1))
    else
        # Ensure code directory exists
        if [ ! -d "$CODE_DIR" ]; then
            mkdir -p "$CODE_DIR"
            MISSING_CODE=$((MISSING_CODE + 1))
        fi
        
        # Check for bug report
        if [ ! -f "$BUG_DIR/bug_report/$i.txt" ]; then
            log_warning "Bug report not found: $BUG_DIR/bug_report/$i.txt"
        fi
    fi
done

if [ $MISSING_BUGS -gt 0 ]; then
    log_warning "Found $MISSING_BUGS missing bug directories"
fi

if [ $MISSING_CODE -gt 0 ]; then
    log_info "Created $MISSING_CODE code directories"
fi

log_success "Dataset structure verification complete"

echo ""

# --- Step 4: Create Output Directories ---
log_info "Step 4/5: Preparing output directories..."

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"
# Cross-platform timestamp
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S" 2>/dev/null || echo "$(date +%s)")
else
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
fi
OUTPUT_DIR="$PROJECT_DIR/results/run_$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

log_success "Output directory: $OUTPUT_DIR"

echo ""

# --- Step 5: Run Experiments ---
log_info "Step 5/5: Running experiments..."
log_info "Starting bug reproduction pipeline..."

TOTAL_BUGS=$((BUG_END - BUG_START + 1))
PROCESSED=0
SUCCESSFUL=0
FAILED=0

for i in $(seq "$BUG_START" "$BUG_END"); do
    BUG_ID=$(printf "%03d" "$i")
    PROCESSED=$((PROCESSED + 1))
    
    echo ""
    log_info "[$PROCESSED/$TOTAL_BUGS] Processing bug $BUG_ID..."
    
    # Create bug-specific log file
    BUG_LOG="$OUTPUT_DIR/bug_${BUG_ID}.log"
    
    # Run the main tool
    if python "$PROJECT_DIR/src/tool_openai.py" \
        --bug_id="$BUG_ID" \
        --max-attempts="$MAX_ATTEMPTS" \
        --retrieval_ablation="full_system" \
        --generation_ablation="all_steps" \
        >> "$BUG_LOG" 2>&1; then
        
        log_success "Bug $BUG_ID completed successfully"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        EXIT_CODE=$?
        log_error "Bug $BUG_ID failed with exit code $EXIT_CODE (see log: $BUG_LOG)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "  Experiment Run Complete"
echo "=========================================="
echo ""
log_info "Total bugs processed: $SUCCESSFUL/$TOTAL_BUGS"
if [ $FAILED -gt 0 ]; then
    log_warning "$FAILED bugs failed (check logs in $OUTPUT_DIR)"
fi

# --- Step 6: Generate Summary Report ---
log_info "Generating summary report..."

SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
cat > "$SUMMARY_FILE" << EOF
RepGen - Experiment Run Summary
==============================
Timestamp: $(date)
Duration: $SECONDS seconds

Configuration:
  - Bug range: $BUG_START-$BUG_END
  - Max attempts: $MAX_ATTEMPTS
  - Total bugs: $TOTAL_BUGS

Results:
  - Successful: $SUCCESSFUL
  - Failed: $FAILED
  - Success rate: $(echo "scale=2; $SUCCESSFUL * 100 / $TOTAL_BUGS" | bc)%

Output directory: $OUTPUT_DIR

For detailed results of each bug, see the individual log files in this directory.
To analyze specific failures, check the corresponding bug_XXX.log file.
EOF

log_success "Summary report: $SUMMARY_FILE"

echo ""
log_success "All experiments completed!"
log_info "Results saved to: $OUTPUT_DIR"
log_info "Summary: $SUMMARY_FILE"

echo ""
echo "=========================================="
echo "  Next Steps"
echo "=========================================="
echo ""
log_info "1. Review results:"
echo "   cat $SUMMARY_FILE"
echo ""
log_info "2. Check specific bug logs:"
echo "   cat $OUTPUT_DIR/bug_001.log"
echo ""
log_info "3. View generated reproduction code:"
echo "   cat dataset/XXX/reproduction_code/XXX.py"
echo ""
log_info "4. Run statistical analysis:"
echo "   jupyter notebook $PROJECT_DIR/results/Statistical_Tests_ICSE26.ipynb"
echo ""

exit 0
