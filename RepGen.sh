#!/bin/bash

###############################################################################
# RepGen Demo Script - Complete End-to-End Demonstration
#
# This script demonstrates the full RepGen pipeline:
#   1. Sets up environment for specified bug range
#   2. Configures API keys (OPENAI_API_KEY)
#   3. Runs local inference (Qwen2.5)
#   4. Runs cloud inference (GPT-4o)
#   5. Runs baseline experiments
#   6. Runs ablation studies
#
# Usage:
#   bash demo.sh
#
# Requirements:
#   - Python 3.8+
#   - Git
#   - Ollama (for local inference)
#   - OpenAI API key (for cloud inference)
#
# Note: Run this in a Unix-based terminal (macOS/Linux/WSL2/Git Bash)
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}==================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

print_bug_header() {
    echo -e "\n${MAGENTA}##################################################################${NC}"
    echo -e "${MAGENTA}# Processing Bug $1 of $2 (Bug ID: $3)${NC}"
    echo -e "${MAGENTA}##################################################################${NC}\n"
}

# Function to ask for execution confirmation
confirm_execution() {
    while true; do
        read -p "Do you want to execute this step? (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer y or n.";;
        esac
    done
}

# Function to validate bug ID
validate_bug_id() {
    local bug_id=$1
    if ! [[ "$bug_id" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    if [ "$bug_id" -lt 1 ] || [ "$bug_id" -gt 106 ]; then
        return 1
    fi
    return 0
}

# Get the project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Change to project directory
cd "$PROJECT_DIR"

print_header "RepGen Demo Script - Bug Range Configuration"

# ============================================================================
# Ask for Bug Range
# ============================================================================
echo "This script will run experiments on a range of bugs (1-106)."
echo ""

# Ask for start bug ID
while true; do
    read -p "Enter START bug ID (1-106): " START_BUG
    if validate_bug_id "$START_BUG"; then
        break
    else
        print_error "Invalid bug ID. Please enter a number between 1 and 106."
    fi
done

# Ask for end bug ID
while true; do
    read -p "Enter END bug ID ($START_BUG-106): " END_BUG
    if validate_bug_id "$END_BUG"; then
        if [ "$END_BUG" -lt "$START_BUG" ]; then
            print_error "End bug ID must be greater than or equal to start bug ID ($START_BUG)."
        else
            break
        fi
    else
        print_error "Invalid bug ID. Please enter a number between 1 and 106."
    fi
done

# Calculate total bugs
TOTAL_BUGS=$((END_BUG - START_BUG + 1))
BUG_RANGE="${START_BUG}-${END_BUG}"

echo ""
print_info "Configuration summary:"
echo "  - Start bug ID: $START_BUG"
echo "  - End bug ID: $END_BUG"
echo "  - Total bugs to process: $TOTAL_BUGS"
echo ""

# Confirm the configuration
while true; do
    read -p "Proceed with this configuration? (y/n): " yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) 
            print_warning "Aborting script."
            exit 0
            ;;
        * ) echo "Please answer y or n.";;
    esac
done

print_header "RepGen Demo Script - Bugs $START_BUG to $END_BUG ($TOTAL_BUGS bugs)"

# ============================================================================
# Step 1: Activate Virtual Environment
# ============================================================================
print_header "Step 1: Activating Virtual Environment"

if confirm_execution; then
    if [ -d "venv" ]; then
        print_success "Virtual environment found"
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_warning "Virtual environment not found. Creating new environment..."
        python3 -m venv venv
        source venv/bin/activate
        print_success "Virtual environment created and activated"
    fi
else
    print_warning "Skipping virtual environment setup"
fi

# ============================================================================
# Step 2: Set Up Environment for Bug Range
# ============================================================================
print_header "Step 2: Setting Up Environment for Bugs $START_BUG-$END_BUG"

echo "This will initialize the dataset, clone repositories, and prepare the environment for bugs $START_BUG-$END_BUG."
echo "Depending on your internet connection and the number of bugs, this may take several minutes..."
echo ""

if confirm_execution; then
    bash scripts/setup.sh --bugs "$BUG_RANGE"
    print_success "Environment setup complete for bugs $START_BUG-$END_BUG"
else
    print_warning "Skipping environment setup"
fi

# ============================================================================
# Step 3: Configure API Keys
# ============================================================================
print_header "Step 3: Configuring OpenAI API Key"

SKIP_CLOUD=false

if confirm_execution; then
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "OPENAI_API_KEY is not set"
        echo ""
        echo "To use cloud-based inference (GPT-4o), you need an OpenAI API key."
        echo "Get your key from: https://platform.openai.com/api-keys"
        echo ""
        read -sp "Enter your OPENAI_API_KEY (or press Enter to skip cloud demo): " API_KEY
        echo ""
        
        if [ -n "$API_KEY" ]; then
            export OPENAI_API_KEY="$API_KEY"
            print_success "OPENAI_API_KEY configured"
        else
            print_warning "Skipping cloud-based inference. Local inference will run only."
            SKIP_CLOUD=true
        fi
    else
        print_success "OPENAI_API_KEY is already set"
    fi
else
    print_warning "Skipping API key configuration"
    SKIP_CLOUD=true
fi

# ============================================================================
# Step 4-7: Run Experiments for Each Bug in Range
# ============================================================================
CURRENT_BUG_NUM=1

for BUG_NUMBER in $(seq $START_BUG $END_BUG); do
    print_bug_header $CURRENT_BUG_NUM $TOTAL_BUGS $BUG_NUMBER
    
    # ========================================================================
    # Step 4: Run Local Inference Demo (Qwen2.5)
    # ========================================================================
    print_header "Step 4: Running Local Inference Demo (Qwen2.5) - Bug $BUG_NUMBER"
    
    echo "This demo will reproduce bug $BUG_NUMBER using local Qwen2.5 inference."
    echo "Make sure Ollama is installed, and the models Qwen2.5 and Qwen2.5-Coder are available, as per the instructions in the README."
    echo ""
    
    if confirm_execution; then
        if bash scripts/quick-start/local.sh "$BUG_NUMBER" 1; then
            print_success "Local inference demo completed for bug $BUG_NUMBER"
        else
            print_warning "Local inference demo encountered an issue for bug $BUG_NUMBER"
        fi
    else
        print_warning "Skipping local inference demo for bug $BUG_NUMBER"
    fi
    
    # ========================================================================
    # Step 5: Run Cloud Inference Demo (GPT-4o)
    # ========================================================================
    print_header "Step 5: Running Cloud Inference Demo (GPT-4o) - Bug $BUG_NUMBER"
    
    if confirm_execution; then
        if [ "$SKIP_CLOUD" = true ]; then
            print_warning "Skipping cloud inference for bug $BUG_NUMBER - OPENAI_API_KEY not set"
        else
            echo "This demo will reproduce bug $BUG_NUMBER using GPT-4o via OpenAI API."
            echo ""
            
            if bash scripts/quick-start/cloud.sh "$BUG_NUMBER" 1; then
                print_success "Cloud inference demo completed for bug $BUG_NUMBER"
            else
                print_warning "Cloud inference demo encountered an issue for bug $BUG_NUMBER"
            fi
        fi
    else
        print_warning "Skipping cloud inference demo for bug $BUG_NUMBER"
    fi
    
    # ========================================================================
    # Step 6: Run Baseline Experiments
    # ========================================================================
    print_header "Step 6: Running Baseline Experiments - Bug $BUG_NUMBER"
    
    echo "This will run baseline experiments (Zero-shot, Few-shot, CoT) using local models such as Qwen2.5, Qwen2.5-Coder (via Ollama) for bug $BUG_NUMBER."
    echo "Execute this setup, if you have Ollama installed, and the Qwen2.5 and Qwen2.5-Coder models available. This may take some time..."
    echo ""
    
    if confirm_execution; then
        if bash scripts/experimental/baselines.sh --bugs "$BUG_NUMBER"; then
            print_success "Baseline experiments completed for bug $BUG_NUMBER"
        else
            print_warning "Baseline experiments encountered an issue for bug $BUG_NUMBER"
        fi
    else
        print_warning "Skipping baseline experiments for bug $BUG_NUMBER"
    fi
    
    # ========================================================================
    # Step 7: Run Ablation Studies
    # ========================================================================
    print_header "Step 7: Running Ablation Studies - Bug $BUG_NUMBER"
    
    echo "This will run ablation studies for bug $BUG_NUMBER:"
    echo "  - Retrieval ablation: NO_TRAINING_LOOP_RANKING"
    echo "  - Generation ablation: no_relevance"
    echo "  To customize the ablations, please change lines 305-306 in this script (RepGen.sh), to the desired configurations."
    echo ""
    
    if confirm_execution; then
        if bash scripts/experimental/ablations.sh \
            --bugs "$BUG_NUMBER" \
            --retrieval-ablations "NO_TRAINING_LOOP_RANKING" \
            --generation-ablations "no_relevance"; then
            print_success "Ablation studies completed for bug $BUG_NUMBER"
        else
            print_warning "Ablation studies encountered an issue for bug $BUG_NUMBER"
        fi
    else
        print_warning "Skipping ablation studies for bug $BUG_NUMBER"
    fi
    
    # Progress update
    print_success "Completed bug $BUG_NUMBER ($CURRENT_BUG_NUM of $TOTAL_BUGS)"
    CURRENT_BUG_NUM=$((CURRENT_BUG_NUM + 1))
done

# ============================================================================
# Final Summary
# ============================================================================
print_header "Demo Complete!"

echo -e "${GREEN}All requested steps for bugs $START_BUG-$END_BUG have been executed!${NC}"
echo ""
echo "Summary:"
echo "  - Total bugs processed: $TOTAL_BUGS"
echo "  - Bug range: $START_BUG to $END_BUG"
echo ""
echo "Results and logs have been saved to:"
echo "  - results/  (primary results)"
echo "  - .code_cache/  (cached code repositories)"
echo ""
echo "For more information, refer to:"
echo "  - README.md (main documentation)"
echo "  - scripts/README.md (script documentation)"
echo ""
print_success "RepGen demo script finished successfully"