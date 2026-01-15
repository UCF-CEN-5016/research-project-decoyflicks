#!/bin/bash

###############################################################################
# RepGen Demo Script - Complete End-to-End Demonstration
#
# This script demonstrates the full RepGen pipeline:
#   1. Sets up environment for a randomly selected bug
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

# Get the project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Change to project directory
cd "$PROJECT_DIR"

# Randomly select a bug number between 1 and 106
BUG_NUMBER=$((1 + RANDOM % 106))
print_header "RepGen Demo Script - Bug Number $BUG_NUMBER"

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
# Step 2: Set Up Environment for Selected Bug
# ============================================================================
print_header "Step 2: Setting Up Environment for Bug $BUG_NUMBER"

echo "This will initialize the dataset, clone repositories, and prepare the environment for bug $BUG_NUMBER."
echo "Depending on your internet connection, this may take a few minutes..."
echo ""

if confirm_execution; then
    bash scripts/setup.sh --bugs "$BUG_NUMBER"
    print_success "Environment setup complete for bug $BUG_NUMBER"
else
    print_warning "Skipping environment setup"
fi

# ============================================================================
# Step 3: Configure API Keys
# ============================================================================
print_header "Step 3: Configuring OpenAI API Key"

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
        fi
    else
        print_success "OPENAI_API_KEY is already set"
    fi
else
    print_warning "Skipping API key configuration"
fi

# ============================================================================
# Step 4: Run Local Inference Demo (Qwen2.5)
# ============================================================================
print_header "Step 4: Running Local Inference Demo (Qwen2.5 via Ollama)"

echo "This demo will reproduce bug $BUG_NUMBER using local Qwen2.5 inference."
echo "Make sure Ollama is running: ollama serve (in a separate terminal)"
echo ""

if confirm_execution; then
    if bash scripts/quick-start/local.sh "$BUG_NUMBER" 1; then
        print_success "Local inference demo completed"
    else
        print_warning "Local inference demo encountered an issue"
    fi
else
    print_warning "Skipping local inference demo"
fi

# ============================================================================
# Step 5: Run Cloud Inference Demo (GPT-4o)
# ============================================================================
print_header "Step 5: Running Cloud Inference Demo (GPT-4o)"

if confirm_execution; then
    if [ -z "$OPENAI_API_KEY" ]; then
        print_warning "Skipping cloud inference - OPENAI_API_KEY not set"
    else
        echo "This demo will reproduce bug $BUG_NUMBER using GPT-4o via OpenAI API."
        echo ""
        
        if bash scripts/quick-start/cloud.sh "$BUG_NUMBER" 1; then
            print_success "Cloud inference demo completed"
        else
            print_warning "Cloud inference demo encountered an issue"
        fi
    fi
else
    print_warning "Skipping cloud inference demo"
fi

# ============================================================================
# Step 6: Run Baseline Experiments
# ============================================================================
print_header "Step 6: Running Baseline Experiments"

echo "This will run baseline experiments (Zero-shot, Few-shot, CoT) for bug $BUG_NUMBER."
echo "This may take some time..."
echo ""

if confirm_execution; then
    if bash scripts/pipeline/baselines.sh --bugs "$BUG_NUMBER"; then
        print_success "Baseline experiments completed"
    else
        print_warning "Baseline experiments encountered an issue"
    fi
else
    print_warning "Skipping baseline experiments"
fi

# ============================================================================
# Step 7: Run Ablation Studies
# ============================================================================
print_header "Step 7: Running Ablation Studies"

echo "This will run ablation studies for bug $BUG_NUMBER:"
echo "  - Retrieval ablation: NO_TRAINING_LOOP_RANKING"
echo "  - Generation ablation: no_relevance"
echo ""

if confirm_execution; then
    if bash scripts/experimental/ablations.sh \
        --bugs "$BUG_NUMBER" \
        --retrieval-ablations "NO_TRAINING_LOOP_RANKING" \
        --generation-ablations "no_relevance"; then
        print_success "Ablation studies completed"
    else
        print_warning "Ablation studies encountered an issue"
    fi
else
    print_warning "Skipping ablation studies"
fi

# ============================================================================
# Final Summary
# ============================================================================
print_header "Demo Complete!"

echo -e "${GREEN}All requested steps for bug $BUG_NUMBER have been executed!${NC}"
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