# PIPELINE.md - Cloud (OpenAI)

## Overview

This guide covers the **cloud-based pipeline** using OpenAI's GPT-4 models. Use this for:
- **Fast inference** - ~30 seconds per bug
- **Small batches** - Test runs, debugging, quick validation
- **Latest models** - Access to newest OpenAI models
- **Reliable performance** - Consistent results with commercial API

**Pipeline Phases:**
- **Setup** - Clones code repositories at specific bug-related commits and organizes files
- **Run** - Executes the retrieval + planning + code generation pipeline using OpenAI API

**Trade-off:** Cost (~$0.50-1 per bug) vs Speed (~30s per bug)

---

## Prerequisites

### 1. OpenAI API Key
- Get your API key from: https://platform.openai.com/api-keys
- Keep it private; don't share in repositories
- Use `OPENAI_API_KEY` environment variable

### 2. Python 3.12+
- Check version: `python3 --version`
- Older versions may have compatibility issues

### 3. Dependencies
```bash
pip install -r requirements.txt
```

---

## Quick Start (5 minutes)

The fastest way to get started with a quick test:

```bash
# 1. Set API key
export OPENAI_API_KEY="sk-..."

# 2. Run quick-start script (auto-handles setup + run)
bash scripts/quick-start/cloud.sh 80-82 1
```

**What this does:**
- Tests on 2 bugs (80-82) with 1 retry attempt
- Uses evaluation dataset (`ae_dataset/`)
- Takes ~1-2 minutes total
- Perfect for verifying your setup works

---

## Setup Instructions

### Option 1: One-Command Setup (Recommended)
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

### Option 2: Step-by-Step Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-..."

# Verify setup
python3 -c "import openai; print('OpenAI module loaded')"
```

---

## Running the Pipeline

### Quick Test (Recommended first run)
```bash
bash scripts/quick-start/cloud.sh 80-82 1
```

### Full Control with Pipeline Script
```bash
bash scripts/pipeline/cloud.sh [OPTIONS]
```

---

## Usage Patterns

### Pattern 1: Quick Validation (1-2 minutes)
```bash
# Test on just 2 bugs
bash scripts/quick-start/cloud.sh 80-82 1
```

### Pattern 2: Small Batch Test (5-10 minutes)
```bash
# Test on 10 bugs with 1 retry
bash scripts/quick-start/cloud.sh 1-10 1

# Or with pipeline script for more control
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --setup --run
```

### Pattern 3: Full Dataset Run (30-60 minutes)
```bash
# All 106 bugs with 5 retry attempts
bash scripts/quick-start/cloud.sh 1-106 5

# Or with 2 retries for balance between cost and success
bash scripts/quick-start/cloud.sh 1-106 2
```

### Pattern 4: Setup and Run Separately
```bash
# Phase 1: Clone all repositories (one-time cost)
bash scripts/pipeline/cloud.sh --bugs 1-106 --dataset dataset --setup

# Phase 2a: Run pipeline
bash scripts/pipeline/cloud.sh --bugs 1-106 --dataset dataset --run --skip-code

# Phase 2b: Re-run on failed cases
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --run --skip-code --max-attempts 3
```

### Pattern 5: Custom Bug Selection
```bash
# Run on specific bugs
bash scripts/pipeline/cloud.sh --bugs 1,5,10,42 --dataset dataset --setup --run

# Or specific ranges and individual bugs
bash scripts/pipeline/cloud.sh --bugs 80-85,90,95-100 --dataset dataset --setup --run
```

---

## Options Reference

```bash
--bugs RANGE              # Bug IDs (REQUIRED)
                          # Formats: 1-10, 80-82, 1,5,10, or mixed 1-5,10,80-82

--dataset PATH            # Dataset location
                          # Options: dataset (default), ae_dataset, or custom path

--setup                   # Phase 1: Clone repos at specific commits
                          # Run once per dataset to prepare code

--run                     # Phase 2: Execute reproduction pipeline
                          # Runs retrieval, planning, code generation

--skip-code               # Skip cloning step; use existing code
                          # Useful for re-running on pre-setup bugs

--force-clone             # Force fresh clones even if code exists
                          # Useful if repositories got corrupted

--max-attempts N          # Retry attempts per bug (default: 1)
                          # Higher values improve success rate but increase cost

--retrieval ABLATION      # Retrieval strategy (default: full_system)
                          # For ablation studies only

--generation ABLATION     # Generation strategy (default: all_steps)
                          # For ablation studies only

--quiet                   # Suppress progress output
                          # Useful for batch runs
```

---

## Output Files

### Main Output
```
dataset/BUG_ID/
├── reproduction_code/
│   └── reproduce_BUG_ID.py      # Executable Python reproduction script
├── plan/
│   └── plan_BUG_ID.txt          # Generated execution plan / strategy
└── refined_bug_report/
    └── BUG_ID.txt               # Refined bug description with analysis
```

### Metrics & Logs
```
results/
├── ExperimentalGroup.csv        # API usage and performance metrics
├── ControlGroup.csv             # Baseline results (if available)
├── logs_TIMESTAMP.txt           # Detailed execution logs
└── Statistical_Tests_ICSE26.ipynb  # Statistical analysis
```

### To check results after run:
```bash
# View reproduction code
cat dataset/001/reproduction_code/reproduce_001.py

# View API usage metrics
cat results/ExperimentalGroup.csv

# View logs
tail -50 results/logs_*.txt
```

---

## Cost Analysis

### Pricing Breakdown
| Aspect | Estimate |
|--------|----------|
| **Per Bug** | $0.50-1.00 |
| **10 Bugs** | $5-10 |
| **50 Bugs** | $25-50 |
| **106 Bugs (Full)** | $50-100 |

**Factors affecting cost:**
- Bug complexity (larger code context = higher tokens)
- Number of retries (`--max-attempts`)
- Model version (GPT-4 Turbo is more expensive than base GPT-4)
- Context size (retrieval quality affects tokens needed)

### Cost Optimization
| Strategy | Savings |
|----------|---------|
| Test with `--bugs 80-82 1` | 95% savings (validate setup first) |
| Use `--max-attempts 1` | Baseline cost |
| Use `--max-attempts 2` | +50% cost, +30-40% success |
| Use `--max-attempts 3` | +100% cost, +50-60% success |
| Small batches first | Avoid full 106-bug spend if issues exist |

### Monitor Costs
```bash
# Check costs in real-time
cat results/ExperimentalGroup.csv | cut -d',' -f1,8,9  # Bug ID, Tokens, Cost

# Calculate total spend
awk -F',' 'NR>1 {sum += $9} END {print "Total Cost: $" sum}' results/ExperimentalGroup.csv
```

---

## Common Workflows

### Workflow 1: Quick Validation
```bash
# Takes: 1-2 minutes
# Cost: <$1
bash scripts/quick-start/cloud.sh 80-82 1
```

### Workflow 2: Debugging on Subset
```bash
# Takes: 5-10 minutes
# Cost: $5-10
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --setup --run --max-attempts 1
```

### Workflow 3: Full Paper Replication
```bash
# Takes: 45-90 minutes
# Cost: $50-100
bash scripts/quick-start/cloud.sh 1-106 2
```

### Workflow 4: High-Quality Results
```bash
# Takes: 120-180 minutes
# Cost: $75-150
bash scripts/quick-start/cloud.sh 1-106 3
```

### Workflow 5: Re-run Failures
```bash
# Takes: 5-20 minutes
# Cost: Variable
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --run --skip-code --max-attempts 5
```

---

## Troubleshooting

### Issue: API Key Error
```
Error: Invalid API key
```
**Solution:**
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY="sk-..."

# Test it
python3 -c "import os; from openai import OpenAI; client = OpenAI(api_key=os.environ['OPENAI_API_KEY']); print('✓ API key valid')"
```

### Issue: Module Not Found
```
Error: ModuleNotFoundError: No module named 'openai'
```
**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import openai; print(f'OpenAI {openai.__version__}')"
```

### Issue: Connection Timeout
```
Error: Connection timeout / API unavailable
```
**Solution:**
- Check your internet connection
- Try again in a few moments (OpenAI may have brief outages)
- Use smaller batch: `--bugs 80-82` instead of full dataset
- Check OpenAI status: https://status.openai.com

### Issue: Rate Limiting
```
Error: Rate limit exceeded
```
**Solution:**
- Wait 1-5 minutes before retrying
- Use smaller batches with delays between runs
- Contact OpenAI support to increase rate limits

### Issue: High Costs / Unexpected Charges
```
Realized spending more than expected
```
**Solution:**
- Cancel remaining runs with Ctrl+C
- Check `results/ExperimentalGroup.csv` to see what was processed
- Use `--max-attempts 1` instead of higher values
- Test on smaller subsets before full runs
- Use local Ollama pipeline for free inference ([OLLAMA_SETUP.md](OLLAMA_SETUP.md))

### Issue: Out of Memory
```
Error: out of memory / cannot allocate memory
```
**Solution:**
- Use smaller bug ranges: `--bugs 1-10` instead of `--bugs 1-106`
- Run multiple batches sequentially
- Note: Cloud API doesn't require local memory; this is usually a system issue

---

## Advanced Configuration

### Using Different OpenAI Models
Edit the Python scripts in `src/` to change model:
```python
# In src/tool.py or src/tool_openai.py
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-4-turbo-2024-04-09"  # Change this line
```

### Custom Retrieval / Generation Settings
```bash
# Use specific ablation configurations
bash scripts/pipeline/cloud.sh \
  --bugs 80-82 \
  --dataset ae_dataset \
  --setup --run \
  --retrieval full_system \
  --generation all_steps
```

---

## Logging & Debugging

### View Real-Time Logs
```bash
# During run (in another terminal)
tail -f results/logs_*.txt
```

### View Full Logs After Run
```bash
# Last 100 lines
tail -100 results/logs_*.txt

# Search for errors
grep ERROR results/logs_*.txt

# Full log
cat results/logs_*.txt
```

### Enable Verbose Output
```bash
# Remove --quiet flag for more output
bash scripts/pipeline/cloud.sh --bugs 80-82 --dataset ae_dataset --setup --run
```

---

## Next Steps

1. **First Run:** Use quick-start script to validate setup
2. **Small Batch:** Run on 10-20 bugs to verify outputs
3. **Full Run:** Execute on all 106 bugs for paper reproduction
4. **Analysis:** Check `results/` directory for metrics
5. **Local Option:** If costs are high, try [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for free runs

---

## See Also

- [scripts/README.md](scripts/README.md) - All available scripts and options
- [README.md](README.md) - Project overview
- [OLLAMA_SETUP.md](OLLAMA_SETUP.md) - Free local alternative
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - Windows-specific issues

