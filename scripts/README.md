# RepGen Scripts - Detailed Configuration Guide

This directory contains shell scripts for running the RepGen bug reproduction pipeline. Scripts are organized into three categories based on their purpose and complexity level. This guide covers **detailed configurations** and **customization options**.

---

## Directory Structure

```
scripts/
├── quick-start/       # Entry point for first-time users (simplified interface)
│   ├── cloud.sh       # Quick test with OpenAI API (2 positional args)
│   └── local.sh       # Quick test with local Ollama (2 positional args)
├── pipeline/          # Full pipeline with advanced control (named flags)
│   ├── cloud.sh       # Full cloud-based pipeline with all options
│   └── local.sh       # Full local-based pipeline with all options
└── experimental/      # Ablation studies and baseline comparisons
    ├── ablations.sh   # Systematic ablation experiments
    └── baseline.sh    # Multi-model baseline comparisons
```

---

## Quick Selection Guide

**New to RepGen?** → Use `quick-start/` scripts  
**Need more control?** → Use `pipeline/` scripts  
**Running baselines or ablations?** → Use `experimental/` scripts  

---

# QUICK START SCRIPTS

Entry point for first-time users. Minimal configuration required.

## `quick-start/cloud.sh` - Cloud Quick Start

One-command execution using OpenAI API with sensible defaults.

### Usage
```bash
bash scripts/quick-start/cloud.sh [BUGS] [MAX_ATTEMPTS] [OPTIONS]
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `BUGS` | `1-10` | Bug IDs to process. Format: `1-10`, `80-82`, or `1,5,10` |
| `MAX_ATTEMPTS` | `5` | Retry attempts per bug if generation fails |
| `OPTIONS` | (none) | Additional flags: `--quiet` |

### Examples

```bash
# Basic: Test on 2 bugs
bash scripts/quick-start/cloud.sh 80-82 1

# Full paper replication: All 106 bugs with 5 retry attempts
bash scripts/quick-start/cloud.sh 1-106 5

# With minimal output
bash scripts/quick-start/cloud.sh 1-10 3 --quiet

# Specific bug IDs
bash scripts/quick-start/cloud.sh 1,5,10 1
```

### Prerequisites
- Python 3.12+
- Virtual environment activated with `pip install -r requirements.txt`
- OpenAI API key set: `export OPENAI_API_KEY="sk-..."`
- Network access for OpenAI API calls

### Output
```
dataset/BUG_ID/
├── reproduction_code/
│   └── reproduce_BUG_ID.py
├── plan/
│   └── plan_BUG_ID.txt
└── refined_bug_report/
    └── BUG_ID.txt
```

### Performance Characteristics
- **Speed:** ~30 seconds per bug
- **Cost:** ~$0.50-1.00 per bug ($50-100 for all 106)
- **Parallelization:** Sequential (use multiple machines for parallel runs)
- **Dataset:** Uses `dataset_cloud/` (evaluation subset)

### Customization
This script wraps `pipeline/cloud.sh` internally. For more control, use the pipeline script directly.

---

## `quick-start/local.sh` - Local Quick Start (Ollama)

One-command execution using local Ollama models. Free, no API costs.

### Usage
```bash
bash scripts/quick-start/local.sh [BUGS] [MAX_ATTEMPTS]
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `BUGS` | `80-82` | Bug IDs to process. Format: `1-10`, `80-82`, or `1,5,10` |
| `MAX_ATTEMPTS` | `1` | Retry attempts per bug |

### Examples

```bash
# Basic: Test on 2 bugs
bash scripts/quick-start/local.sh 80-82 1

# Full paper replication with retries
bash scripts/quick-start/local.sh 1-106 3

# Specific bug IDs
bash scripts/quick-start/local.sh 1,5,10,42 1
```

### Prerequisites
- Ollama installed and running: `ollama serve` (in separate terminal)
- Required models: `ollama pull qwen2.5:7b qwen2.5-coder:7b`
- Python 3.12+
- Virtual environment activated with `pip install -r requirements.txt`
- 16GB+ RAM (7GB minimum per model, shared)
- ~20GB disk space for models

### Output
Same as cloud script (see above)

### Performance Characteristics
- **Speed:** 3-5 minutes per bug (CPU-only; faster with GPU)
- **Cost:** Free (after model download ~2.5 hours)
- **Parallelization:** Sequential
- **Dataset:** Uses `dataset_cloud/`

### GPU Acceleration
For faster execution on systems with NVIDIA GPUs or Apple Silicon:
```bash
# The scripts automatically detect and use CUDA/Metal if available
# Verify with: ollama list
# Performance: ~1-2 minutes per bug with GPU
```

---

# PIPELINE SCRIPTS

Advanced control for custom workflows. Use these for fine-grained configuration.

## `pipeline/cloud.sh` - Cloud Pipeline (Full Control)

Customizable cloud pipeline with setup/run separation and detailed options.

### Usage
```bash
bash scripts/pipeline/cloud.sh [OPTIONS]
```

### Core Options

#### Bug Selection (REQUIRED - one must be specified)
```bash
--bugs RANGE              # Bug IDs: 1-10, 80-82, 80,81,82, or 1,3,5,10-15
```

Examples:
```bash
--bugs 1-10              # Range: bugs 1 through 10
--bugs 80-82             # Range: bugs 80 through 82
--bugs 1,5,10            # Specific IDs (comma-separated, no spaces)
--bugs 1-10,20,25-30     # Mixed: ranges and specific IDs
```

#### Execution Control (specify at least one)
```bash
--setup                  # Phase 1: Clone repositories at bug-specific commits
--run                    # Phase 2: Execute retrieval, planning, and generation
```

#### Dataset Selection
```bash
--dataset PATH           # Default: dataset
                         # Options: dataset (all 106) or ae_dataset (subset)
```

#### Code Management
```bash
--skip-code              # Use existing cloned repositories (skip cloning)
--force-clone            # Force fresh clones even if code exists
```

#### Retry Configuration
```bash
--max-attempts N         # Default: 1
                         # Retries if generation fails
                         # Recommended: 2-3 for high success rate
```

#### Ablation/Customization
```bash
--retrieval STRATEGY     # Default: full_system
                         # Retrieval component configuration

--generation STRATEGY    # Default: all_steps
                         # Generation component configuration
```

#### Output Control
```bash
--quiet                  # Suppress progress output (still logs to file)
--log-file PATH          # Save logs to specific file
                         # Default: results/logs_TIMESTAMP.txt
```

### Examples

#### Basic Execution
```bash
# Setup only (clone code, don't run generation)
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --setup

# Run only (use existing code)
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --run --skip-code

# Full pipeline (setup + run)
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --setup --run
```

#### Advanced Workflows
```bash
# Full replication with retries
bash scripts/pipeline/cloud.sh \
  --bugs 1-106 \
  --dataset dataset \
  --setup --run \
  --max-attempts 3

# Specific bugs with logging
bash scripts/pipeline/cloud.sh \
  --bugs 1,5,10,42,80-85 \
  --dataset dataset \
  --setup --run \
  --log-file results/my_run.log

# Ablation configuration
bash scripts/pipeline/cloud.sh \
  --bugs 80-82 \
  --dataset ae_dataset \
  --setup --run \
  --retrieval full_system \
  --generation all_steps

# Re-run failed bugs
bash scripts/pipeline/cloud.sh \
  --bugs 15-25 \
  --dataset dataset \
  --run --skip-code \
  --max-attempts 3
```

#### Debugging
```bash
# Minimal output
bash scripts/pipeline/cloud.sh --bugs 80-82 --setup --run --quiet

# With custom logging
bash scripts/pipeline/cloud.sh \
  --bugs 1-5 \
  --setup --run \
  --log-file results/debug.log

# Force fresh clones (helpful if cache is corrupted)
bash scripts/pipeline/cloud.sh \
  --bugs 1-5 \
  --setup --run \
  --force-clone
```

### Implementation Details

#### Default Paths
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATASET_PATH="${PROJECT_DIR}/dataset"
CACHE_DIR="${PROJECT_DIR}/.code_cache"
```

#### Logging
- **Console output:** Real-time colored progress indicators
- **Log files:** `${PROJECT_DIR}/results/logs_YYYYMMDD_HHMMSS.txt`
- **Log levels:** INFO, SUCCESS, WARNING, ERROR, DEBUG
- **Timestamp format:** `YYYY-MM-DD HH:MM:SS`

#### Progress Indication
- Automatically detects terminal capabilities (colors, Unicode symbols)
- Falls back to plain text on Windows CMD (supports colors in Git Bash/WSL)
- Spinner for long operations, progress bar for batch processing
- `--quiet` mode disables progress display (logs still written)

### Cost Management

**Estimate costs before running:**
```bash
# Cost calculation
Bugs × Average_cost_per_bug × max_attempts
```

| Scenario | Bugs | Attempts | Est. Cost |
|----------|------|----------|-----------|
| Quick test | 10 | 1 | $5-10 |
| Validation | 50 | 1 | $25-50 |
| Full replication | 106 | 5 | $250-500 |
| Optimization run | 106 | 3 | $150-300 |

**Tips to reduce costs:**
- Test with `--bugs 80-82` first (2 bugs, ~$1-2)
- Use `--max-attempts 1` for initial runs
- Use `--skip-code` for re-runs (only generation cost)
- Switch to local Ollama pipeline once validated

---

## `pipeline/local.sh` - Local Pipeline (Ollama, Full Control)

Full customizable pipeline using local Ollama models. Same options as cloud, but uses local models.

### Usage
```bash
bash scripts/pipeline/local.sh [OPTIONS]
```

### Options
Identical to `pipeline/cloud.sh` (see above). Examples:

```bash
# Setup only
bash scripts/pipeline/local.sh --bugs 1-10 --setup

# Run with retries
bash scripts/pipeline/local.sh --bugs 1-10 --run --skip-code --max-attempts 3

# Full pipeline
bash scripts/pipeline/local.sh --bugs 1-106 --dataset dataset --setup --run
```

### Prerequisites
- Ollama running: `ollama serve` (separate terminal)
- Models: `ollama pull qwen2.5:7b qwen2.5-coder:7b`
- Python 3.12+, dependencies installed
- 16GB+ RAM, ~20GB disk space

### Performance Difference
- **Cloud:** ~30 sec/bug (constant speed)
- **Local CPU:** ~3-5 min/bug (varies by hardware)
- **Local GPU:** ~1-2 min/bug (NVIDIA/Apple Silicon)

### When to Use Each
- **Cloud:** Fast validation, small batches, guaranteed performance
- **Local:** Large batches, privacy required, budget-conscious, offline work

---

# EXPERIMENTAL SCRIPTS

Advanced scripts for systematic experiments and comparisons.

## `experimental/ablations.sh` - Ablation Studies

Run systematic ablation experiments to measure individual component impact.

### Usage
```bash
bash scripts/experimental/ablations.sh <start_bug_id> <end_bug_id> [OPTIONS]
```

### Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `start_bug_id` | Yes | First bug ID (e.g., `1`, `80`) |
| `end_bug_id` | Yes | Last bug ID (e.g., `10`, `82`) |

### Options
```bash
--tool_script=SCRIPT      # Python tool to use
                          # Default: ${PROJECT_DIR}/src/tool_openai.py
                          # Options: tool.py, tool_openai.py

--dataset_path=PATH       # Dataset path
                          # Default: ${PROJECT_DIR}/dataset
```

### Examples

```bash
# Basic ablation on bugs 1-5
bash scripts/experimental/ablations.sh 1 5

# On evaluation subset
bash scripts/experimental/ablations.sh 80 82

# With custom tool script
bash scripts/experimental/ablations.sh 1 10 --tool_script=tool.py

# Custom dataset
bash scripts/experimental/ablations.sh 1 5 --dataset_path=ae_dataset
```

### Ablation Configurations

The script runs two types of ablations:

#### Retrieval Ablations (Component Impact)
Measures impact of removing each retrieval component:

| Ablation | Impact Measured |
|----------|-----------------|
| `NO_BM25` | Basic search functionality |
| `NO_ANN` | Approximate nearest neighbor ranking |
| `NO_RERANKER` | Re-ranking module |
| `NO_TRAINING_LOOP_EXTRACTION` | Training loop detection |
| `NO_TRAINING_LOOP_RANKING` | Training loop ranking priority |
| `NO_MODULE_PARTITIONING` | Module partitioning logic |
| `NO_DEPENDENCY_EXTRACTION` | Dependency extraction |

#### Generation Ablations (Generation Strategy Impact)
Measures impact of removing each generation step:

| Ablation | Impact Measured |
|----------|-----------------|
| `no_refine` | Iterative refinement |
| `no_plan` | Planning phase |
| `no_compilation` | Static compilation checks |
| `no_relevance` | Relevance filtering |
| `no_static_analysis` | Static code analysis |
| `no_runtime_feedback` | Runtime feedback loops |

### Output Structure
```
logs/ablation_study_YYYYMMDD_HHMMSS/
├── master.log                           # Combined log
├── summary.csv                          # Results summary
└── ablation_${TYPE}_${NAME}/
    ├── bug_${ID}/
    │   ├── generation.log
    │   ├── execution.log
    │   └── results.json
    └── ...
```

### Output Format (summary.csv)
```csv
ablation_type,ablation_name,bug_id,status,generation_time,execution_time,tokens_used,cost
retrieval,NO_BM25,1,SUCCESS,15.3,2.1,4520,$0.045
retrieval,NO_ANN,1,FAILED,12.8,0.5,3100,$0.031
generation,no_plan,1,SUCCESS,18.5,1.9,5200,$0.052
...
```

### Cost Estimate
```bash
# Cost per bug (full ablations)
13 ablations × cost_per_bug ≈ $6.50-13 per bug
```

For bugs 1-10: ~$65-130 total

---

## `experimental/baseline.sh` - Baseline Comparisons

Compare performance across multiple models and prompting techniques.

### Usage
```bash
bash scripts/experimental/baseline.sh <start_bug_id> <end_bug_id> [OPTIONS]
```

### Arguments
| Argument | Required | Description |
|----------|----------|-------------|
| `start_bug_id` | Yes | First bug ID |
| `end_bug_id` | Yes | Last bug ID |

### Examples

```bash
# Compare on bugs 1-5
bash scripts/experimental/baseline.sh 1 5

# On evaluation subset
bash scripts/experimental/baseline.sh 80 82
```

### Model Configurations

#### Ollama Models (Local)
Requires: `ollama serve` running

```bash
qwen2.5:7b               # Default Qwen model
qwen2.5-coder:7b         # Specialized for code
deepseek-r1:7b           # Reasoning-focused
llama3:8b                # Meta's Llama
```

Install with: `ollama pull MODEL_NAME`

#### Groq Models (API)
Requires: `GROQ_API_KEY` environment variable

```bash
llama-3.3-70b-versatile  # Fast inference
```

#### DeepSeek Models (API)
Requires: `DEEPSEEK_API_KEY` environment variable

```bash
deepseek-reasoner        # Reasoning model
```

#### OpenAI Models (API)
Requires: `OPENAI_API_KEY` environment variable

```bash
gpt-4-turbo-2024-04-09   # Latest GPT-4 Turbo
```

### Prompting Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| `zero_shot` | No examples, direct instruction | Baseline |
| `few_shot` | 2-3 examples included | Better accuracy |
| `cot` | Chain-of-thought reasoning | Complex problems |

### Output Structure
```
results/baselines_YYYYMMDD_HHMMSS/
├── summary.csv
├── model_${MODEL_NAME}/
│   ├── technique_${TECHNIQUE}/
│   │   ├── bug_${ID}/
│   │   │   ├── generation.log
│   │   │   ├── result.json
│   │   │   └── reproduction_code.py
│   │   └── ...
│   └── ...
└── comparison_matrix.csv
```

### Results Comparison (comparison_matrix.csv)
```csv
model,technique,avg_success_rate,avg_tokens,avg_cost,avg_time
qwen2.5:7b,zero_shot,0.45,2100,$0.000,185
qwen2.5:7b,few_shot,0.62,3200,$0.000,215
qwen2.5-coder:7b,zero_shot,0.48,2150,$0.000,175
gpt-4-turbo,zero_shot,0.85,3500,$0.125,32
gpt-4-turbo,few_shot,0.91,4200,$0.150,35
...
```

### Cost Estimate
```bash
# Per combination (model × technique)
cost_per_combination = bugs × cost_per_bug × max_attempts

# Full baseline (assume 4 models, 3 techniques)
total_cost ≈ 4 × 3 × (bugs × $0.50) = 6× cost of single model
```

For bugs 1-5 with OpenAI only: ~$15-25  
For all models: ~$100-150+

---

# ADVANCED CUSTOMIZATION

## Modifying Script Behavior

### Environment Variables

Override default behavior by setting environment variables:

```bash
# Force output colors
export FORCE_COLOR=1
bash scripts/pipeline/cloud.sh --bugs 1-5 --setup --run

# Disable colors
export NO_COLOR=1
bash scripts/pipeline/cloud.sh --bugs 1-5 --setup --run

# Custom Python path
export PYTHON=/usr/local/bin/python3.12
bash scripts/pipeline/cloud.sh --bugs 1-5 --setup --run
```

### Script Configuration Files

Create a `.env` file in project root for defaults:

```bash
# .env
OPENAI_API_KEY="sk-..."
OLLAMA_ENDPOINT="http://localhost:11434"
PROJECT_DIR="/path/to/project"
DEFAULT_DATASET="dataset"
DEFAULT_BUGS="80-82"
MAX_RETRIES=3
```

Load with: `source .env` before running scripts

### Modifying Script Paths

Edit first lines of any script to customize locations:

```bash
# In pipeline/cloud.sh, modify:
PROJECT_DIR="/custom/path"
CACHE_DIR="/custom/cache"
DATASET_PATH="/custom/dataset"
```

### Customizing Output Format

Edit logging functions in script to change output style:

```bash
# In pipeline/cloud.sh ~line 70-90:
log_info() {
    # Customize format here
    echo "[$(date)] $1"
}
```

---

# TROUBLESHOOTING & DEBUGGING

## Common Issues

### Script Not Found
```bash
# Error: No such file or directory
bash: scripts/pipeline/cloud.sh: No such file or directory

# Solution: Navigate to project root
cd /Users/mehilshah/Downloads/Research/ICSE26-RepGen
bash scripts/pipeline/cloud.sh --bugs 1-5 --setup --run
```

### Permission Denied
```bash
# Error: Permission denied: scripts/pipeline/cloud.sh
chmod +x scripts/pipeline/cloud.sh
bash scripts/pipeline/cloud.sh --bugs 1-5 --setup --run
```

### API Key Not Found
```bash
# Error: OPENAI_API_KEY not set
export OPENAI_API_KEY="sk-..."
bash scripts/quick-start/cloud.sh 80-82 1
```

### Ollama Connection Failed
```bash
# Error: Cannot connect to Ollama
# Solution 1: Start Ollama
ollama serve

# Solution 2: Check endpoint (in another terminal)
curl http://localhost:11434/api/tags

# Solution 3: Custom endpoint
OLLAMA_ENDPOINT="http://custom:11434" bash scripts/quick-start/local.sh 80-82 1
```

## Debugging Options

### Enable Verbose Logging
```bash
# Save detailed logs
bash scripts/pipeline/cloud.sh \
  --bugs 1-5 \
  --setup --run \
  --log-file results/debug_verbose.log

# Monitor in real-time
tail -f results/debug_verbose.log
```

### Single Bug Test
```bash
# Test on single bug for faster iteration
bash scripts/pipeline/cloud.sh --bugs 80 --setup --run

# Check output structure
ls -la dataset/080/*/
cat dataset/080/reproduction_code/*
```

### Dry Run (Setup Only)
```bash
# Clone code without generation (fast)
bash scripts/pipeline/cloud.sh --bugs 1-5 --setup

# Verify cloned code exists
ls -la .code_cache/001/ .code_cache/002/ ...
```

---