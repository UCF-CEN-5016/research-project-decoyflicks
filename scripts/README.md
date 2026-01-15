# RepGen Scripts

This directory contains shell scripts for running the RepGen bug reproduction pipeline. Scripts are organized into three categories based on their purpose.

---

## 📁 Directory Structure

```
scripts/
├── quick-start/       # Entry point for first-time users
│   ├── cloud.sh       # Quick test with OpenAI API
│   └── local.sh       # Quick test with local Ollama
├── pipeline/          # Full pipeline with advanced options
│   ├── cloud.sh       # Full cloud-based pipeline
│   └── local.sh       # Full local-based pipeline
└── experimental/      # Ablation studies and baselines
    ├── ablations.sh   # Ablation experiments
    └── baseline.sh    # Baseline comparisons
```

---

## 🚀 Quick Start (Choose One)

### Cloud (OpenAI - Fastest)
```bash
bash scripts/quick-start/cloud.sh 80-82 1
```
- **Setup:** 15 min
- **Speed:** 30s/bug
- **Cost:** $50-100 for all 106 bugs
- **Requires:** OpenAI API key

### Local (Ollama - Free)
```bash
bash scripts/quick-start/local.sh 80-82 1
```
- **Setup:** 45 min
- **Speed:** 3-5 min/bug
- **Cost:** Free
- **Requires:** 16GB+ RAM, Ollama

---

## 📋 Script Documentation

### Quick Start Scripts

**Purpose:** Simplest entry point for running the pipeline. Pre-configured with sensible defaults.

#### `quick-start/cloud.sh` - Cloud Quick Start
- **Function:** One-command execution using OpenAI API
- **Usage:** `bash scripts/quick-start/cloud.sh [BUGS] [MAX_ATTEMPTS] [OPTIONS]`
- **Arguments:**
  - `BUGS` (default: `1-10`): Bug IDs to process (e.g., `1-10`, `80-82`, `1,5,10`)
  - `MAX_ATTEMPTS` (default: `5`): Retry attempts per bug
  - `OPTIONS`: Pass through flags (e.g., `--quiet`)
- **Examples:**
  ```bash
  bash scripts/quick-start/cloud.sh 80-82 1              # Test on 2 bugs
  bash scripts/quick-start/cloud.sh 1-106 5              # Full paper replication
  bash scripts/quick-start/cloud.sh 1-10 3 --quiet       # Minimal output
  ```
- **Prerequisites:**
  - Python 3.12+, virtual environment with dependencies
  - OpenAI API key set as `OPENAI_API_KEY` environment variable
- **Output:** Generates `dataset/BUG_ID/{reproduction_code, plan, refined_bug_report}`
- **Dataset:** Uses `ae_dataset/` (evaluation dataset)

#### `quick-start/local.sh` - Local Quick Start (Ollama)
- **Function:** One-command execution using local Ollama models
- **Usage:** `bash scripts/quick-start/local.sh [BUGS] [MAX_ATTEMPTS]`
- **Arguments:**
  - `BUGS` (default: `80-82`): Bug IDs to process
  - `MAX_ATTEMPTS` (default: `1`): Retry attempts per bug
- **Examples:**
  ```bash
  bash scripts/quick-start/local.sh 80-82 1              # Test on 2 bugs
  bash scripts/quick-start/local.sh 1-106 5              # Full paper replication
  ```
- **Prerequisites:**
  - Ollama installed: `https://ollama.ai/download`
  - Models pulled: `ollama pull qwen2.5:7b qwen2.5-coder:7b`
  - Ollama service running: `ollama serve` (in another terminal)
  - Python 3.12+, virtual environment with dependencies
- **Output:** Same as cloud version
- **Dataset:** Uses `ae_dataset/`

---

### Pipeline Scripts

**Purpose:** Full control over pipeline execution with advanced options. For experienced users or custom workflows.

#### `pipeline/cloud.sh` - Cloud Pipeline (Full Control)
- **Function:** Customizable pipeline with setup/run separation using OpenAI API
- **Usage:** `bash scripts/pipeline/cloud.sh [OPTIONS]`
- **Options:**
  ```
  --bugs RANGE                Bug IDs: 1-10, 80-82, or 80,81,82 [REQUIRED]
  --dataset PATH              dataset (default) or ae_dataset
  --setup                     Clone repositories at specific commits
  --run                       Execute pipeline for each bug
  --skip-code                 Skip cloning (use existing setup)
  --force-clone               Force fresh repository clones
  --max-attempts N            Retry attempts per bug (default: 1)
  --retrieval ABLATION        Retrieval config (default: full_system)
  --generation ABLATION       Generation config (default: all_steps)
  --quiet                     Minimal output
  ```
- **Examples:**
  ```bash
  # Setup only
  bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --setup

  # Run on pre-setup bugs
  bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --run --skip-code

  # Full pipeline (setup + run) with retries
  bash scripts/pipeline/cloud.sh --bugs 1-106 --dataset dataset --setup --run --max-attempts 3

  # With ablation settings
  bash scripts/pipeline/cloud.sh --bugs 80-82 --dataset ae_dataset --setup --run \
    --retrieval full_system --generation all_steps
  ```
- **When to use:**
  - Running on main `dataset/` with 106 bugs
  - Needing setup and run phases separately
  - Custom retrieval/generation configurations
  - Debugging with `--quiet` flag

#### `pipeline/local.sh` - Local Pipeline (Ollama, Full Control)
- **Function:** Full customizable pipeline using local Ollama models
- **Usage:** `bash scripts/pipeline/local.sh [OPTIONS]`
- **Options:** Same as `pipeline/cloud.sh` (see above)
- **Examples:**
  ```bash
  bash scripts/pipeline/local.sh --bugs 1-10 --setup --run
  bash scripts/pipeline/local.sh --bugs 80-82 --dataset ae_dataset --run --skip-code
  ```
- **Prerequisites:** Same as `quick-start/local.sh`

---

### Experimental Scripts

**Purpose:** Advanced experiments including ablation studies and baseline comparisons.

#### `experimental/ablations.sh` - Ablation Studies
- **Function:** Run systematic ablation experiments to measure component impact
- **Usage:** `bash scripts/experimental/ablations.sh <start_bug_id> <end_bug_id> [OPTIONS]`
- **Arguments:**
  - `start_bug_id` (required): First bug ID to process
  - `end_bug_id` (required): Last bug ID to process
- **Options:**
  ```
  --tool_script=SCRIPT        Specify Python tool (default: tool_openai.py)
  --dataset_path=PATH         Dataset path (default: dataset)
  ```
- **Examples:**
  ```bash
  bash scripts/experimental/ablations.sh 1 10
  bash scripts/experimental/ablations.sh 80 82 --tool_script=tool.py
  ```
- **Runs:**
  - **Retrieval ablations:** NO_BM25, NO_ANN, NO_RERANKER, NO_TRAINING_LOOP_EXTRACTION, NO_TRAINING_LOOP_RANKING, NO_MODULE_PARTITIONING, NO_DEPENDENCY_EXTRACTION
  - **Generation ablations:** no_refine, no_plan, no_compilation, no_relevance, no_static_analysis, no_runtime_feedback
- **Output:** Logs and metrics for each ablation variant
- **Requires:** OpenAI API key

#### `experimental/baseline.sh` - Baseline Comparisons
- **Function:** Compare performance across multiple models and prompting techniques
- **Usage:** `bash scripts/experimental/baseline.sh <start_bug_id> <end_bug_id>`
- **Arguments:**
  - `start_bug_id` (required): First bug ID
  - `end_bug_id` (required): Last bug ID
- **Examples:**
  ```bash
  bash scripts/experimental/baseline.sh 1 10
  bash scripts/experimental/baseline.sh 80 82
  ```
- **Models tested:**
  - Ollama: qwen2.5:7b, deepseek-r1:7b, qwen2.5-coder:7b, llama3:8b
  - Groq: llama-3.3-70b-versatile
  - DeepSeek API: deepseek-reasoner
  - OpenAI: gpt-4-turbo-2024-04-09
- **Prompting techniques:** zero_shot, few_shot, cot
- **Output:** Comparative metrics across all model/technique combinations

---

## 🔧 Common Workflows

### First-time run (just test if it works)
```bash
# Cloud
bash scripts/quick-start/cloud.sh 80-82 1

# Local
bash scripts/quick-start/local.sh 80-82 1
```

### Full paper replication
```bash
# Cloud - all 106 bugs with retries
bash scripts/quick-start/cloud.sh 1-106 5

# Local
bash scripts/quick-start/local.sh 1-106 5
```

### Processing specific bug IDs
```bash
# Just bugs 1, 5, and 10 (cloud)
bash scripts/pipeline/cloud.sh --bugs 1,5,10 --dataset dataset --setup --run

# Just bugs 80, 81, 82 (local)
bash scripts/pipeline/local.sh --bugs 80-82 --dataset ae_dataset --setup --run
```

### Setup and run separately (useful for debugging)
```bash
# Setup only (clone code, won't run yet)
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --setup

# Run on already-setup bugs (faster re-runs)
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --run --skip-code
```

### Run ablations on subset
```bash
bash scripts/experimental/ablations.sh 80 82
```

---

## 💡 Tips

- **Terminal Colors:** Scripts automatically disable colors on Windows CMD (use Git Bash or WSL for colors)
- **Quiet Mode:** Add `--quiet` to quick-start scripts to suppress progress output
- **Range Format:** Support multiple formats:
  - `1-10` (range)
  - `1,5,10` (specific IDs)
  - `80-82` (short range)
- **Cost Estimation:** Cloud runs at ~$0.50-1 per bug, so 106 bugs ≈ $50-100 total
- **Error Recovery:** Scripts have automatic retry logic (use `--max-attempts N`)
- **Logging:** Pipeline scripts create logs in `results/` directory

---

## 📚 References

- **Cloud Setup:** See [PIPELINE.md](../PIPELINE.md)
- **Local Setup:** See [OLLAMA_SETUP.md](../OLLAMA_SETUP.md)
- **Windows Setup:** See [WINDOWS_SETUP.md](../WINDOWS_SETUP.md)
- **Main Guide:** See [README.md](../README.md)
