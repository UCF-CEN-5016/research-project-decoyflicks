## Overview

This guide documents the **local pipeline** for RepGen using Ollama with open-source **Qwen2.5** models. It is designed for users who want to run the full reproduction pipeline without API costs.

**Best suited for:**
- **Completely free execution** (no API usage)
- **Private and offline workflows**
- **Full dataset runs** (all 106 bugs)
- **Transparency and inspection** of intermediate model outputs

**Trade-off:** Significantly slower than the cloud pipeline (≈3–5 min/bug on CPU vs. ≈30 sec/bug with GPT-4), but zero monetary cost.



## Prerequisites

### System Requirements
- **RAM:** 16GB minimum (7B models); 24GB+ recommended
- **Disk:** ~30GB free (models + datasets)
- **CPU:** Modern multi-core processor
- **GPU (optional):** NVIDIA (CUDA) or Apple Silicon (Metal) for acceleration

### System Checks
```bash
# Memory
free -h              # Linux
vm_stat              # macOS

# Disk
df -h                # Linux/macOS

# Hardware
lscpu                # Linux
system_profiler SPHardwareDataType  # macOS
````



## Step 1: Install Ollama

### macOS / Linux

```bash
brew install ollama   # macOS
ollama --version
```

### Windows

1. Download from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer
3. Verify:

```bash
ollama --version
```



## Step 2: Download Required Models

### Default Models

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

**Disk usage:** ~10GB
**Download time:** ~10–20 minutes

### Verify

```bash
ollama list
```

### Optional (Experimental Scripts Only)

```bash
ollama pull deepseek-r1:7b
ollama pull llama3:8b
```



## Step 3: Python Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
.\venv\Scripts\activate         # Windows

pip install -r requirements.txt

python3 -c "import requests; print('Dependencies OK')"
```



## Step 4: Start Ollama Service

Ollama must be running before executing any pipeline scripts.

```bash
ollama serve
```

Expected output:

```
Listening on 127.0.0.1:11434
```

NOTE: If it throws an error saying "bind: address already in use", it means that ollama is running in the background, and you can skip the step where we serve the model, and move to the next step.

Keep this terminal open and use a second terminal for pipeline execution.



## Quick Start

```bash
# Terminal 1
ollama serve

# Terminal 2
bash scripts/quick-start/local.sh 80-82 1
```

**Purpose:** Validate setup on 2 bugs
**Time:** ~3–5 minutes



## Running the Pipeline

### Common Patterns

#### Quick Validation

```bash
bash scripts/quick-start/local.sh 80-82 1
```

#### Small Batch

```bash
bash scripts/quick-start/local.sh 1-10 1
```

#### Full Dataset

```bash
bash scripts/quick-start/local.sh 1-106 1
```

#### Advanced Control

```bash
bash scripts/pipeline/local.sh \
  --bugs 1-10 \
  --dataset dataset \
  --setup \
  --run \
  --max-attempts 2
```

#### Setup and Run Separately

```bash
# Setup (one-time)
bash scripts/pipeline/local.sh --bugs 1-106 --dataset dataset --setup

# Run
bash scripts/pipeline/local.sh --bugs 1-106 --dataset dataset --run --skip-code
```

## Options Reference

```bash
--bugs RANGE              # REQUIRED (e.g., 1-10, 80-82, 1,5,10)
--dataset PATH            # dataset (default) or ae_dataset
--setup                   # Clone repositories
--run                     # Execute reproduction pipeline
--skip-code               # Reuse existing clones
--force-clone             # Force fresh clones
--max-attempts N          # Retry attempts per bug (default: 1)
--retrieval ABLATION      # Ablation only
--generation ABLATION     # Ablation only
```

### Optimization Tips

* Prefer GPU or Apple Silicon where available
* Run smaller batches on low-RAM systems
* Keep `--max-attempts` low for speed
* Monitor resources with `top` or equivalent

## Output Structure

```
dataset/BUG_ID/
├── reproduction_code/
│   └── reproduce_BUG_ID.py
├── plan/
│   └── plan_BUG_ID.txt
└── refined_bug_report/
    └── BUG_ID.txt
```

### Logs

```
results/
└── logs_TIMESTAMP.txt
```



## Troubleshooting

### Ollama Not Found

```bash
ollama --version
```

Reinstall if missing or fix PATH.

### Cannot Connect to Ollama

```bash
ollama serve
curl http://localhost:11434/api/tags
```

### Model Missing

```bash
ollama pull qwen2.5:7b qwen2.5-coder:7b
```

### Out of Memory

* Run fewer bugs per batch
* Close other applications
* Use a higher-memory system

### Slow Performance

* Verify GPU usage (`ollama list`)
* CPU-only runs at 3–5 min/bug are expected



## Storage Management

```bash
# Dataset size
du -sh dataset/

# Model size
du -sh ~/.ollama/

# Cleanup outputs
rm -rf dataset/*/reproduction_code dataset/*/plan results/logs_*.txt
```



## Related Documentation

* `scripts/README.md` – Script details
* `README.md` – Project overview
* `OPENAI_PIPELINE.md` – Cloud pipeline
* `WINDOWS_SETUP.md` – Windows notes
* [https://ollama.ai](https://ollama.ai) – Official Ollama documentation
