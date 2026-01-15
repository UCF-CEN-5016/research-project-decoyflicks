## Overview

This guide describes the **cloud-based pipeline** for RepGen using OpenAI GPT-4 models. It is intended for:

- **Fast inference** (~30 seconds per bug)
- **Small batches** (test runs, debugging, validation)
- **Latest models** via the OpenAI commercial API
- **Stable and reproducible performance**

**Pipeline phases:**
- **Setup**: Clone repositories at bug-specific commits and organize files
- **Run**: Execute retrieval, planning, and code generation using the OpenAI API

**Trade-off:** Higher cost (~$0.50–$1 per bug) in exchange for speed and reliability


## Prerequisites

### 1. OpenAI API Key
- Create a key at: https://platform.openai.com/api-keys
- Keep it private and out of version control
- Expose it via the `OPENAI_API_KEY` environment variable

### 2. Python 3.12+
```bash
python3 --version
````

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Setup

### One-Command Setup (Recommended)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

### Step-by-Step Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate        # macOS/Linux
.\venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-..."

# Verify
python3 -c "import openai; print('OpenAI module loaded')"
```



## Running the Pipeline

### Quick Test (First Run)

```bash
bash scripts/quick-start/cloud.sh 80-82 1
```

### Full Pipeline Control

```bash
bash scripts/pipeline/cloud.sh [OPTIONS]
```



## Common Usage Patterns

### Quick Validation

```bash
bash scripts/quick-start/cloud.sh 80-82 1
```

### Small Batch

```bash
bash scripts/quick-start/cloud.sh 1-10 1
# or
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --setup --run
```

### Full Dataset

```bash
bash scripts/quick-start/cloud.sh 1-106 2
```

### Setup and Run Separately

```bash
# Setup (one-time)
bash scripts/pipeline/cloud.sh --bugs 1-106 --dataset dataset --setup

# Run
bash scripts/pipeline/cloud.sh --bugs 1-106 --dataset dataset --run --skip-code
```

### Re-run Failed Bugs

```bash
bash scripts/pipeline/cloud.sh --bugs 1-10 --dataset dataset --run --skip-code --max-attempts 3
```

### Custom Bug Selection

```bash
bash scripts/pipeline/cloud.sh --bugs 1,5,10,42 --dataset dataset --setup --run
bash scripts/pipeline/cloud.sh --bugs 80-85,90,95-100 --dataset dataset --setup --run
```



## Options Reference

```bash
--bugs RANGE              # REQUIRED. e.g., 1-10, 80-82, 1,5,10, or mixed
--dataset PATH            # Dataset path (default: dataset)
--setup                   # Clone repositories at bug-specific commits
--run                     # Execute retrieval, planning, and generation
--skip-code               # Reuse existing clones
--force-clone             # Re-clone even if code exists
--max-attempts N          # Retries per bug (default: 1)
--retrieval ABLATION      # Retrieval strategy (ablation only)
--generation ABLATION     # Generation strategy (ablation only)
--quiet                   # Suppress progress output
```



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



## Cost Overview

### Estimates

| Scope    | Cost        |
| -- | -- |
| Per bug  | $0.50–$1.00 |
| 10 bugs  | $5–$10      |
| 50 bugs  | $25–$50     |
| 106 bugs | $50–$100    |

### Cost Drivers

* Bug complexity and context size
* `--max-attempts` value
* Selected GPT-4 variant

### Cost Control Tips

* Validate with 2–3 bugs first
* Start with `--max-attempts 1`
* Scale gradually to full runs
* Use the Ollama pipeline for free local inference when possible



## Troubleshooting

### Invalid API Key

```bash
echo $OPENAI_API_KEY
export OPENAI_API_KEY="sk-..."
python3 -c "from openai import OpenAI; OpenAI(); print('API key OK')"
```

### Missing OpenAI Module

```bash
source venv/bin/activate
pip install -r requirements.txt
python3 -c "import openai; print(openai.__version__)"
```

### Connection or Rate Limit Errors

* Retry after a short delay
* Reduce batch size
* Check OpenAI service status

### Unexpected Costs

* Stop execution with `Ctrl+C`
* Reduce retries and batch size
* Switch to the local Ollama pipeline if needed

### Out of Memory

* Run smaller bug ranges
* Execute batches sequentially
* Cloud API itself does not require large local memory



## Advanced Configuration

### Change OpenAI Model

```python
# src/tool_openai.py
model = "gpt-4-turbo-2024-04-09"
```

### Ablation Configurations

```bash
bash scripts/pipeline/cloud.sh \
  --bugs 80-82 \
  --dataset ae_dataset \
  --setup --run \
  --retrieval full_system \
  --generation all_steps
```


## Logging

```bash
# Live logs
tail -f results/logs_*.txt

# Errors
grep ERROR results/logs_*.txt
```

## Related Documentation

* `scripts/README.md` – Script details
* `README.md` – Project overview
* `QWEN_PIPELINE.md` – Local inference
* `WINDOWS_SETUP.md` – Windows-specific guidance