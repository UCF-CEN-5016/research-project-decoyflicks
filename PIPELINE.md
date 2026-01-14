# PIPELINE.md - Cloud (OpenAI)

## Overview

This guide covers the **cloud-based pipeline** using OpenAI's GPT-4 models. Use this for fast inference on small batches or when you need the latest models.

**Pipeline Phases:**
- **Setup** - Clones code repositories at specific bug-related commits and organizes files
- **Run** - Executes the retrieval + planning + code generation pipeline using OpenAI API

---

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

---

## Run

Executes the entire pipeline using OpenAI API. Each bug goes through:
1. **Retrieval** - Extracts relevant code context
2. **Planning** - Creates execution strategies  
3. **Code Generation** - Produces reproduction code

### Quick Test (2 bugs)
```bash
bash scripts/quick_start.sh 80-82 1
```
Simplest entry point. Pre-configured for evaluation dataset (ae_dataset) with 1 retry attempt.

### Full Pipeline
```bash
bash scripts/pipeline.sh --bugs START-END --dataset DATASET --setup --run
```
Customizable pipeline script. Use `--setup` to clone code, `--run` to execute, or combine both.

**Examples:**
```bash
# Setup only (prepare code)
bash scripts/pipeline.sh --bugs 1-10 --dataset dataset --setup

# Run only (if already setup)
bash scripts/pipeline.sh --bugs 1-10 --dataset dataset --run --skip-code

# Full (setup + run) with retries
bash scripts/pipeline.sh --bugs 1-106 --dataset dataset --setup --run --max-attempts 3
```

---

## Options

```bash
--bugs RANGE              # Required: 1-10, 80-82, or 80,81,82
--dataset PATH            # dataset (default) or ae_dataset
--setup                   # Clone repositories at specific commits
--run                     # Execute pipeline for each bug
--skip-code              # Skip cloning (use if already setup)
--max-attempts N         # Retry attempts per bug (default: 1)
--force-clone            # Force fresh repository clones
```

**When to use each:**
- `--setup` alone: Prepare files without running (good for debugging)
- `--run --skip-code`: Run on pre-setup bugs (faster re-runs)
- Both together: Complete workflow (slower but guaranteed fresh state)

---

## Output

For each bug, the pipeline generates:

```
dataset/BUG_ID/
├── reproduction_code/
│   └── reproduce_BUG_ID.py      # Executable reproduction script
├── plan/
│   └── plan_BUG_ID.txt          # Generated execution plan
└── refined_bug_report/
    └── BUG_ID.txt               # Refined bug description
```

Also check `results/` directory for:
- `ExperimentalGroup.csv` - API usage metrics and costs
- `logs_*.txt` - Detailed execution logs

---

## Cost

**Pricing:**
- ~$0.50-1 per bug (varies by bug complexity)
- ~$50-100 for all 106 bugs in dataset
- Cloud costs scale with: bug complexity, number of retries, context size

**Tips to reduce cost:**
- Use smaller batches for testing: `--bugs 1-10` instead of `--bugs 1-106`
- Set `--max-attempts 1` (default) unless you need retries
- Monitor `results/ExperimentalGroup.csv` for usage trends

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `API key error` | `export OPENAI_API_KEY="sk-..."` |
| `Module not found` | `pip install -r requirements.txt` |
| `Permission denied` | `chmod +x scripts/*.sh` |
| `Out of memory` | Use smaller batch: `--bugs 1-10` |

