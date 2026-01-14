# RepGen Pipeline - User Guide

## Overview

`pipeline.sh` is the single, unified entry point for the entire bug reproduction workflow. It's highly configurable and handles:
- Setting up experimental datasets
- Cloning code from repositories at specific commits
- Copying bug reports and context files
- Running the reproduction pipeline

**Supported Platforms:** macOS, Linux, Windows (Git Bash / WSL required on Windows)

## Quick Start

### 1. Basic Usage (Setup + Run)
```bash
export OPENAI_API_KEY="sk-..."

# Setup and run on bugs 80-82
bash scripts/pipeline.sh --bugs 80-82 --setup --run

# Using default dataset folder
```

### 2. Quick Start Script
For even simpler usage:
```bash
export OPENAI_API_KEY="sk-..."

# Setup and run bugs 80-82, max 1 attempt
bash scripts/quick_start.sh 80-82 1

# Setup and run bugs 1-10, max 5 attempts
bash scripts/quick_start.sh 1-10 5
```

---

## Detailed Usage

### Command Syntax
```bash
bash scripts/pipeline.sh [OPTIONS]
```

### Options

| Option | Value | Description |
|--------|-------|-------------|
| `--bugs` | `RANGE` | **Required**. Bugs to process. Format: `1-10`, `80-82`, or `80,81,82` |
| `--dataset` | `PATH` | Dataset path (default: `dataset`). Use `ae_dataset` for experimental |
| `--setup` | - | Setup: clone code and copy files |
| `--run` | - | Run: execute pipeline |
| `--skip-code` | - | Skip code cloning (if already setup) |
| `--force-clone` | - | Force re-clone of repositories |
| `--max-attempts` | `N` | Max generation attempts per bug (default: 1) |
| `--retrieval` | `ABLATION` | Retrieval ablation config (default: full_system) |
| `--generation` | `ABLATION` | Generation ablation config (default: all_steps) |
| `--help` | - | Show help message |

---

## Common Workflows

### 1. Experimental Dataset (ae_dataset)

Create and run on separate experimental dataset without touching original:

```bash
# Setup ae_dataset with bugs 80-82
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup

# Run pipeline on ae_dataset
export OPENAI_API_KEY="sk-..."
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --run
```

**Result:** Files in `ae_dataset/80/`, `ae_dataset/81/`, `ae_dataset/82/`

### 2. Original Dataset (dataset)

Setup and run on original dataset:

```bash
export OPENAI_API_KEY="sk-..."

bash scripts/pipeline.sh --bugs 1-10 --dataset dataset --setup --run
```

### 3. Setup Only (No Run)

Clone code and copy files without running pipeline:

```bash
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup
```

Files are now ready in `ae_dataset/`. Run anytime later with `--run`.

### 4. Run Only (Skip Setup)

Run pipeline on already-setup dataset:

```bash
export OPENAI_API_KEY="sk-..."

bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --run --skip-code
```

### 5. Re-clone Repositories

Force fresh clones (skips cache):

```bash
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --force-clone
```

### 6. Full Paper Replication

Setup and run all 106 bugs:

```bash
export OPENAI_API_KEY="sk-..."

bash scripts/pipeline.sh --bugs 1-106 --dataset dataset --setup --run --max-attempts 5
```

### 7. Custom Ablations

Run with specific retrieval/generation ablations:

```bash
export OPENAI_API_KEY="sk-..."

bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --run \
    --retrieval NO_BM25 \
    --generation no_plan \
    --max-attempts 3
```

---

## Dataset Structure

### After Setup

```
ae_dataset/
├── 080/
│   ├── code/               ← Repository code (cloned)
│   ├── bug_report/         ← Bug report (copied from original)
│   ├── context/            ← Context files (copied from original)
│   └── reproduction_code/  ← Pipeline outputs (generated)
├── 081/
│   └── ...
└── 082/
    └── ...
```

### Original Dataset (Untouched)

```
dataset/
├── 080/
│   ├── bug_report/
│   ├── context/
│   ├── code/  (already existed)
│   └── ...
└── ...
```

---

## Performance Tips

1. **Reuse Cached Repos:** Repository clones are cached in `.code_cache/`. They're reused automatically.

2. **Setup Once, Run Multiple Times:**
   ```bash
   # First time
   bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup
   
   # Later runs (no re-cloning)
   bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --run --skip-code
   ```

3. **Force Fresh Clone:** Use `--force-clone` if you need latest repo state:
   ```bash
   bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --force-clone
   ```

4. **Process Subsets:** Run bugs in smaller batches:
   ```bash
   bash scripts/pipeline.sh --bugs 1-50 --setup --run
   bash scripts/pipeline.sh --bugs 51-106 --setup --run
   ```

---

## Environment Requirements

- Python 3.12+
- Git
- **Bash shell** (required for all platforms):
  - **macOS/Linux:** Native bash available
  - **Windows:** Use [Git Bash](https://git-scm.com/download/win) or [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install)
- OpenAI API key (for running pipeline): `export OPENAI_API_KEY="sk-..."`

### Windows-Specific Setup

**Option 1: Git Bash (Recommended for beginners)**
1. Install [Git for Windows](https://git-scm.com/download/win) which includes Git Bash
2. Open "Git Bash" from Start Menu
3. Navigate to your project: `cd /c/path/to/project`
4. Run scripts: `bash scripts/pipeline.sh --bugs 1-10 --setup --run`

**Option 2: WSL 2 (Windows Subsystem for Linux)**
1. Install [WSL 2 with Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install)
2. Open Ubuntu terminal
3. Install Python 3.12 and dependencies
4. Run scripts normally as you would on Linux

### First-Time Setup
```bash
# Create virtual environment (in project root)
python3 -m venv venv

# Activate (macOS/Linux/WSL)
source venv/bin/activate

# Activate (Windows - Git Bash only)
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Output Locations

After running, find outputs at:

**Original dataset:**
```
dataset/<bug_id>/reproduction_code/reproduce_<bug_id>.py
```

**Experimental dataset:**
```
ae_dataset/<bug_id>/reproduction_code/reproduce_<bug_id>.py
```

---

## Examples

### Example 1: Quick Test
```bash
export OPENAI_API_KEY="sk-..."
bash scripts/quick_start.sh 80-82 1
```

### Example 2: Setup Multiple Datasets
```bash
# Setup dataset for bugs 1-50
bash scripts/pipeline.sh --bugs 1-50 --dataset dataset --setup

# Setup ae_dataset for bugs 80-82 (parallel work)
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --force-clone
```

### Example 3: Run with Options
```bash
export OPENAI_API_KEY="sk-..."

bash scripts/pipeline.sh \
    --bugs 80-82 \
    --dataset ae_dataset \
    --run \
    --max-attempts 3 \
    --retrieval full_system \
    --generation all_steps
```

### Example 4: Process Bugs One by One
```bash
export OPENAI_API_KEY="sk-..."

for bug in 80 81 82; do
    bash scripts/pipeline.sh \
        --bugs "$bug" \
        --dataset ae_dataset \
        --setup --run
done
```

---

## Troubleshooting

### Windows: "bash: command not found"
**Solution:** Install Git Bash or WSL 2:
- [Git Bash installation](https://git-scm.com/download/win)
- [WSL 2 setup guide](https://learn.microsoft.com/en-us/windows/wsl/install)

Then run commands from Git Bash terminal, not CMD.exe or PowerShell.

### API Key Error
```
[ERROR] OPENAI_API_KEY not set
```
**Solution:**
```bash
# macOS/Linux/WSL
export OPENAI_API_KEY="sk-your-key-here"

# Windows Git Bash
export OPENAI_API_KEY="sk-your-key-here"
```

### Windows: Virtual environment not activating
**Solution:** Use the correct activation command for Git Bash:
```bash
source venv/Scripts/activate  # Git Bash on Windows
source venv/bin/activate      # macOS/Linux/WSL
```

### Code Directory Missing
```
[ERROR] Code directory missing for bug XXX
```
**Solution:** Run with `--setup` first:
```bash
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup
```

### Git Clone Failed
**Solution:** Check internet connection or repo URL. Try `--force-clone`:
```bash
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --force-clone
```

---

## Architecture

```
pipeline.sh
├── SETUP PHASE (--setup)
│   ├── Parse bug range
│   ├── Clone repos from Dataset.csv (cached in .code_cache/)
│   ├── Checkout specific commits
│   ├── Copy code to dataset/<bug_id>/code/
│   ├── Copy bug reports from original dataset
│   └── Copy context files from original dataset
│
└── RUN PHASE (--run)
    ├── Activate Python environment
    ├── Check OPENAI_API_KEY
    └── For each bug:
        └── Run tool_openai.py with:
            ├── Bug ID
            ├── Dataset path
            ├── Max attempts
            ├── Ablation configs
            └── Generate: ae_dataset/<bug_id>/reproduction_code/
```

---

## Notes

- **Non-destructive:** Original `dataset/` folder is never modified when using `--dataset ae_dataset`
- **Efficient:** Repository clones are cached in `.code_cache/` and reused
- **Flexible:** All parameters are optional with sensible defaults
- **Configurable:** Supports all retrieval and generation ablations
- **Modular:** `--setup` and `--run` can be used separately

