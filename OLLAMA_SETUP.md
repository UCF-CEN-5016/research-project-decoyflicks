# OLLAMA_SETUP.md - Local (Free)

## Overview

This guide covers the **local pipeline** using Ollama with open-source Qwen2.5 models. Use this for:
- **Completely free** - No API costs, only electricity
- **Private data** - Everything runs locally on your machine
- **Full dataset processing** - Perfect for running all 106 bugs without per-bug fees
- **Learning & transparency** - Inspect model outputs, understand decision-making
- **Offline operation** - Once models are downloaded, no internet needed

**Trade-off:** Slower than cloud (~3-5 min/bug on CPU vs 30 sec with GPT-4) but no costs

---

## Prerequisites

### System Requirements
- **RAM:** 16GB minimum (for 7B models); 24GB+ recommended for GPU use
- **Disk:** ~30GB free space (10GB for models + 20GB for datasets)
- **CPU:** Modern multi-core processor (Intel i7+ / AMD Ryzen 5+ equivalent)
- **GPU (Optional):** NVIDIA (CUDA) or Apple Silicon (Metal) for 3-5x speedup

### Check Your System
```bash
# Check RAM available
free -h              # Linux
vm_stat              # macOS
Get-WmiObject Win32_OperatingSystem | Select TotalVisibleMemorySize  # Windows PowerShell

# Check disk space
df -h                # Linux/macOS
dir C:               # Windows

# Check hardware
system_profiler SPHardwareDataType  # macOS
lscpu                # Linux
```

---

## Step 1: Install Ollama

### macOS / Linux
```bash
# Visit https://ollama.ai/download
# Download the installer and run it
# Or for macOS:
brew install ollama

# Verify installation
ollama --version
```

### Windows
```bash
# 1. Download from https://ollama.ai/download
# 2. Run the installer (OllamaSetup.exe)
# 3. Use Windows Terminal (PowerShell) or Git Bash
# 4. Verify installation
ollama --version
```

### Manual Verification
```bash
# Test that ollama command works
ollama --version

# Should output: ollama version X.X.X
```

---

## Step 2: Pull Required Models

### Quick Install (Default)
```bash
ollama pull qwen2.5:7b           # Reasoning/analysis (~5GB)
ollama pull qwen2.5-coder:7b     # Code generation (~5GB)
```

**Download time:** 10-20 minutes (depends on internet speed)

### Verify Models Downloaded
```bash
ollama list

# Should show:
# NAME                  ID              SIZE     MODIFIED
# qwen2.5:7b           ...              5.0 GB   ...
# qwen2.5-coder:7b     ...              5.0 GB   ...
```

### Optional: Additional Models (for Experimental Scripts)
If using `scripts/experimental/baseline.sh`, you may need:
```bash
ollama pull deepseek-r1:7b       # DeepSeek reasoning (~6GB)
ollama pull qwen2.5-coder:7b     # Already pulled above
ollama pull llama3:8b            # Meta's Llama 3 (~7GB)
```

---

## Step 3: Setup RepGen Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate       # macOS/Linux
# or
.\venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Verify dependencies
python3 -c "import requests; print('✓ Dependencies installed')"
```

---

## Step 4: Start Ollama Service

### Important: Start Ollama First!
Ollama must be running before you execute the pipeline scripts. It runs as a service listening on `localhost:11434`.

```bash
# Start Ollama (keep this terminal open)
ollama serve

# Output should show:
# Listening on 127.0.0.1:11434
# ...
```

**Note:** This runs in the foreground. Open a second terminal for the pipeline script.

---

## Quick Start (5 minutes)

### Two-Terminal Setup
```bash
# Terminal 1: Start Ollama (keep running)
ollama serve

# Terminal 2: Run quick test
bash scripts/quick-start/local.sh 80-82 1
```

**What this does:**
- Auto-validates Ollama is running and models are available
- Tests on 2 bugs (80-82) from evaluation dataset
- Takes ~3-5 minutes depending on hardware
- Perfect for verifying everything works

---

## Running the Pipeline

### Pattern 1: Quick Test (Validate Setup)
```bash
# Terminal 1: (keep running)
ollama serve

# Terminal 2:
bash scripts/quick-start/local.sh 80-82 1
```
**Time:** 3-5 minutes | **Validates:** Setup works

### Pattern 2: Small Batch Test
```bash
bash scripts/quick-start/local.sh 1-10 1
```
**Time:** 30-50 minutes | **Validates:** Works on full dataset

### Pattern 3: Full Dataset (All 106 Bugs)
```bash
bash scripts/quick-start/local.sh 1-106 1
```
**Time:** 8-15 hours (CPU) | **Cost:** Free | **Produces:** Complete results

### Pattern 4: Full Pipeline with Advanced Options
```bash
bash scripts/pipeline/local.sh \
  --bugs 1-10 \
  --dataset dataset \
  --setup \
  --run \
  --max-attempts 2
```

### Pattern 5: Setup and Run Separately
```bash
# Phase 1: Clone all repositories (run once)
bash scripts/pipeline/local.sh --bugs 1-106 --dataset dataset --setup

# Phase 2: Run pipeline on already-setup bugs
bash scripts/pipeline/local.sh --bugs 1-106 --dataset dataset --run --skip-code
```

---

## Options Reference

```bash
--bugs RANGE              # Bug IDs (REQUIRED)
                          # Formats: 1-10, 80-82, 1,5,10, or mixed

--dataset PATH            # dataset (default) or ae_dataset

--setup                   # Clone repositories at specific commits

--run                     # Execute reproduction pipeline

--skip-code               # Skip cloning; use existing code
                          # Faster for re-runs on pre-setup bugs

--force-clone             # Force fresh clones even if code exists

--max-attempts N          # Retry attempts per bug (default: 1)
                          # 2-3 recommended for better success

--retrieval ABLATION      # Retrieval strategy (expert only)
                          # For ablation studies

--generation ABLATION     # Generation strategy (expert only)
                          # For ablation studies
```

---

## Performance Guide

### Expected Speed by Hardware

| Hardware | Speed | Time (106 bugs) |
|----------|-------|-----------------|
| **CPU (2022 i7/Ryzen5)** | 3-5 min/bug | 8-15 hours |
| **CPU (older)** | 10+ min/bug | 18-30 hours |
| **Apple Silicon (M1/M2)** | 2-3 min/bug | 4-6 hours |
| **GPU (NVIDIA RTX 3080)** | 1 min/bug | 2-3 hours |
| **GPU (NVIDIA RTX 4090)** | 30 sec/bug | ~1 hour |

### Performance Optimization

#### 1. Use GPU Acceleration (5-10x faster)
**For NVIDIA GPU:**
```bash
# Install CUDA support (optional, advanced)
# Check Ollama documentation for GPU setup

# Verify GPU is detected
ollama list  # Shows if GPU is being used
```

**For Apple Silicon (M1/M2/M3):**
- Already optimized! Apple Metal acceleration is automatic
- Expected: 2-3 min/bug

#### 2. Adjust Batch Size
```bash
# If running slowly, try smaller batches
bash scripts/quick-start/local.sh 1-5 1      # Run 5 bugs, check progress
bash scripts/quick-start/local.sh 6-10 1     # Then run next 5

# Better memory management on systems with limited RAM
```

#### 3. Monitor System Resources
```bash
# In another terminal, monitor CPU/RAM usage
top              # Linux/macOS
Watch-Process    # Windows PowerShell
```

#### 4. Reduce Retry Attempts
```bash
# Default (--max-attempts 1) is usually sufficient
# Retries add time with marginal improvement

# If changing, use:
bash scripts/quick-start/local.sh 80-82 1    # 1 attempt (fast)
bash scripts/quick-start/local.sh 80-82 2    # 2 attempts (better success rate)
```

---

## Output Files

### Main Output
```
dataset/BUG_ID/
├── reproduction_code/
│   └── reproduce_BUG_ID.py      # Executable reproduction script
├── plan/
│   └── plan_BUG_ID.txt          # Execution plan / strategy
└── refined_bug_report/
    └── BUG_ID.txt               # Refined analysis of the bug
```

### Logs
```
results/
├── logs_TIMESTAMP.txt           # Detailed execution logs
└── (no API metrics since local)
```

### To Check Results
```bash
# View generated reproduction code
cat dataset/001/reproduction_code/reproduce_001.py

# View execution plan
cat dataset/001/plan/plan_001.txt

# View logs
tail -100 results/logs_*.txt
```

---

## Common Issues & Solutions

### Issue: `ollama: command not found`
**Cause:** Ollama not installed or not in PATH  
**Solution:**
```bash
# Reinstall from https://ollama.ai/download

# Or verify installation
ollama --version

# If still not found, add to PATH (macOS)
export PATH="/usr/local/bin:$PATH"
```

### Issue: `Connection refused` / `cannot connect to Ollama`
**Cause:** Ollama service not running  
**Solution:**
```bash
# Terminal 1: Start Ollama service
ollama serve

# Terminal 2: Verify it's running
curl http://localhost:11434/api/tags

# If curl command works, connection is good
```

### Issue: Models Not Available
```
Error: model 'qwen2.5:7b' not found
```
**Solution:**
```bash
# Pull the models
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b

# Verify they're available
ollama list
```

### Issue: Out of Memory Error
```
Error: cannot allocate memory / OOM
```
**Solution:**
```bash
# Option 1: Run on smaller bug sets
bash scripts/quick-start/local.sh 1-5 1     # Try 5 bugs first

# Option 2: Check available memory
free -h                # Linux
vm_stat                # macOS

# Option 3: Close other applications to free RAM

# Option 4: Add more RAM or use a system with more memory
```

### Issue: Slow Performance (Much Slower Than Expected)
**Cause:** Running on CPU instead of GPU  
**Solution:**
```bash
# Check if GPU is being used
ollama list

# For NVIDIA: Set up CUDA (see Performance Guide section)
# For Apple: Ensure you have recent macOS (Metal support automatic)
# For CPU: Expected speed is 3-5 min/bug; this is normal
```

### Issue: Models Keep Crashing / Partial Outputs
**Cause:** Model running out of memory or system instability  
**Solution:**
```bash
# Option 1: Retry with more attempts
bash scripts/quick-start/local.sh 80-82 3    # 3 attempts instead of 1

# Option 2: Reduce model size (not recommended)
ollama pull qwen2.5:7b-q4_K_M      # 4-bit quantized (smaller, faster)

# Option 3: Check system health
top              # Monitor CPU/RAM during run
dmesg            # Check system logs for hardware issues
```

---

## Storage Management

### Check Disk Usage
```bash
# Total size of datasets
du -sh dataset/          # ~20GB for all 106 bugs

# Total size of models
du -sh ~/.ollama/        # ~10GB for pulled models

# Free up space (remove dataset)
rm -rf dataset/
# Models are in ~/.ollama and can be redownloaded with 'ollama pull'
```

### Cleanup Old Runs
```bash
# Remove output from previous runs (if needed)
rm -rf dataset/*/reproduction_code
rm -rf dataset/*/plan
rm -rf results/logs_*.txt

# Models (qwen2.5:7b, qwen2.5-coder:7b) stay in ~/.ollama/models
```

---

## Advanced Topics

### Using Different Models
You can substitute different open-source models:

```bash
# CodeLlama (good for code tasks)
ollama pull codellama:7b

# Llama 2 (larger, potentially better quality)
ollama pull llama2:7b

# Use in pipeline
bash scripts/pipeline/local.sh --bugs 1-10 --setup --run
# (Scripts auto-detect and use available models)
```

### Running Multiple Instances
```bash
# Terminal 1: Ollama service
ollama serve

# Terminal 2: First batch
bash scripts/quick-start/local.sh 1-50 1 &

# Terminal 3: Second batch
bash scripts/quick-start/local.sh 51-106 1 &

# Monitor progress in separate terminals
```

### Monitoring Long Runs
```bash
# Terminal for live log monitoring
tail -f results/logs_*.txt

# Terminal for progress tracking
watch 'ls dataset/*/reproduction_code/reproduce_*.py 2>/dev/null | wc -l'
# Shows how many bugs have been processed
```

---

## Troubleshooting Checklist

Before asking for help, verify:
- [ ] Ollama is running: `curl http://localhost:11434/api/tags`
- [ ] Models are available: `ollama list`
- [ ] Python environment activated: `which python` shows venv path
- [ ] Dependencies installed: `pip list | grep requests`
- [ ] Dataset exists: `ls dataset/001` shows files
- [ ] Disk space: `df -h` shows >10GB free
- [ ] Sufficient RAM: `free -h` or `vm_stat` shows >8GB available

---

## Performance Summary

| Aspect | Free (Ollama) | Cloud (OpenAI) |
|--------|--------------|----------------|
| **Cost** | Free | $50-100 |
| **Speed** | 3-5 min/bug | 30 sec/bug |
| **Privacy** | 100% local | Data sent to OpenAI |
| **Quality** | Good (Qwen 7B) | Excellent (GPT-4) |
| **Internet Required** | After model download | Always |
| **Time for 106 bugs** | 8-15 hours | 45-90 min |

---

## Next Steps

1. **Install Ollama** from https://ollama.ai/download
2. **Pull models:** `ollama pull qwen2.5:7b qwen2.5-coder:7b`
3. **Start service:** `ollama serve` (in Terminal 1)
4. **Run quick test:** `bash scripts/quick-start/local.sh 80-82 1` (in Terminal 2)
5. **Check results:** Look at generated files in `dataset/001/reproduction_code/`
6. **Run full dataset** (takes 8-15 hours): `bash scripts/quick-start/local.sh 1-106 1`

---

## See Also

- [scripts/README.md](scripts/README.md) - All available scripts and options
- [README.md](README.md) - Project overview
- [PIPELINE.md](PIPELINE.md) - Cloud (OpenAI) alternative
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - Windows-specific issues
- [Ollama Official Docs](https://ollama.ai) - Model gallery and advanced setup
