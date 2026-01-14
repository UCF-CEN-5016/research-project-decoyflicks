# OLLAMA_SETUP.md - Local (Free)

## Overview

This guide covers the **local pipeline** using Ollama with open-source Qwen2.5 models. Use this for:
- **Free inference** - No API costs or internet connectivity needed
- **Private data** - Everything runs locally
- **Full dataset processing** - Best for running all 106 bugs
- **Learning** - Inspect model outputs in detail

**Trade-off:** Slower than cloud (3-5 min/bug on CPU vs 30s with cloud)

---

## Install Ollama

**Step 1:** Download and install Ollama from: https://ollama.ai/download  
Available for macOS, Linux, Windows (WSL2).

**Step 2:** Download the required models:
```bash
ollama pull qwen2.5:7b           # Reasoning model (~5GB)
ollama pull qwen2.5-coder:7b     # Code generation model (~5GB)
```
Takes ~5-10 minutes depending on internet speed. Models are ~5GB each.

**Step 3:** Verify installation:
```bash
ollama list
# Should show: qwen2.5:7b and qwen2.5-coder:7b
```

---

## Setup RepGen

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## Run

### Quick Test (2 bugs)
```bash
# Terminal 1: Start Ollama service
ollama serve

# Terminal 2: Run quick test
bash scripts/ollama_quick_start.sh 80-82 1
```
Auto-validates setup (checks Ollama is running, models available), then runs evaluation set.

### Full Pipeline
```bash
# Terminal 1: Start Ollama service (keep running)
ollama serve

# Terminal 2: Run pipeline
bash scripts/ollama_pipeline.sh --bugs START-END --setup --run
```
Customizable pipeline. Use `--setup` to clone code, `--run` to execute, or combine both.

**Examples:**
```bash
# Setup only (prepare code)
bash scripts/ollama_pipeline.sh --bugs 1-10 --setup

# Run only (if already setup)
bash scripts/ollama_pipeline.sh --bugs 1-10 --run --skip-code

# Full (setup + run) with retries
bash scripts/ollama_pipeline.sh --bugs 1-106 --setup --run --max-attempts 2
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
- `--setup` alone: Prepare files for later runs
- `--run --skip-code`: Reprocess bugs without re-cloning (fast re-runs)
- Both together: Complete workflow from scratch

---

## Performance

Local inference speed depends heavily on your hardware:

| Hardware | Speed | Time for 106 bugs |
|----------|-------|-------------------|
| **CPU (i7/Ryzen 5)** | 3-5 min/bug | 8-15 hours |
| **GPU (RTX 3080)** | 1 min/bug | 2-3 hours |
| **GPU (RTX 4090)** | 30s/bug | ~1 hour |
| **Apple Silicon (M1/M2)** | 2-3 min/bug | 4-6 hours |

**Tips for faster inference:**
- GPU acceleration is a huge speedup. Check if your GPU is supported
- Reduce `--max-attempts` (default 1 is usually sufficient)
- Process bugs in smaller batches to monitor progress

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

All results are stored locally. No cloud dependencies.

---

## Requirements

| Requirement | Details |
|-------------|---------|
| **RAM** | 16GB minimum (for 7B models). 24GB+ recommended for GPU |
| **Disk** | ~30GB total (10GB models + 20GB datasets) |
| **CPU** | Modern multi-core (Intel i7+ / AMD Ryzen 5+) for acceptable speed |
| **GPU** | Optional but highly recommended. NVIDIA (CUDA) or Apple Silicon (Metal) |

**Storage check:**
```bash
du -sh .  # Check current directory size
ollama list  # Shows downloaded model sizes
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ollama: command not found` | Reinstall from https://ollama.ai/download or add to PATH |
| `Models not available` | Run `ollama pull qwen2.5:7b qwen2.5-coder:7b` again |
| `Connection refused` | Ollama not running - start with `ollama serve` in another terminal |
| `Out of memory` | Reduce batch size: `--bugs 1-5` or add more RAM. Check: `free -h` |
| `Slow performance` | Normal for CPU. Use GPU if available. Check: `ollama -h` for GPU setup |
| `Partial outputs` | Model may have stopped. Retry with `--max-attempts 2` or `--force-clone` |
