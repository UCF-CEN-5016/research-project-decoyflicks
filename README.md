# RepGen - Bug Reproduction with LLMs

ICSE 2026: *Reproducing Deep Learning Bugs Leveraging Intelligent Agents*

**What is RepGen?**  
RepGen automatically generates bug reproduction code for deep learning frameworks. It uses an intelligent agent pipeline to analyze bug reports, extract relevant context, generate plans, and produce executable reproduction scripts.

**How it works:**
1. **Retrieval Phase** - Extracts relevant code context from the buggy repository
2. **Planning Phase** - Creates execution plans based on bug description and context
3. **Code Generation** - Generates reproduction code with feedback loops for refinement

---

## 🚀 Quick Start (Choose One)

### Option A: Cloud (OpenAI) - Fastest

Use OpenAI's GPT-4 models for fast, accurate inference. Best for quick tests or small batches. Requires internet connection and API credits.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."

bash scripts/quick_start.sh 80-82 1
```

**⏱ Setup:** 15 min | **Speed:** 30s/bug | **💰 Cost:** $50-100 total (all 106 bugs)

**→ See:** [PIPELINE.md](PIPELINE.md) for full reference

---

### Option B: Local (Ollama) - Free

Run inference locally using open-source Qwen2.5 models via Ollama. No internet or API costs. Slower but completely private. Requires 16GB+ RAM.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b qwen2.5-coder:7b

# Run in 2 terminals:
# Terminal 1: ollama serve
# Terminal 2: bash scripts/ollama_quick_start.sh 80-82 1
```

**⏱ Setup:** 45 min | **Speed:** 3-5 min/bug | **💰 Cost:** Free

**→ See:** [OLLAMA_SETUP.md](OLLAMA_SETUP.md) for full reference

---

## Requirements

| | |
|---|---|
| **Python** | 3.12+ |
| **Shell** | Bash |
| **For Ollama** | [Download](https://ollama.ai/download) + 16GB RAM |
| **For Cloud** | [OpenAI API key](https://platform.openai.com/api-keys) |
| **Windows?** | See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) |

---

## Common Commands

**For first-time users:** Start with quick start scripts above. These handle setup + run automatically.

```bash
# Quick test (cloud)
bash scripts/quick_start.sh 80-82 1

# Quick test (local)
bash scripts/ollama_quick_start.sh 80-82 1

# Advanced: Setup only (don't run yet)
bash scripts/pipeline.sh --bugs 1-50 --dataset dataset --setup

# Advanced: Run only (if already setup)
bash scripts/pipeline.sh --bugs 1-50 --dataset dataset --run --skip-code

# Advanced: Full pipeline (setup + run)
bash scripts/pipeline.sh --bugs 1-106 --dataset dataset --setup --run
```

---

## Dataset

```
dataset/           # Main dataset (001-106)
  001/, 002/, ...
  bug_report/      # Original bug description
  code/            # Buggy code
  reproduction_code/  # Generated bug reproduction

ae_dataset/        # Evaluation cases
```

---

## 📖 Documentation

| Link | Purpose |
|------|---------|
| [PIPELINE.md](PIPELINE.md) | Cloud (OpenAI) setup, reference, troubleshooting |
| [OLLAMA_SETUP.md](OLLAMA_SETUP.md) | Local (Ollama) setup, reference, troubleshooting, performance |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | Windows-specific configuration |
