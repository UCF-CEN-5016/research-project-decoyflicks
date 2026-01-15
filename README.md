# RepGen - Bug Reproduction with LLMs

ICSE 2026: *Reproducing Deep Learning Bugs Leveraging Intelligent Agents*

**What is RepGen?**  
RepGen automatically generates bug reproduction code for deep learning systems. It uses an intelligent agent pipeline to analyze bug reports, extract relevant context, generate plans, and produce executable reproduction scripts.

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

bash scripts/quick-start/cloud.sh 80-82 1
```

**⏱ Setup:** 15 min | **Speed:** 30s/bug | **💰 Cost:** $50-100 total (all 106 bugs)

**→ Full Reference:** [OPENAI_PIPELINE.md](OPENAI_PIPELINE.md)  
**→ Script Details:** [scripts/README.md](scripts/README.md#quick-start-scripts)

---

### Option B: Local (Ollama) - Free

Run inference locally using open-source Qwen2.5 models via Ollama. No internet or API costs. Slower but completely private. Requires 16GB+ RAM.

```bash
# Terminal 1: Start Ollama service
ollama serve

# Terminal 2: Setup + Run (in separate terminal)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash scripts/quick-start/local.sh 80-82 1
```

**⏱ Setup:** 45 min | **Speed:** 3-5 min/bug | **💰 Cost:** Free

**→ Full Reference:** [QWEN_PIPELINE.md](QWEN_PIPELINE.md)  
**→ Script Details:** [scripts/README.md](scripts/README.md#quick-start-scripts)

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

**For first-time users:** Use quick start scripts in `scripts/quick-start/`. They handle all setup and execution automatically.

### Quick Tests (Recommended for first run)
```bash
# Cloud: Test on 2 bugs with 1 attempt
bash scripts/quick-start/cloud.sh 80-82 1

# Local (Ollama): Test on 2 bugs with 1 attempt
bash scripts/quick-start/local.sh 80-82 1
```

### Full Replication (Paper Results)
```bash
# Cloud: All 106 bugs with 5 retry attempts
bash scripts/quick-start/cloud.sh 1-106 5

# Local: All 106 bugs with 5 retry attempts
bash scripts/quick-start/local.sh 1-106 5
```

### Advanced: Custom Ranges with Pipeline Scripts
For more control, use pipeline scripts in `scripts/pipeline/`:

```bash
# Cloud: Setup only (don't run yet)
bash scripts/pipeline/cloud.sh --bugs 1-50 --dataset dataset --setup

# Cloud: Run only (if already setup)
bash scripts/pipeline/cloud.sh --bugs 1-50 --dataset dataset --run --skip-code

# Cloud: Full pipeline (setup + run) with retries
bash scripts/pipeline/cloud.sh --bugs 1-106 --dataset dataset --setup --run --max-attempts 3

# Local: Same commands but with scripts/pipeline/local.sh
bash scripts/pipeline/local.sh --bugs 1-10 --dataset ae_dataset --setup --run
```

### Experimental Runs
```bash
# Run ablation studies (requires OpenAI API)
bash scripts/experimental/ablations.sh 80 82

# Run baseline comparisons across models
bash scripts/experimental/baseline.sh 80 82
```

**→ Full script reference:** [scripts/README.md](scripts/README.md)

---

## Dataset

The project includes two datasets:

### Main Dataset (`dataset/`)
- **Bugs:** 001-106 (reproducible deep learning bugs)
- **Structure per bug:**
  ```
  dataset/BUG_ID/
  ├── bug_report/           # Original bug report text
  ├── context/              # Relevant code context files
  ├── plan/                 # (Generated) Execution plan
  ├── reproduction_code/    # (Generated) Reproduction script
  ├── refined_bug_report/   # (Generated) Refined description
  └── ablations/            # (Experimental) Ablation results
  ```

### Evaluation Dataset (`ae_dataset/`)
- Smaller subset for quick testing and validation
- Used in quick-start scripts by default

### Dataset Metadata
- **dataset/Dataset.csv** - Bug metadata and attributes
- **results/ExperimentalGroup.csv** - Results with API usage metrics
- **results/ControlGroup.csv** - Baseline results
- **results/Statistical_Tests_ICSE26.ipynb** - Statistical analysis

---

## 📖 Full Documentation

| Document | Purpose |
|----------|---------|
| [scripts/README.md](scripts/README.md) | **Script reference** - All available scripts, options, and workflows |
| [OPENAI_PIPELINE.md](OPENAI_PIPELINE.md) | **Cloud (OpenAI)** - Setup, advanced options, cost, troubleshooting |
| [QWEN_PIPELINE.md](QWEN_PIPELINE.md) | **Local (Qwen/Ollama)** - Setup, model configuration, performance tips |
| [WINDOWS_SETUP.md](WINDOWS_SETUP.md) | **Windows-specific** - WSL/Git Bash setup, compatibility notes |

---

## 🏗️ Project Structure

```
ICSE26-RepGen/
├── README.md                  # This file
├── OPENAI_PIPELINE.md         # Cloud (OpenAI) guide
├── QWEN_PIPELINE.md           # Local (Qwen/Ollama) guide
├── WINDOWS_SETUP.md           # Windows setup
├── requirements.txt           # Python dependencies
│
├── scripts/                  # Shell scripts (organized by purpose)
│   ├── README.md            # Script reference guide
│   ├── quick-start/         # Entry point scripts
│   │   ├── cloud.sh         # Quick test with OpenAI
│   │   └── local.sh         # Quick test with Ollama
│   ├── pipeline/            # Full pipeline with options
│   │   ├── cloud.sh         # Full cloud pipeline
│   │   └── local.sh         # Full local pipeline
│   └── experimental/        # Ablations and baselines
│       ├── ablations.sh     # Ablation studies
│       └── baseline.sh      # Baseline comparisons
│
├── src/                     # Python source code
│   ├── tool.py             # Main reproduction tool
│   ├── tool_openai.py      # OpenAI-specific implementation
│   ├── baselines.py        # Baseline implementations
│   └── run_ablations.py    # Ablation experiment runner
│
├── retrieval/              # Retrieval module
│   ├── pipeline.py         # Main retrieval pipeline
│   ├── core/               # Core retrieval components
│   ├── models/             # Neural ranking models
│   └── config.py           # Configuration
│
├── dataset/                # Main dataset (001-106 bugs)
│   ├── Dataset.csv        # Dataset metadata
│   └── 001/, 002/, ...
│
├── figures/               # Plots and visualizations
├── results/               # Output results and metrics
└── .code_cache/          # (Generated) Repository cache
```

---

## 🔄 Pipeline Workflow

```
1. RETRIEVAL PHASE
   ├── Extract relevant code files from repository
   ├── BM25 search + ANN ranking
   ├── Training loop detection & ranking
   └── Dependency extraction & module analysis

2. PLANNING PHASE
   ├── Analyze bug report + code context
   ├── Generate execution plan
   └── Identify test case requirements

3. CODE GENERATION PHASE
   ├── Generate reproduction code
   ├── Compilation checking
   ├── Static analysis validation
   └── Runtime feedback loops
       └── Refine code if execution fails
```

---

## 📊 Expected Results

After running the pipeline, check:
- **Reproduction code:** `dataset/BUG_ID/reproduction_code/reproduce_BUG_ID.py`
- **Execution plan:** `dataset/BUG_ID/plan/plan_BUG_ID.txt`
- **API metrics:** `results/ExperimentalGroup.csv`
- **Logs:** `results/logs_*.txt`

---

## 💡 Tips & Best Practices

### Getting Started
1. **First run?** Start with quick-start scripts: `bash scripts/quick-start/cloud.sh 80-82 1`
2. **Test locally first** before running full dataset
3. **Check logs** in `results/` directory for debugging

### Performance
- **Cloud:** ~30 seconds per bug (with OpenAI GPT-4)
- **Local:** ~3-5 minutes per bug (with Qwen2.5 on CPU)
- **Parallelization:** Scripts process bugs sequentially; use multiple machines for parallelization

### Cost Management
- **Cloud:** ~$0.50-1 per bug; budget $50-100 for full dataset
- **Local:** Free after initial setup (~10 GB disk space)
- **Use retries:** `--max-attempts 3` improves success rate with marginal cost increase

### Debugging
- **Add `--quiet` flag** to reduce output volume
- **Run subset first:** Test with `--bugs 80-82` before full runs
- **Check logs:** Pipeline creates detailed logs in `results/` directory
- **Verify setup:** For Ollama, confirm models are loaded with `ollama list`

### Windows
- Use Git Bash or WSL2 (scripts use Bash)
- See [WINDOWS_SETUP.md](WINDOWS_SETUP.md) for platform-specific issues

---

## 📚 References

**Paper:** Reproducing Deep Learning Bugs Leveraging Intelligent Agents (ICSE 2026)

**Citation:**
```bibtex
[Citation to be added on publication]
```

---

## ❓ Troubleshooting

**Issue:** `command not found: bash scripts/quick_start.sh`  
**Solution:** Script paths changed. Use: `bash scripts/quick-start/cloud.sh` (note: `quick-start` not `quick_start`)

**Issue:** Python errors in pipeline  
**Solution:** Check virtual environment is activated and dependencies installed: `pip install -r requirements.txt`

**Issue:** Ollama connection errors  
**Solution:** Ensure Ollama is running: `ollama serve` in separate terminal, then retry

**Issue:** High API costs  
**Solution:** Test on smaller subset first with `--bugs 80-82`, use `--max-attempts 1` instead of 5

**Issue:** Out of memory  
**Solution:** Reduce batch size or run on smaller bug ranges; for cloud, no local memory needed
