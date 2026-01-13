# RepGen Replication Artifact - Complete Package

## 📦 What's Included

This artifact provides a complete, production-ready system for replicating the ICSE'26 paper "Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent" (86% reproducibility).

---

## 🎯 Quick Start Commands

```bash
# One-liner to replicate the entire paper
export OPENAI_API_KEY="sk-..." && bash replicate.sh

# Or with custom options
bash replicate.sh --bug-start 1 --bug-end 106 --max-attempts 5

# Or quick test first (5 bugs)
bash quick_start.sh 1 5 5
```

---

## 📄 Documentation (Start Here!)

### For the Impatient (2 minutes)
📍 **[START_HERE.md](START_HERE.md)** - Absolute quickest path to replication
- 2-minute setup
- TL;DR commands
- Common issues & fixes

### For the Practical (10 minutes)
📍 **[REPLICATE_ME.md](REPLICATE_ME.md)** - Easy-to-follow guide
- Step-by-step instructions
- All command variants
- Expected results
- File structure overview

### For the Thorough (30+ minutes)
📍 **[REPLICATION_GUIDE.md](REPLICATION_GUIDE.md)** - Comprehensive guide
- Detailed setup for each OS
- API key configuration
- Advanced usage
- Parallel execution
- Statistical analysis
- Paper citation info

### Original Paper Info
📍 **[README.md](README.md)** - Original paper description
- Paper abstract
- Project structure
- Dependencies
- Original scripts

---

## 🚀 Replication Scripts (The Magic!)

### Main Replication Script
📍 **replicate.sh** (9.2 KB, 606 lines)
- Complete automated replication pipeline
- Environment setup
- Dependency installation
- API key configuration
- Dataset verification
- Experiment execution
- Result aggregation & reporting

**Usage:**
```bash
bash replicate.sh [--bug-start N] [--bug-end N] [--max-attempts N] [--skip-setup]
```

### Quick Start Wrapper
📍 **quick_start.sh** (1.6 KB)
- Simplified interface to `replicate.sh`
- Minimal configuration needed
- Perfect for first-time users

**Usage:**
```bash
bash quick_start.sh START_BUG END_BUG ATTEMPTS
bash quick_start.sh 1 5 5        # Test 5 bugs
bash quick_start.sh 1 106 5      # Full paper (106 bugs)
```

---

## 🔧 Main Tools

### Primary Reproduction Engine
📍 **tool_openai.py** (871 lines)
- Uses OpenAI API (GPT-4.1)
- Handles retrieval ablations
- Handles generation ablations
- Produces reproduction code
- Generates logs

**Usage:**
```bash
python tool_openai.py --bug_id="001" --max-attempts=5
```

### Ollama Version (Local Models)
📍 **tool.py** (823 lines)
- Uses local Ollama models
- No API key required
- Same functionality as tool_openai.py
- Models: qwen2.5:7b, qwen2.5-coder:7b, llama3-8b, deepseek-r1-7b

### Ablation Studies
📍 **run_ablations.py** (162 lines)
- Systematic ablation runner
- Tests different retrieval configurations
- Tests different generation configurations
- Comprehensive evaluation

**Usage:**
```bash
python run_ablations.py --start_bug_id 1 --end_bug_id 106 --max-gen-attempts 5
```

### Baseline Methods
📍 **baselines.py** (1126 lines)
- Implements baseline approaches
- Supports multiple models (Llama, DeepSeek, GPT, local)
- Different prompting techniques (zero-shot, few-shot, CoT)

**Usage:**
```bash
python baselines.py --bug_id "001" --model "qwen2.5-7b" --technique "zero_shot"
```

### Dataset Creation
📍 **dataset_creation.py** (117 lines)
- Creates dataset from GitHub issues
- Fetches issue bodies from GitHub API
- Clones repositories

---

## 📊 Supporting Files

### Configuration & Environment
- **requirements.txt** - All Python dependencies (annoy, numpy, torch, transformers, pandas, openai, etc.)
- **.env** template - Optional environment variables

### Original Scripts (for reference)
- **script.sh** - SLURM batch job generator
- **baseline_script.sh** - Baseline execution script

---

## 📁 Project Structure

```
ICSE26-RepGen/
│
├── 🎯 START_HERE.md                  # ⭐ Start with this (2 min)
├── 📋 REPLICATE_ME.md                # Easy overview (10 min)
├── 📖 REPLICATION_GUIDE.md           # Full guide (30+ min)
├── README.md                         # Original paper README
│
├── 🚀 REPLICATION SCRIPTS
├── replicate.sh                      # ⭐ Main script - RUN THIS!
├── quick_start.sh                    # Quick wrapper
├── script.sh                         # SLURM generator
├── baseline_script.sh                # Baseline runner
│
├── 🔧 TOOLS
├── tool_openai.py                    # ⭐ Main engine (GPT-4.1)
├── tool.py                           # Main engine (Ollama)
├── run_ablations.py                  # ⭐ Ablation studies
├── baselines.py                      # Baseline methods
├── dataset_creation.py               # Dataset creator
│
├── 📦 RETRIEVAL MODULE
├── retrieval/
│   ├── __init__.py
│   ├── config.py                     # Configuration
│   ├── pipeline.py                   # Main pipeline
│   ├── core/                         # Core functionality
│   │   ├── code_indexer.py
│   │   ├── dependency_analyzer.py
│   │   ├── module_analyzer.py
│   │   ├── training_code_detector.py
│   │   ├── utils.py
│   │   └── ...
│   └── models/                       # Model implementations
│       ├── hybrid_search.py
│       └── ...
│
├── 📊 DATASET
├── dataset/
│   ├── Dataset.csv                   # Metadata for all 106 bugs
│   ├── 001/
│   │   ├── bug_report/              # Original bug report
│   │   ├── code/                    # Project source code
│   │   ├── context/                 # Retrieved context
│   │   ├── plan/                    # Generation plan
│   │   ├── refined_bug_report/      # Processed bug report
│   │   ├── reproduction_code/       # ✨ Generated output
│   │   └── ablations/               # Ablation outputs
│   ├── 002/ ... 106/                # More bugs
│   └── ...
│
├── 📈 RESULTS
├── results/
│   ├── run_20260113_*/              # Auto-generated for each run
│   │   ├── summary.txt              # High-level results
│   │   ├── bug_001.log              # Log for each bug
│   │   └── ...
│   ├── ControlGroup.csv             # Baseline results
│   ├── ExperimentalGroup.csv        # RepGen results
│   └── Statistical_Tests_ICSE26.ipynb  # Analysis notebook
│
├── 📉 FIGURES
├── figures/
│   ├── parameter-tuning/
│   └── ...
│
└── requirements.txt                  # Python dependencies
```

---

## 🎯 What Gets Reproduced

Running the replication script produces:

✅ **For each of 106 bugs:**
- Generated reproduction code (`.py` files)
- Generation logs with detailed debugging info
- Plan generation results
- Context retrieval outputs
- Bug report refinements

✅ **Aggregated results:**
- Summary report with success metrics
- Success rate (~80.19%)
- Failure analysis
- Execution logs
- Performance statistics

✅ **Statistical outputs:**
- CSV files with detailed results
- Jupyter notebooks with analysis
- Comparison with baselines
- Ablation study results

---

## 📋 Prerequisites Checklist

- [ ] Python 3.12 installed
- [ ] pip available
- [ ] OpenAI API key (or use local models)
- [ ] Internet connection (for API calls)
- [ ] ~5-10 GB disk space
- [ ] 8+ GB RAM (16GB+ recommended)

---

## 🚦 Getting Started

### Path 1: Just Run It (Fastest) ⚡
```bash
export OPENAI_API_KEY="sk-..."
bash replicate.sh
# Takes ~2-4 hours for all 106 bugs
```

### Path 2: Test First (Recommended) 🧪
```bash
export OPENAI_API_KEY="sk-..."
bash quick_start.sh 1 5 5     # Test with 5 bugs first
bash quick_start.sh 1 106 5   # Then run full paper
```

### Path 3: Custom Configuration 🔧
```bash
# Read the guides first
cat START_HERE.md                # 2 min overview
cat REPLICATE_ME.md              # 10 min guide
cat REPLICATION_GUIDE.md         # Full details

# Then run with custom options
bash replicate.sh --bug-start 1 --bug-end 50 --max-attempts 10
```

---

## 📚 Documentation Guide

| Document | Time | For Whom | Content |
|----------|------|----------|---------|
| **START_HERE.md** | 2 min | Everyone | Quick start, TL;DR commands |
| **REPLICATE_ME.md** | 10 min | Practical users | Step-by-step guide with examples |
| **REPLICATION_GUIDE.md** | 30+ min | Detail-oriented | Comprehensive guide, all options |
| **README.md** | 15 min | Paper interest | Paper description, background |

---

## 🎓 Paper Information

**Title:** Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent

**Venue:** 48th ACM/IEEE International Conference on Software Engineering (ICSE'26)

**Preprint:** https://arxiv.org/abs/2512.14990

**Citation:**
```bibtex
@inproceedings{RepGen2026,
  title={Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent},
  author={Shah, Mehil and others},
  booktitle={48th ACM/IEEE International Conference on Software Engineering (ICSE'26)},
  year={2026}
}
```

---

## 🆘 Troubleshooting

**Issue: "Python 3.12 not found"**
- Solution: Install Python 3.12 from https://www.python.org/downloads/

**Issue: "OPENAI_API_KEY not set"**
- Solution: `export OPENAI_API_KEY="sk-..."` then retry

**Issue: "Module not found"**
- Solution: `pip install -r requirements.txt`

**Issue: "Slow or stalling"**
- Solution: Check OpenAI API status or use smaller bug range

👉 **More help:** See `REPLICATION_GUIDE.md` → Troubleshooting

---

## 📞 Support

1. **Quick questions?** → `START_HERE.md`
2. **How do I...?** → `REPLICATE_ME.md`
3. **Something broke** → `REPLICATION_GUIDE.md` (Troubleshooting)
4. **Background info** → `README.md`

---

## 🏆 Success Metrics

When you run the complete replication, expect:

- **Success Rate:** ~80.19% (85/106 bugs reproduced)
- **Average Time/Bug:** 1-2 minutes
- **Total Runtime:** ~2-4 hours
- **Generated Code:** 85+ Python reproduction scripts
- **Detailed Logs:** 106 log files with full execution traces

---

## 📦 What You Get

✅ Complete, working replication system
✅ Comprehensive documentation (4 guides)
✅ 3 main tools (OpenAI, Ollama, Baselines)
✅ Full dataset (106 real-world DL bugs)
✅ Automated result aggregation
✅ Statistical analysis infrastructure

**Everything needed to reproduce the paper from scratch!**

---

**Ready?** Start here: **[START_HERE.md](START_HERE.md)** ⭐

```bash
export OPENAI_API_KEY="sk-..." && bash replicate.sh
```

Good luck! 🚀
