# RepGen - Automated Deep Learning Bug Reproduction

This repository contains the code and tools for the paper:  
**"Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent"**  
Accepted by ICSE 2026.

**Supported Platforms:** macOS, Linux, Windows (Git Bash / WSL)

---

## Prerequisites

Before you start, ensure you have the following installed:

### Required
- **Python 3.12 or later** ([Download](https://www.python.org/downloads/))
  - Verify: `python3 --version`
- **Git** ([Download](https://git-scm.com/downloads))
  - Verify: `git --version`
- **Bash shell**
  - macOS/Linux: Included
  - Windows: [Git Bash](https://git-scm.com/download/win) or [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install)

### Required API Key
- **OpenAI API key** - Required for running the pipeline
  - Get it from: [OpenAI API Keys](https://platform.openai.com/api-keys)
  - Store safely: `export OPENAI_API_KEY="sk-..."`

### Optional
- Virtual environment tool (usually included with Python 3.3+)
- Disk space: ~50GB for full dataset + cloned repositories
- Internet connection for downloading models and repositories

---

## Installation

### Step 1: Clone or Download Repository

```bash
git clone <repository-url>
cd ICSE26-RepGen
```

Or download and extract the repository manually.

### Step 2: Create Virtual Environment

**macOS/Linux/WSL:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Git Bash):**
```bash
python3 -m venv venv
source venv/Scripts/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Expected packages:
- torch, transformers (for models)
- pandas, numpy (for data handling)
- scikit-learn, rank_bm25, sentence_transformers (for retrieval)
- openai (for LLM integration)
- And others (see requirements.txt)

### Step 4: Set Up API Keys

```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
```

To make it persistent:
- **macOS/Linux:** Add to `~/.bashrc` or `~/.zshrc`
- **Windows Git Bash:** Add to `~/.bashrc` in your Git Bash home
- **WSL:** Add to `~/.bashrc` in Ubuntu

---

## Quick Start

Once installed, you can start running:

```bash
# 1. Set your OpenAI API key (if not already set)
export OPENAI_API_KEY="sk-..."

# 2. Run the pipeline (setup + run bugs 80-82)
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --run

# Or use quick start script
bash scripts/quick_start.sh 80-82 1
```

### What This Does:
1. **Setup Phase:** Clones code repositories at specific commits
2. **Run Phase:** Generates bug reproduction code using LLM

---

## Platform-Specific Notes

### macOS/Linux
- Everything works out-of-the-box
- Use `source venv/bin/activate` for virtual environment

### Windows Users

**Option 1: Git Bash (Recommended)**
- Install [Git for Windows](https://git-scm.com/download/win)
- Always run commands from Git Bash terminal
- Use `source venv/Scripts/activate` for virtual environment
- For detailed setup: [WINDOWS_SETUP.md](WINDOWS_SETUP.md)

**Option 2: WSL 2**
- Install [WSL 2 with Ubuntu](https://learn.microsoft.com/en-us/windows/wsl/install)
- Use Linux commands normally
- For detailed setup: [WINDOWS_SETUP.md](WINDOWS_SETUP.md)

---

## Usage Guide

### Basic Usage

```bash
# Setup only (clone code, copy files)
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup

# Run only (execute pipeline on already-setup dataset)
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --run --skip-code

# Setup + Run (complete workflow)
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --run
```

### Common Options

| Option | Value | Description |
|--------|-------|-------------|
| `--bugs` | `RANGE` | **Required.** Bug IDs: `1-10`, `80-82`, or `80,81,82` |
| `--dataset` | `PATH` | Dataset path (default: `dataset`) |
| `--setup` | - | Setup phase: clone code and copy files |
| `--run` | - | Run phase: execute pipeline |
| `--skip-code` | - | Skip code cloning (if already done) |
| `--force-clone` | - | Force fresh repository clones |
| `--max-attempts` | `N` | Max generation attempts per bug (default: 1) |
| `--help` | - | Show help message |

### Examples

```bash
# Quick test: 1 bug, 1 attempt
bash scripts/quick_start.sh 80 1

# Reproduce paper: all 106 bugs
bash scripts/pipeline.sh --bugs 1-106 --setup --run --max-attempts 5

# Setup experimental dataset
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup

# Run only (after setup)
export OPENAI_API_KEY="sk-..."
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --run

# Process bugs in batches
bash scripts/pipeline.sh --bugs 1-50 --setup --run
bash scripts/pipeline.sh --bugs 51-106 --setup --run

# Custom configuration
bash scripts/pipeline.sh --bugs 80-82 --run \
    --max-attempts 3 \
    --force-clone
```

---

## Key Features

✅ **Full automation** - Clone repositories, copy files, generate reproduction code  
✅ **LLM integration** - OpenAI API support with fallback options  
✅ **Dataset support** - Original (dataset/) and experimental (ae_dataset/) datasets  
✅ **Flexible configuration** - Highly customizable via command-line options  
✅ **Efficient** - Repository caching reduces re-cloning  
✅ **Cross-platform** - Works on macOS, Linux, Windows (Git Bash/WSL)  
✅ **Modular phases** - Setup and run can be done independently  

---

## Project Structure

```
ICSE26-RepGen/
├── README.md                          ← Start here
├── PIPELINE.md                        ← Detailed pipeline docs
├── WINDOWS_SETUP.md                   ← Windows-specific guide
├── requirements.txt                   ← Python dependencies
│
├── scripts/
│   ├── pipeline.sh                    ← Main unified pipeline
│   ├── quick_start.sh                 ← Simplified entry point
│   ├── replicate.sh                   ← Full paper replication
│   ├── baseline_script.sh             ← Baseline runs
│   └── script.sh                      ← SLURM submission
│
├── src/
│   ├── tool_openai.py                 ← Main reproduction tool
│   ├── tool.py                        ← Core logic
│   ├── baselines.py                   ← Baseline implementations
│   ├── dataset_creation.py            ← Dataset utilities
│   └── run_ablations.py               ← Ablation experiments
│
├── retrieval/
│   ├── pipeline.py                    ← Code retrieval
│   ├── core/                          ← Retrieval models
│   └── models/                        ← Pretrained models
│
├── dataset/                           ← Original dataset (read-only)
│   └── 001-106/                       ← Bug folders
│       ├── bug_report/                ← Original bug reports
│       ├── code/                      ← Repository code
│       ├── context/                   ← Context files
│       └── ...
│
├── ae_dataset/                        ← Experimental dataset (generated)
│   └── 080-082/                       ← Your experimental bugs
│       ├── code/
│       ├── bug_report/
│       ├── context/
│       └── reproduction_code/         ← Generated outputs
│
├── results/                           ← Pipeline outputs
│   ├── run_YYYYMMDD_HHMMSS/           ← Run results
│   ├── ControlGroup.csv               ← Results summary
│   └── ...
│
└── logs/                              ← Pipeline logs
```

---

## Output Locations

After running, find outputs at:

**Experimental dataset (ae_dataset):**
```
ae_dataset/<bug_id>/reproduction_code/reproduce_<bug_id>.py
```

**Original dataset (dataset):**
```
dataset/<bug_id>/reproduction_code/reproduce_<bug_id>.py
```

**Summary results:**
```
results/run_YYYYMMDD_HHMMSS/
├── summary.txt                    ← Overall results
├── bug_001.log                    ← Per-bug logs
└── ...
```

---

## Troubleshooting

### Common Issues

**Q: `bash: command not found` on Windows**
- **A:** Install [Git Bash](https://git-scm.com/download/win). Use Git Bash terminal, not CMD.exe.

**Q: `python: command not found`**
- **A:** Install [Python 3.12+](https://www.python.org/downloads/) and ensure "Add to PATH" is checked.

**Q: `OPENAI_API_KEY not set` error**
- **A:** Set your key: `export OPENAI_API_KEY="sk-your-key-here"`

**Q: Virtual environment won't activate**
- **A:** Use correct path for your OS:
  - Windows: `source venv/Scripts/activate`
  - macOS/Linux: `source venv/bin/activate`

**Q: Git clone fails with SSL errors**
- **A:** Try: `git config --global http.sslVerify false`

**Q: Code directory missing for bug XXX**
- **A:** Run setup first: `bash scripts/pipeline.sh --bugs XXX --setup`

For more help, see:
- [PIPELINE.md](PIPELINE.md) - Complete usage guide
- [WINDOWS_SETUP.md](WINDOWS_SETUP.md) - Windows-specific setup

---

## Documentation

- **[PIPELINE.md](PIPELINE.md)** - Comprehensive pipeline guide
  - Detailed command reference
  - Advanced workflows
  - Performance tips
  - Complete troubleshooting

- **[WINDOWS_SETUP.md](WINDOWS_SETUP.md)** - Windows-specific guide
  - Step-by-step setup
  - Git Bash vs WSL comparison
  - Windows-specific troubleshooting

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.12 | 3.12+ |
| RAM | 8GB | 16GB+ |
| Disk | 20GB | 50GB+ |
| Internet | Required | High-speed |
| Git | Latest | Latest |

---

## Citation

If you use RepGen in your research, please cite:

```bibtex
@inproceedings{repgen2026,
  title={Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent},
  booktitle={Proceedings of the 48th International Conference on Software Engineering (ICSE 2026)},
  year={2026}
}
```

---

## License

[Specify your license here]

---

## Contact & Support

For issues, questions, or contributions:
- Open an issue on GitHub
- See [PIPELINE.md](PIPELINE.md#troubleshooting) for troubleshooting
- Check [WINDOWS_SETUP.md](WINDOWS_SETUP.md) if on Windows

---

## Quick Reference

**First time setup (1 time):**
```bash
python3 -m venv venv
source venv/bin/activate      # or: source venv/Scripts/activate (Windows)
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

**Run pipeline (repeatable):**
```bash
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --run
```

**View results:**
```bash
ls ae_dataset/080/reproduction_code/
cat ae_dataset/080/reproduction_code/reproduce_080.py
```

---

**Happy bug reproducing! 🚀**
- **Using dataset:** `dataset/<bug_id>/reproduction_code/reproduce_<bug_id>.py`

## More Information

For detailed documentation, see **[PIPELINE.md](PIPELINE.md)**

---

**Repository:** ICSE26-RepGen  
**Contact:** [Your contact info]
