# RepGen - ICSE'26 Paper Replication Artifact

**Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent**

This is a complete, production-ready replication package for the ICSE'26 paper.

## 🚀 Quick Start

```bash
export OPENAI_API_KEY="sk-..." && bash scripts/replicate.sh
```

## 📚 Documentation

**Pick your starting point:**

- **[START_HERE.md](START_HERE.md)** ⭐ - 2-minute quick start (RECOMMENDED)
- **[STRUCTURE.md](STRUCTURE.md)** - Directory structure & file organization
- **[REPLICATE_ME.md](REPLICATE_ME.md)** - Easy-to-follow guide
- **[REPLICATION_GUIDE.md](REPLICATION_GUIDE.md)** - Comprehensive guide with troubleshooting
- **[ARTIFACT_README.md](ARTIFACT_README.md)** - Complete artifact overview
- **[DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)** - What was created

## 📁 Directory Structure

```
ICSE26-RepGen/
├── 📚 Documentation (6 guides)
├── 📜 scripts/              # Executable scripts
│   ├── replicate.sh        # Main replication pipeline ⭐
│   ├── quick_start.sh      # Quick wrapper for testing
│   ├── script.sh           # SLURM batch generator
│   └── baseline_script.sh  # Baseline runner
│
├── 🛠️  src/                 # Python tools
│   ├── tool_openai.py      # Main engine (GPT-4.1) ⭐
│   ├── tool.py             # Main engine (Ollama/local)
│   ├── run_ablations.py    # Ablation studies
│   ├── baselines.py        # Baseline methods
│   └── dataset_creation.py # Dataset creator
│
├── 📦 retrieval/           # Retrieval module
├── 📊 dataset/             # 106 bugs + metadata
├── results/                # Generated results
└── figures/                # Paper figures
```

## 🎯 What Can You Do

✅ **Replicate entire paper** - All 106 bugs with one command  
✅ **Test first** - Run with 5 bugs before full replication  
✅ **Customize** - Run specific bug ranges and parameters  
✅ **Use any LLM** - OpenAI, Groq, DeepSeek, or local models  
✅ **Get results** - Auto-generated logs and summary reports  

## 📊 Expected Results

- **Success Rate**: ~80.19% (85/106 bugs)
- **Time per Bug**: 1-2 minutes
- **Total Runtime**: 2-4 hours
- **Generated**: 85+ reproduction scripts, 106+ detailed logs

## 🚀 Commands

### Main Replication
```bash
# Fast (full paper)
export OPENAI_API_KEY="sk-..." && bash scripts/replicate.sh

# Safe (test first)
bash scripts/quick_start.sh 1 5 5       # Test
bash scripts/quick_start.sh 1 106 5     # Full

# Custom range
bash scripts/replicate.sh --bug-start 1 --bug-end 50

# Get help
bash scripts/replicate.sh --help
```

### Individual Tools
```bash
# Main tool (OpenAI)
python src/tool_openai.py --bug_id="001"

# Main tool (Local)
python src/tool.py --bug_id="001"

# Ablations
python src/run_ablations.py --start_bug_id 1 --end_bug_id 106

# Baselines
python src/baselines.py --bug_id "001" --model "qwen2.5-7b"
```

## ⚙️ Requirements

- Python 3.12
- OpenAI API key (from https://platform.openai.com/api-keys)
- ~5-10 GB disk space
- 8+ GB RAM (16GB+ recommended)
- Internet connection

**Alternative**: Use Ollama for local models (no API key needed)

## 📖 Next Steps

1. **Read**: `cat START_HERE.md` (2 minutes)
2. **Setup**: `export OPENAI_API_KEY="sk-..."`
3. **Test**: `bash scripts/quick_start.sh 1 5 5`
4. **Run**: `bash scripts/replicate.sh`

## 🔗 Files Overview

| File | Purpose |
|------|---------|
| `scripts/replicate.sh` | Main replication pipeline |
| `src/tool_openai.py` | Core reproduction engine |
| `START_HERE.md` | Quick start guide |
| `REPLICATION_GUIDE.md` | Comprehensive guide |
| `retrieval/` | Context retrieval system |
| `dataset/` | 106 real-world bugs |

## 📚 Full Documentation

- [START_HERE.md](START_HERE.md) - Entry point (2 min)
- [REPLICATE_ME.md](REPLICATE_ME.md) - Easy guide (10 min)
- [REPLICATION_GUIDE.md](REPLICATION_GUIDE.md) - Full details (30+ min)
- [STRUCTURE.md](STRUCTURE.md) - Directory structure
- [ARTIFACT_README.md](ARTIFACT_README.md) - Complete overview

## 🎓 Paper Information

**Title**: Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent

**Venue**: ICSE'26 (48th ACM/IEEE International Conference on Software Engineering)

**Preprint**: https://arxiv.org/abs/2512.14990

**Citation**:
```bibtex
@inproceedings{RepGen2026,
  title={Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent},
  author={Shah, Mehil and others},
  booktitle={48th ACM/IEEE International Conference on Software Engineering (ICSE'26)},
  year={2026}
}
```

## ✨ Highlights

✓ **Fully Automated** - One command runs everything  
✓ **Well Organized** - Scripts, tools, and docs separated  
✓ **Production Ready** - 600+ lines of robust code  
✓ **Comprehensive Docs** - 5+ detailed guides  
✓ **Flexible** - Works with multiple LLM backends  
✓ **Result Tracking** - Auto-generates reports  

---

**Ready to replicate?** Start with: `cat START_HERE.md`

```bash
export OPENAI_API_KEY="sk-..." && bash scripts/replicate.sh
```
