# 🚀 START HERE - RepGen Paper Replication

**TL;DR** - Run this command to replicate the entire ICSE'26 paper in one line:

```bash
export OPENAI_API_KEY="sk-..." && bash scripts/replicate.sh
```

---

## 2-Minute Setup

### Step 1: Check Python (30 seconds)
```bash
python3 --version  # Must show 3.12.x
```
❌ If not 3.12, [install it](https://www.python.org/downloads/)

### Step 2: Get API Key (1 minute)
1. Go to: https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-`)
4. Run: `export OPENAI_API_KEY="sk-your-key-here"`

### Step 3: Run (30 seconds)
```bash
bash scripts/replicate.sh
```

Done! ✅

---

## What You're Getting

**Replication Scripts:**
- `replicate.sh` - Complete automated replication (606 lines)
- `quick_start.sh` - Quick version (40 lines)

**Documentation:**
- `REPLICATE_ME.md` - Easy overview with examples
- `REPLICATION_GUIDE.md` - Full detailed guide (1000+ lines)
- `README.md` - Original paper description

**Tools:**
- `tool_openai.py` - Main reproduction engine
- `run_ablations.py` - Ablation studies
- `baselines.py` - Baseline comparisons

---

## Testing First? (5 minutes)

Run on just 5 bugs first to test your setup:

```bash
export OPENAI_API_KEY="sk-..."
bash scripts/quick_start.sh 1 5 5
```

✅ If this works, run the full paper:
```bash
bash scripts/quick_start.sh 1 106 5
```

---

## What Happens When You Run It

The `replicate.sh` script automatically:

1. ✅ **Sets up Python environment** - Creates virtual environment
2. ✅ **Installs dependencies** - From `requirements.txt`
3. ✅ **Configures API keys** - OPENAI_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY
4. ✅ **Verifies dataset** - Checks all 106 bugs are present
5. ✅ **Runs experiments** - Processes each bug
6. ✅ **Generates reports** - Creates summary and logs

**Total time:** ~2-4 hours (for all 106 bugs)

---

## Expected Results

```
RepGen - Experiment Run Summary
==============================
Results:
  - Successful: 85
  - Failed: 21
  - Success rate: 80.19%
```

---

## File Guide

Pick what you need:

| Need | Read |
|------|------|
| **Just run it** | You're here! 👈 |
| **Quick overview** | `REPLICATE_ME.md` (5 min read) |
| **Full details** | `REPLICATION_GUIDE.md` (30 min read) |
| **Troubleshoot** | `REPLICATION_GUIDE.md` → Troubleshooting |
| **Paper info** | `README.md` |

---

## Common Issues

### "Python 3.12 not found"
```bash
# Install:
brew install python@3.12  # macOS
# or
sudo apt-get install python3.12  # Linux
```

### "OPENAI_API_KEY not set"
```bash
# Check if set:
echo $OPENAI_API_KEY

# Set it:
export OPENAI_API_KEY="sk-..."

# Make permanent (add to ~/.bashrc):
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

### "Slow/stalling"
```bash
# Check API status:
curl https://status.openai.com/

# Try smaller batch:
bash replicate.sh --bug-start 1 --bug-end 10
```

💡 **More issues?** See `REPLICATION_GUIDE.md` → Troubleshooting

---

## Command Variants

```bash
# ✅ Full paper (all 106 bugs)
bash replicate.sh

# ✅ Test first (5 bugs)
bash quick_start.sh 1 5 5

# ✅ Custom range
bash replicate.sh --bug-start 1 --bug-end 20

# ✅ More attempts (default: 5)
bash replicate.sh --max-attempts 10

# ✅ Skip environment setup (if already done)
bash replicate.sh --skip-setup

# ✅ Set API key directly
bash replicate.sh --openai-api-key "sk-..."

# ✅ Get help
bash replicate.sh --help
```

---

## Results Location

After running, check results here:

```bash
# View summary
cat results/run_*/summary.txt

# View specific bug result
cat dataset/001/reproduction_code/001.py

# View logs
cat results/run_*/bug_001.log
```

---

## Alternative: Use Local Models (No API Key)

Don't have an API key? Use local models:

```bash
# Install Ollama: https://ollama.ai

# Pull model:
ollama pull qwen2.5:7b

# Keep Ollama running:
ollama serve &

# Script will automatically use local models
bash replicate.sh
```

No API costs, all local! 🎉

---

## What Gets Generated

For each bug, you'll get:

```
dataset/001/
├── bug_report/           # Original bug report
├── code/                 # Project code
├── context/              # Retrieved context
├── plan/                 # Generation plan
├── refined_bug_report/   # Processed bug report
└── reproduction_code/    # ✨ Generated reproduction code
    └── 001.py            # This is the output!
```

---

## Next Steps

1. **Ready?** Run: `bash replicate.sh`
2. **Questions?** Read: `REPLICATE_ME.md` or `REPLICATION_GUIDE.md`
3. **Stuck?** Check: `REPLICATION_GUIDE.md` → Troubleshooting
4. **Learn more?** Read: `README.md` (original paper info)

---

## Paper Citation

If you use this, cite:

```bibtex
@inproceedings{RepGen2026,
  title={Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent},
  author={Shah, Mehil and others},
  booktitle={ICSE'26},
  year={2026}
}
```

Preprint: https://arxiv.org/abs/2512.14990

---

## One More Thing

This artifact includes:

✅ Complete replication script (`replicate.sh`)
✅ Quick start wrapper (`quick_start.sh`)
✅ Full documentation (3 guides)
✅ All source code (`tool_openai.py`, `tool.py`, etc.)
✅ Dataset (106 bugs in `dataset/` folder)
✅ Results infrastructure (auto-generates reports)

**Everything you need to reproduce the paper is here!**

---

**Let's go!** 🚀

```bash
export OPENAI_API_KEY="sk-..." && bash replicate.sh
```

Good luck! 🎉
