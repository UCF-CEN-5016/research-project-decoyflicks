# RepGen - ICSE'26 Paper Replication

## One-Line Replication

```bash
export OPENAI_API_KEY="sk-..." && bash replicate.sh --bug-start 1 --bug-end 106
```

**That's it!** This single command will:
- Set up your Python environment
- Install all dependencies
- Configure API keys
- Run the complete paper replication (106 bugs)
- Generate results and summary reports

---

## Quick Start (5 minutes)

### 1. Prerequisites
```bash
# Check Python version (must be 3.12+)
python3 --version

# Get OpenAI API key from: https://platform.openai.com/api-keys
```

### 2. Set API Key
```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
```

### 3. Run Replication
```bash
# Test with 5 bugs first
bash quick_start.sh 1 5 5

# Or run full paper (106 bugs)
bash quick_start.sh 1 106 5
```

### 4. Check Results
```bash
# View summary
cat results/run_*/summary.txt

# Check specific bug results
cat dataset/001/reproduction_code/001.py
```

---

## Script Overview

### 📋 Main Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **replicate.sh** | Complete automated replication | `bash replicate.sh [options]` |
| **quick_start.sh** | Simplified version | `bash quick_start.sh START END ATTEMPTS` |
| **tool_openai.py** | Main reproduction engine | `python tool_openai.py --bug_id=001` |
| **run_ablations.py** | Ablation studies | `python run_ablations.py --start_bug_id=1 --end_bug_id=106` |

### 🔧 Configuration

**`replicate.sh` options:**
```bash
bash replicate.sh \
  --bug-start 1           # First bug ID
  --bug-end 106           # Last bug ID  
  --max-attempts 5        # Max generation attempts per bug
  --skip-setup            # Skip environment setup
  --openai-api-key KEY    # Set API key directly
  --help                  # Show help message
```

---

## Detailed Setup Instructions

### Step 1: Install Python 3.12

**macOS:**
```bash
brew install python@3.12
ln -s /usr/local/bin/python3.12 /usr/local/bin/python3
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install python3.12 python3.12-venv python3.12-dev
```

**Windows/WSL:**
```bash
# Install from https://www.python.org/downloads/
# Or using WSL: apt-get install python3.12
```

### Step 2: Verify Installation
```bash
python3 --version  # Should show 3.12.x
pip3 --version     # Should be available
```

### Step 3: Get API Keys (Optional)

Choose based on which models you want to use:

**OpenAI (GPT-4.1) - Recommended:**
1. Go to https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key
4. Set: `export OPENAI_API_KEY="sk-..."`

**Alternative: Use Local Models (No API key needed):**
1. Install Ollama: https://ollama.ai
2. Pull model: `ollama pull qwen2.5:7b`
3. Keep Ollama running in background
4. Script will use local models automatically

### Step 4: Run Replication

**Option A: Full Paper (106 bugs, ~2-4 hours)**
```bash
export OPENAI_API_KEY="sk-..."
bash replicate.sh
```

**Option B: Test First (5 bugs, ~5 minutes)**
```bash
export OPENAI_API_KEY="sk-..."
bash quick_start.sh 1 5 5
```

**Option C: Custom Range**
```bash
export OPENAI_API_KEY="sk-..."
bash replicate.sh --bug-start 1 --bug-end 20 --max-attempts 5
```

---

## Understanding Output

After running, you'll see:

```
results/run_20260113_120000/
├── summary.txt           # High-level results
├── bug_001.log          # Log for each bug
├── bug_002.log
└── ...
```

### Summary Report Example
```
RepGen - Experiment Run Summary
==============================
Timestamp: Wed Jan 13 12:00:00 PST 2026
Configuration:
  - Bug range: 1-106
  - Max attempts: 5
  - Total bugs: 106

Results:
  - Successful: 85
  - Failed: 21
  - Success rate: 80.19%
```

### Generated Reproduction Code
For each successful bug:
```bash
cat dataset/001/reproduction_code/001.py
```

---

## Troubleshooting

### Problem: "Python 3.12 not found"
```bash
# Verify installation
python3 --version
python3.12 --version

# If not installed, install it:
brew install python@3.12  # macOS
# or
sudo apt-get install python3.12  # Linux
```

### Problem: "OPENAI_API_KEY not set"
```bash
# Check if set
echo $OPENAI_API_KEY

# Set it
export OPENAI_API_KEY="sk-your-actual-key"

# Make permanent (add to ~/.bashrc or ~/.zshrc):
echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.bashrc
source ~/.bashrc
```

### Problem: "Module not found" errors
```bash
# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print('OK')"
```

### Problem: Slow or stalling
```bash
# Check API status
curl https://status.openai.com/

# Check your rate limits
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Try smaller batch:
bash replicate.sh --bug-start 1 --bug-end 10
```

### Problem: GPU/CUDA errors
```bash
# Run on CPU instead
export CUDA_VISIBLE_DEVICES="-1"
bash replicate.sh
```

---

## Advanced Usage

### Run Ablations
```bash
python3 run_ablations.py \
  --start_bug_id 1 \
  --end_bug_id 106 \
  --max-gen-attempts 5
```

### Run Baselines
```bash
python3 baselines.py \
  --bug_id "001" \
  --model "qwen2.5-7b" \
  --technique "zero_shot"
```

### Manual Reproduction (single bug)
```bash
python3 tool_openai.py \
  --bug_id="001" \
  --max-attempts=5 \
  --retrieval_ablation="full_system" \
  --generation_ablation="all_steps"
```

### View Detailed Logs
```bash
# Check what happened for a specific bug
tail -f results/run_*/bug_001.log

# Check for errors
grep -i "error" results/run_*/bug_*.log

# Count successes
grep -c "success" results/run_*/bug_*.log
```

---

## Expected Results

Based on the paper, you should achieve:

| Metric | Expected Value |
|--------|-----------------|
| Overall Success Rate | ~80.19% |
| Time per Bug | ~1-2 minutes |
| Total Runtime (106 bugs) | ~2-4 hours |
| Number of Successful Reproductions | ~85/106 |

---

## Paper Information

**Title:** Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent

**Venue:** 48th ACM/IEEE International Conference on Software Engineering (ICSE'26)

**Preprint:** https://arxiv.org/abs/2512.14990

**Authors:** Mehil Shah, et al.

### Citation
```bibtex
@inproceedings{RepGen2026,
  title={Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent},
  author={Shah, Mehil and others},
  booktitle={48th ACM/IEEE International Conference on Software Engineering (ICSE'26)},
  year={2026}
}
```

---

## Files in This Repository

```
/
├── replicate.sh                      # Main replication script
├── quick_start.sh                    # Quick-start wrapper
├── REPLICATION_GUIDE.md              # Detailed guide
├── README.md                         # Original paper README
├── tool_openai.py                    # Main reproduction tool (OpenAI)
├── tool.py                           # Main reproduction tool (Ollama)
├── run_ablations.py                  # Ablation studies
├── baselines.py                      # Baseline methods
├── requirements.txt                  # Python dependencies
├── dataset/                          # Bug dataset (106 bugs)
│   ├── Dataset.csv                  # Metadata
│   ├── 001/
│   │   ├── bug_report/
│   │   ├── code/
│   │   ├── context/
│   │   ├── plan/
│   │   ├── refined_bug_report/
│   │   └── reproduction_code/       # Generated output
│   ├── 002/
│   └── ...
├── retrieval/                        # Retrieval module
├── results/                          # Results directory
│   ├── run_YYYYMMDD_HHMMSS/         # Results from each run
│   │   ├── summary.txt
│   │   └── bug_*.log
│   ├── ControlGroup.csv
│   ├── ExperimentalGroup.csv
│   └── Statistical_Tests_ICSE26.ipynb
└── figures/                          # Paper figures
```

---

## Getting Help

1. **Read the error message carefully** - logs usually indicate the exact problem
2. **Check `REPLICATION_GUIDE.md`** - detailed troubleshooting section
3. **Review log files** - each bug has detailed logs in `results/run_*/`
4. **Verify prerequisites** - ensure Python 3.12 and API keys are set correctly
5. **Test with fewer bugs first** - use `bash quick_start.sh 1 5 5` to test

---

## Support Commands

```bash
# Check Python version
python3 --version

# Check pip packages
pip list | grep -E "(torch|openai|transformers)"

# Check API key is set
echo "API Key set: $([[ -n "$OPENAI_API_KEY" ]] && echo 'YES' || echo 'NO')"

# View recent results
ls -lth results/

# Show latest summary
cat $(ls -t results/run_*/summary.txt | head -1)

# Count successful bugs in latest run
grep -c "SUCCESS" $(ls -d results/run_* | tail -1)/bug_*.log
```

---

## Performance Notes

- **First run is slower** due to environment setup
- **API rate limiting**: OpenAI has rate limits (check your account)
- **Network**: Stable internet connection recommended
- **Storage**: Requires ~5-10GB for dataset + outputs
- **Memory**: 8GB minimum, 16GB+ recommended

---

## Customization

To modify default behavior, edit `replicate.sh`:

```bash
# Change default bug range
BUG_START=1
BUG_END=106

# Change max attempts per bug
MAX_ATTEMPTS=5

# Change Python version requirement
PYTHON_VERSION="3.12"
```

---

**Happy replicating! 🚀**

For questions or issues, refer to the detailed `REPLICATION_GUIDE.md` or the original paper.

Last updated: January 2026
