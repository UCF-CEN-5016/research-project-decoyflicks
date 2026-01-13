# RepGen - ICSE'26 Paper Replication Guide

## Quick Start

To replicate the paper's results with a single command:

```bash
# Set your OpenAI API key (if using GPT-4.1 models)
export OPENAI_API_KEY="sk-..."

# Run the complete replication script
bash replicate.sh

# Or with custom bug range and attempts
bash replicate.sh --bug-start 1 --bug-end 106 --max-attempts 5
```

## What This Script Does

The `replicate.sh` script is a comprehensive automation tool that:

1. **Environment Setup** - Configures Python 3.12 virtual environment
2. **Dependency Installation** - Installs all required packages from `requirements.txt`
3. **Environment Configuration** - Sets up API keys and CUDA environment
4. **Dataset Verification** - Validates the dataset structure
5. **Runs Experiments** - Executes bug reproduction for all specified bugs
6. **Generates Reports** - Creates summary reports with success metrics

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or WSL on Windows
- **Python**: 3.12 (must be installed)
- **RAM**: 16GB+ recommended (8GB minimum)
- **GPU**: NVIDIA GPU recommended (but not required)

### Installation
1. Python 3.12 - Install from [python.org](https://www.python.org/downloads/)
2. Ensure `pip` is available: `python3 --version`

### API Keys (Optional)

The script supports multiple LLM backends. Configure based on which models you want to use:

#### OpenAI (GPT-4.1)
```bash
export OPENAI_API_KEY="sk-..."
```
Get your API key from: https://platform.openai.com/api-keys

#### Groq (Llama models)
```bash
export GROQ_API_KEY="gsk_..."
```
Get your API key from: https://console.groq.com/

#### DeepSeek
```bash
export DEEPSEEK_API_KEY="sk-..."
```
Get your API key from: https://platform.deepseek.com/

#### Local Models (Ollama)
To run models locally without API keys, install Ollama:
```bash
# Install Ollama from https://ollama.ai
# Then pull required models:
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull llama3-8b
ollama pull deepseek-r1-7b

# Ensure Ollama service is running:
# On macOS: Ollama runs as a background service after installation
# On Linux: ollama serve
```

## Script Usage

### Basic Usage
```bash
bash replicate.sh
```

### Custom Bug Range
Process only specific bugs (e.g., bugs 1-10 for testing):
```bash
bash replicate.sh --bug-start 1 --bug-end 10
```

### Custom Attempts
Increase generation attempts (default: 5):
```bash
bash replicate.sh --max-attempts 10
```

### Skip Environment Setup
If you've already set up the environment:
```bash
bash replicate.sh --skip-setup
```

### Full Example
```bash
export OPENAI_API_KEY="sk-..."
bash replicate.sh --bug-start 1 --bug-end 106 --max-attempts 5
```

### View Help
```bash
bash replicate.sh --help
```

## Output Structure

After running, results are saved in `results/run_YYYYMMDD_HHMMSS/`:

```
results/run_20260113_120000/
├── summary.txt              # Overall results summary
├── bug_001.log              # Log for each bug
├── bug_002.log
├── ...
└── bug_106.log
```

### Summary Report Example
```
RepGen - Experiment Run Summary
==============================
Configuration:
  - Bug range: 1-106
  - Max attempts: 5
  - Total bugs: 106

Results:
  - Successful: 85
  - Failed: 21
  - Success rate: 80.19%
```

## Troubleshooting

### Python Version Error
```bash
# Check your Python version
python3 --version

# If not 3.12, install it or adjust the script
```

### Import Errors
```bash
# Manually install missing packages
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print('torch OK')"
```

### CUDA/GPU Issues
If you encounter GPU errors:
```bash
# Run on CPU instead:
export CUDA_VISIBLE_DEVICES="-1"
bash replicate.sh
```

### API Key Issues

**OpenAI Error: "OPENAI_API_KEY not set"**
```bash
# Verify the key is set
echo $OPENAI_API_KEY

# Set it properly
export OPENAI_API_KEY="sk-your-actual-key"

# Test the API
python3 << 'EOF'
from openai import OpenAI
client = OpenAI()
print("OpenAI connection successful")
EOF
```

**Rate Limiting**
If you hit rate limits, the script automatically retries with exponential backoff. For manual runs, add delays between bug IDs:
```python
import time
time.sleep(60)  # Wait 60 seconds between bugs
```

### Dataset Structure Issues
Verify your dataset has the required structure:
```bash
# Check a specific bug
ls -la dataset/001/
# Should contain: bug_report/, code/, context/, plan/, reproduction_code/

# Check bug report exists
cat dataset/001/bug_report/001.txt
```

## Advanced Configuration

### Modify Script Defaults
Edit `replicate.sh` and change:
```bash
BUG_START=1           # First bug to process
BUG_END=106           # Last bug to process
MAX_ATTEMPTS=5        # Max generation attempts
PYTHON_VERSION="3.12" # Required Python version
```

### Use Different Models
Edit `tool_openai.py` (lines ~50-55) to change the default model:
```python
# Change from:
model="gpt-4.1"
# To:
model="gpt-4-turbo"  # or other OpenAI models
```

### Run Ablations
For ablation studies, modify the script to run different configurations:
```bash
# Edit and run ablations
python3 run_ablations.py --start_bug_id 1 --end_bug_id 106 --max-gen-attempts 5
```

## Experiment Execution

### Main Tool: tool_openai.py
The primary tool that reproduces bugs:
```bash
python3 tool_openai.py \
  --bug_id="001" \
  --max-attempts=5 \
  --retrieval_ablation="full_system" \
  --generation_ablation="all_steps"
```

### Ablation Studies: run_ablations.py
Run systematic ablations:
```bash
python3 run_ablations.py \
  --start_bug_id 1 \
  --end_bug_id 106 \
  --max-gen-attempts 5 \
  --max-run-attempts 3
```

### Baseline Comparisons: baselines.py
Run baseline methods:
```bash
python3 baselines.py \
  --bug_id "001" \
  --model "qwen2.5-7b" \
  --technique "zero_shot" \
  --examples 3
```

## Output Analysis

### View Results for a Specific Bug
```bash
# Check if bug was successfully reproduced
cat dataset/001/reproduction_code/001.py

# View generation logs
cat results/run_20260113_120000/bug_001.log
```

### Aggregate Results
```bash
# Count successful bugs
grep -c "SUCCESS" results/run_20260113_120000/*.log

# Count failures
grep -c "FAILED" results/run_20260113_120000/*.log
```

### Statistical Analysis
```bash
# Run the provided Jupyter notebook
jupyter notebook results/Statistical_Tests_ICSE26.ipynb

# Or analyze with Python
python3 << 'EOF'
import pandas as pd
results = pd.read_csv('results/ExperimentalGroup.csv')
print(f"Mean success: {results['success'].mean():.2%}")
EOF
```

## Paper Figures and Tables

After running experiments, generate paper figures:

1. **Figure 1**: Framework diagram - see `figures/` directory
2. **Table 1-3**: Results tables - see `results/` CSV files
3. **Figure 2-4**: Result plots - generated from CSV data

### Generate Results CSV
```bash
python3 << 'EOF'
import os
import pandas as pd

bugs = []
for i in range(1, 107):
    bug_id = f"{i:03d}"
    reproduction_file = f"dataset/{bug_id}/reproduction_code/{bug_id}.py"
    if os.path.exists(reproduction_file):
        bugs.append({'bug_id': bug_id, 'status': 'success'})
    else:
        bugs.append({'bug_id': bug_id, 'status': 'failed'})

df = pd.DataFrame(bugs)
success_rate = (df['status'] == 'success').sum() / len(df)
print(f"Reproduction Success Rate: {success_rate:.2%}")
print(df.head())
EOF
```

## Performance Optimization

### Parallel Execution
To run multiple bugs in parallel (if using SLURM):
```bash
# Use the provided script generator
bash script.sh 1 106

# This generates parallel SLURM jobs for each bug
```

### Memory Issues
If running out of memory:
```bash
# Process bugs one at a time with fresh processes
for i in {1..106}; do
    python3 tool_openai.py --bug_id=$(printf "%03d" $i)
done
```

### Batch Processing
Process bugs in smaller batches:
```bash
# Batch 1: Bugs 1-30
bash replicate.sh --bug-start 1 --bug-end 30

# Batch 2: Bugs 31-60
bash replicate.sh --bug-start 31 --bug-end 60

# Batch 3: Bugs 61-106
bash replicate.sh --bug-start 61 --bug-end 106
```

## Paper Citation

If you use this replication script or the RepGen tool, please cite:

```bibtex
@inproceedings{RepGen2026,
  title={Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent},
  author={Shah, Mehil and others},
  booktitle={48th ACM/IEEE International Conference on Software Engineering (ICSE'26)},
  year={2026}
}
```

Preprint: https://arxiv.org/abs/2512.14990

## Support and Issues

For issues or questions:

1. Check this guide's troubleshooting section
2. Review log files in `results/run_*/`
3. Check GitHub issues if available
4. Ensure all prerequisites are installed

### Debug Mode
Enable verbose logging:
```bash
# Set Python to verbose mode
PYTHONVERBOSE=2 bash replicate.sh

# Or in Python directly
python3 -v tool_openai.py --bug_id="001"
```

## Contact

For issues specific to this replication guide, please refer to the project documentation and the paper for detailed methodology.

---

**Last Updated**: January 2026
**Paper**: ICSE'26 - "Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent"
