# Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent (ICSE '26)

This repository contains the official artifact for the ICSE 2026 paper **"Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent"**.

It provides the complete implementation of **RepGen**, a novel automated approach for reproducing Deep Learning (DL) bugs, along with a benchmark dataset of 106 real-world bugs and scripts to replicate the experimental results.

## Abstract

Despite their wide adoption in various domains (e.g., healthcare, finance, software engineering), Deep Learning (DL)-based applications suffer from many bugs, failures, and vulnerabilities. Reproducing these bugs is essential for their resolution, but it is extremely challenging due to the inherent nondeterminism of DL models and their tight coupling with hardware and software environments. According to recent studies, only about 3% of DL bugs can be reliably reproduced using manual approaches. To address these challenges, we present RepGen, a novel, automated, and intelligent approach for reproducing deep learning bugs. RepGen constructs a learning-enhanced context from a project, develops a comprehensive plan for bug reproduction, employs an iterative generate-validate-refine mechanism, and thus generates such code using an LLM that reproduces the bug at hand. We evaluate RepGen on 106 real-world deep learning bugs and achieve a reproduction rate of 80.19%, a 19.81% improvement over the state-of-the-art measure. A developer study involving 27 participants shows that RepGen improves the success rate of DL bug reproduction by 23.35%, reduces the time to reproduce by 56.8%, and lowers participants' cognitive load. 

## 1. Artifact Overview

### Directory Structure

```plaintext
repgen/
├── dataset/             # Benchmark: 106 DL bugs (IDs 001-106) with metadata
├── figures/             # Visualization assets for the paper (RQ1, RQ2)
├── results/             # Raw experimental data and statistical analysis notebooks
├── retrieval/           # RAG Module: Context retrieval implementation
├── scripts/             # Automation & Experimentation Scripts
│   ├── quick-start/     # Entry points for fast reproduction (Local/Cloud)
│   ├── pipeline/        # Core pipeline configuration and parameters
│   └── experimental/    # Baselines (Zero-shot, CoT) and ablation studies
├── src/                 # Source Code: Main agent logic, generation, and execution
├── requirements.txt     # Python project dependencies
└── README.md            # Documentation

```

## 2. Environment Setup

This artifact is compatible with **Linux** (Ubuntu 20.04+), **macOS** (Apple Silicon recommended), and **Windows** (via WSL2 or Git Bash).

> **Important:** All scripts in this repository are shell scripts (`.sh` files) designed to run in Unix-based terminals. To run these scripts:
> - **macOS/Linux:** Use the native terminal
> - **Windows:** Use **Git Bash** (included with [Git for Windows](https://git-scm.com/download/win)) or **WSL2** (Windows Subsystem for Linux)
> 
> Git Bash and WSL2 are freely available on all operating systems and provide a Unix-compatible environment for running shell scripts.

### Prerequisites

* **Python:** Version 3.8 or higher.
* **Git:** Required for version control.
* **Ollama (Local Inference):** Required only if running without cloud APIs.
* [Download for macOS/Linux](https://ollama.ai/download)
* [Download for Windows](https://www.google.com/search?q=https://ollama.ai/download/windows)

### Automated Setup (Recommended)

We provide a setup script that automates environment configuration, dependency installation, and dataset preparation:

```bash
bash scripts/setup.sh --bugs 1-10
```

**Setup Script Options:**

| Flag | Description | Example |
| --- | --- | --- |
| `--bugs` | **Required.** Bug IDs to set up (ranges or lists) | `1-10`, `80-82`, `1,5,10` |
| `--skip-code` | Skip cloning code repositories (metadata only) | `--skip-code` |
| `--force-clone` | Re-clone repositories even if they already exist | `--force-clone` |
| `--quiet` | Suppress informational messages | `--quiet` |
| `--log-file` | Write detailed logs to a file | `--log-file setup.log` |

**Examples:**

```bash
# Set up bugs 1-10 (recommended starting point)
bash scripts/setup.sh --bugs 1-10

# Set up specific bugs (80, 81, 82)
bash scripts/setup.sh --bugs 80-82

# Set up all bugs with logging
bash scripts/setup.sh --bugs 1-106 --log-file setup.log

# Set up without cloning code (faster)
bash scripts/setup.sh --bugs 1-10 --skip-code

# Activate the environment after setup
source venv/bin/activate
```

### Installation

**1. Clone the Repository**

```bash
git clone https://github.com/mehilshah/ICSE26-RepGen
cd ICSE26-RepGen
```

**2. Virtual Environment Setup**

We strongly recommend using a virtual environment to manage dependencies.

* **macOS / Linux / Windows (WSL2):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

* **Windows (Git Bash/PowerShell):**
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

**3. Model Configuration**

#### **Mode A: Local Inference (Ollama)**

If using local LLMs, ensure the Ollama service is active and models are pulled:

```bash
# Start the service (in a separate terminal)
ollama serve

# Pull required models
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
```

#### **Mode B: Cloud Inference (APIs)**

If using GPT-4, Llama-3, or DeepSeek, export your API keys. You may add these to a `.env` file or export them directly:

```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="gsk-..."
export DEEPSEEK_API_KEY="sk-..."
```

## 3. Usage & Experiments

We provide scripts to run the tool in different modes. Logs and generated reproduction scripts will be saved in the `results/` directory.

> **Reminder:** All commands below should be executed in a Unix-based terminal:
> - macOS/Linux: Use the native terminal
> - Windows: Use **Git Bash** or **WSL2** (not PowerShell or Command Prompt)

### Quick Start

#### Demo Script (Recommended for First-Time Users)

To experience the complete RepGen pipeline, run the demo script:

```bash
bash scripts/demo.sh
```

This interactive demo will:
1. Set up the environment for the first two bugs
2. Prompt for your OpenAI API key
3. Run local inference (Qwen2.5)
4. Run cloud inference (GPT-4o)
5. Execute baseline code
6. Run ablation studies

**Prerequisites for Demo:**
- Ollama running locally (`ollama serve` in a separate terminal) for local inference
- OpenAI API key (optional, for cloud-based inference)

#### Individual Script Execution

| Mode | Description | Command |
| --- | --- | --- |
| **Local** | Uses Qwen2.5 (requires Ollama) | `bash scripts/quick-start/local.sh [RANGE] [ATTEMPTS]` |
| **Cloud** | Uses GPT-4o (requires API Key) | `bash scripts/quick-start/cloud.sh [RANGE] [ATTEMPTS]` |

**Example:** Run reproduction for bug IDs 80 through 85 with 1 attempt per context:

```bash
bash scripts/quick-start/local.sh 80-85 1
```

### Replication Packages

#### Baseline Experiments

Reproduce the comparison baselines (Zero-shot, Few-shot, Chain-of-Thought) cited in the paper.

```bash
bash scripts/pipeline/baselines.sh --bugs 1-106
```

#### Ablation Studies

Reproduce the component analysis (e.g., RepGen without Retrieval, RepGen without Iterative Refinement).

```bash
bash scripts/pipeline/ablations.sh --bugs 1-106
```

> **Note:** For advanced customization, refer to the documentation in `scripts/README.md`.

## 4. Troubleshooting

* **Error: "Ollama service is not running"**
* Ensure you have run `ollama serve` in a dedicated terminal window before executing the scripts.

* **Error: "Dataset not found"**
* Verify that the `dataset/` directory exists in the root `repgen/` folder and contains subdirectories `001` to `106`.

* **Permission Denied**
* If scripts fail to execute, grant permissions: `chmod +x scripts/**/*.sh`.

## 5. Citation

If you use RepGen or the dataset in your research, please cite our ICSE 2026 paper:

```bibtex
@inproceedings{shah2026repgen,
  author    = {Shah, Mehil B and Rahman, Mohammad Masudur and Khomh, Foutse},
  title     = {Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent},
  booktitle = {Proceedings of the 48th International Conference on Software Engineering (ICSE)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2512.14990}
}
```
