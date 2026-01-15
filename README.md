# Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent (ICSE'26)

This repository contains the artifact for the ICSE 2026 paper **"Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent"**. It includes the complete source code, the dataset of 106 real-world DL bugs, and automated scripts to replicate the paper's experiments.

This work has been accepted for publication in the 48th ACM/IEEE International Conference on Software Engineering. The preprint can be found at https://arxiv.org/abs/2512.14990. 

Abstract: Despite their wide adoption in various domains (e.g., healthcare, finance, software engineering), Deep Learning (DL)-based applications suffer from many bugs, failures, and vulnerabilities. Reproducing these bugs is essential for their resolution, but it is extremely challenging due to the inherent nondeterminism of DL models and their tight coupling with hardware and software environments. According to recent studies, only about 3% of DL bugs can be reliably reproduced using manual approaches. To address these challenges, we present RepGen, a novel, automated, and intelligent approach for reproducing deep learning bugs. RepGen constructs a learning-enhanced context from a project, develops a comprehensive plan for bug reproduction, employs an iterative generate-validate-refine mechanism, and thus generates such code using an LLM that reproduces the bug at hand. We evaluate RepGen on 106 real-world deep learning bugs and achieve a reproduction rate of 80.19%, a 19.81% improvement over the state-of-the-art measure. A developer study involving 27 participants shows that RepGen improves the success rate of DL bug reproduction by 23.35%, reduces the time to reproduce by 56.8%, and lowers participants' cognitive load. 

Preprint: ICSE26_RepGen.pdf (in this repository), https://arxiv.org/abs/2512.14990

## 1. Artifact Overview

### Directory Structure

```plaintext
repgen/
├── dataset/             # Benchmark of 106 DL bugs (001-106) and metadata
├── figures/             # Figures and plots used in the paper (RQ1, RQ2)
├── results/             # Raw experimental data and statistical test notebooks
├── retrieval/           # Core retrieval module (RAG implementation)
├── scripts/             # Automation scripts for reproduction
│   ├── quick-start/     # Simple scripts for quickly running the artifact
│   └── pipeline/        # Core scripts that can be used to customize the parameters
│   └── experimental/    # Full pipelines for baselines and ablation studies
├── src/                 # Main tool source code (generation & execution)
├── requirements.txt     # Python dependencies
└── README.md
```

## 2. Environment Setup

This artifact supports **Linux** (Ubuntu 20.04+), **macOS** (Apple Silicon recommended), and **Windows** (via WSL2 or Git Bash).

### Prerequisites

* **Python:** Version 3.8 or higher.
* **Git:** Required to clone the repository.
* **Ollama (Local Mode Only):** Required if running inference locally without cloud APIs.
* **macOS/Linux:** [Download](https://ollama.ai/download).
* **Windows:** Install [Ollama for Windows](https://www.google.com/search?q=https://ollama.ai/download/windows).

### Installation Steps

**1. Clone the Repository**

```bash
git clone https://github.com/mehilshah/ICSE26-RepGen
cd ICSE26-RepGen

```

**2. Install Dependencies**
We recommend using a virtual environment.

* **macOS / Linux / Windows (WSL2):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


* **Windows (Git Bash):**
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

**3. Model Setup (Local Mode Only)**
If running locally, start the Ollama service and pull the required models:

```bash
# In a separate terminal window:
ollama serve

# Pull the models:
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b

```

**4. API Configuration (Cloud Mode Only)**
If running experiments with GPT-4, Llama-3, or DeepSeek, export your API keys:

```bash
export OPENAI_API_KEY="your_key_here"
export GROQ_API_KEY="your_key_here"
export DEEPSEEK_API_KEY="your_key_here"
```

## 3. How to Run

### Option A: Quick Start (Local Inference)

Run the pipeline using Qwen2.5 and Qwen2.5-Coder.

* **Command:** `bash scripts/quick-start/local.sh [BUG_RANGE] [ATTEMPTS]`
* **Example:**
```bash
bash scripts/quick-start/local.sh 80-80 1
```

### Option B: Quick Start (Cloud Inference)

Run the pipeline using GPT-4.1.

* **Command:** `bash scripts/quick-start/cloud.sh [BUG_RANGE] [ATTEMPTS]`
* **Example:**
```bash
bash scripts/quick-start/cloud.sh 1-3 5
```

### Option C: Baseline Experiments

Reproduce the baseline comparisons (Zero-shot, Few-shot, CoT).

* **Command:** `bash scripts/pipeline/baselines.sh --bugs [RANGE]`
* **Example:**
```bash
bash scripts/pipeline/baselines.sh --bugs 1-2
```

### Option D: Ablation Studies

Reproduce the ablation study results.

* **Command:** `bash scripts/pipeline/ablation.sh --bugs [RANGE]`
* **Example:**
```bash
bash scripts/pipeline/ablation.sh --bugs 1-2

```
For more run commands, and possible customizations, please refer to the README file in the /scripts directory.

## 5. Troubleshooting

* **"Ollama service is not running"**: Run `ollama serve` in a new terminal.
* **"Dataset not found"**: Ensure the `dataset` folder is in the root `repgen/` directory and contains numbered folders (001-106).
* **Permission Denied**: Run `chmod +x scripts/**/*.sh`.