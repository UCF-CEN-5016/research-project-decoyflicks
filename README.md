# Artifact: Imitation Game - Reproducing Deep Learning Bugs

**Paper Title:** Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent

**Venue:** ICSE 2026

**Authors:** Mehil B. Shah, Mohammad Masudur Rahman, Foutse Khomh

## Badges Claimed

We are applying for the following badges:

1. **Available:** The artifact is permanently archived on Zenodo with a DOI.
2. **Functional:** The artifact is documented, consistent, complete, and exercisable.
3. **Reusable:** The artifact includes a highly automated script (`RepGen.sh`) to ensure dependency isolation, comprehensive documentation for all experimental pipelines, and modular scripts that facilitate repurposing the agent for new tasks.

## 1. Purpose

This artifact provides the complete implementation of **RepGen**, an automated approach for reproducing Deep Learning (DL) bugs. It allows researchers to:

1. **Replicate** the experimental results presented in the paper (Table 2 and Table 3).
2. **Reuse** the RepGen agent to attempt reproduction of new DL bugs.
3. **Analyze** the benchmark dataset of 106 real-world DL bugs.

The artifact contains:

* **Source Code:** The implementation of the LLM-based agent, RAG module, and execution sandbox.
* **Dataset:** A curated benchmark of 106 DL bugs with metadata and reproduction criteria.
* **Scripts:** Automated pipelines to reproduce the baseline comparisons (Zero-shot, CoT) and the ablation studies.

## 2. Provenance

* **Preprint:** [arXiv:2512.14990](https://arxiv.org/abs/2512.14990)
* **Archival Repository:** https://zenodo.org/records/18263581
* **GitHub Repository:** [https://github.com/mehilshah/ICSE26-RepGen](https://github.com/mehilshah/ICSE26-RepGen)

## 3. Data

The artifact includes a dataset of **106 real-world Deep Learning bugs** collected from GitHub repositories and Stack Overflow issues.

* **Location:** `/dataset/`
* **Structure:**
* Directories `001` to `106` represent individual bugs.
* Each directory contains `metadata.json` (bug description, faulty code snippet, expected error) and `ground_truth.py` (author-verified reproduction script).


* **Privacy/Ethics:** All data is derived from public open-source repositories (MIT/Apache 2.0 licenses). No personal data or proprietary code is included.

## 4. Setup

To ensure the artifact is easy to install and isolated from your system packages, we provide an automated setup script that creates a Python virtual environment.

### Hardware & OS Requirements

* **Operating System:** Linux (Ubuntu 20.04+), macOS (Apple Silicon supported), or Windows via **WSL2**.
  * *Note:* Git Bash provides a Unix-like shell but still relies on the native Windows toolchain. Some dependencies may fail to build in this environment (see Known Issues).

* **Python:** Version **3.12**.
  RepGen depends on several Python libraries (e.g., **ANNOY**) that include native C++ extensions. While Python 3.12 is the most stable target across platforms, installation on **native Windows environments** may still trigger source builds that require a fully configured Windows SDK. To ensure reproducibility and avoid platform-specific compilation issues, we recommend Linux, macOS, or Windows via **WSL2**.

* **Disk Space:** Approximately 5 GB (dependencies, models, and dataset).

* **GPU (Optional):** An NVIDIA GPU with **16GB+ VRAM** is recommended for local inference. If no compatible GPU is detected, RepGen automatically falls back to CPU or API-based inference.

### Installation Steps

1. **Clone the Repository:**
```bash
git clone https://github.com/mehilshah/ICSE26-RepGen
cd ICSE26-RepGen
```


2. **Run Automated Setup:**
This script initializes the virtual environment, installs requirements, and prepares the dataset structure.
```bash
# This process takes approximately 5-10 minutes
bash scripts/setup.sh --bugs 1-10
```


3. **Activate Environment:**
```bash
# Linux / macOS / WSL2
source venv/bin/activate

# Windows Git Bash
source venv/Scripts/activate
```


4. **Configure LLM Backend:**
* **Option A (Local):** Install [Ollama](https://ollama.ai) and run `ollama serve` in a separate terminal.
* **Option B (Cloud API):** If you prefer using OpenAI or DeepSeek, export your API key:
```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export GROQ_API_KEY="sk-....."
```

## 5. Usage & Replication

### Verify Installation (Functional Check)

To verify the tool is functional, run the demo script on a single bug (Bug ID 1). This is a quick test to ensure the environment is valid.

```bash
bash scripts/quick-start/cloud.sh 1 1

```

**Expected Output:** The script should generate a reproduction plan, execute Python code, and output `SUCCESS: Bug reproduced` or `FAILURE`.

### Replicating Main Results (Paper Table 2)

To reproduce the efficiency of RepGen compared to baselines (RQ1):

```bash
# Run RepGen on the full dataset (106 bugs)
# Note: This may take several hours depending on hardware.
bash scripts/quick-start/local.sh 1-106 1

# Run Baselines (Zero-Shot and CoT) (Needs Ollama, and API Keys)
bash scripts/experimental/baselines.sh --bugs 1-106
```

### Replicating Ablation Studies (Paper Table 3)

To reproduce the component analysis (RQ2):

```bash
bash scripts/experimental/ablations.sh --bugs 1-106
```

## 6. Directory Structure

```plaintext
repgen/
├── dataset/             # 106 DL bugs with metadata
├── results/             # Generated reproduction scripts and logs
├── scripts/             # Automation for experiments
│   ├── setup.sh         # Environment setup and dependency installation
│   ├── quick-start/     # Scripts for running individual bugs
│   └── pipeline/        # Replication scripts for paper results
├── src/                 # RepGen Python source code
├── requirements.txt     # Python dependencies
└── README.md
```

## 7. License

This project is licensed under the **MIT License**. See the `LICENSE` file for details. The dataset is derived from public repositories; original licenses for specific projects are respected.

### 8. Known Issues

On native Windows installations, some dependencies (notably **ANNOY**) include C++ extensions that may be built from source if a compatible prebuilt wheel is unavailable. This requires a fully configured Windows C++ toolchain, including the Windows SDK. Even when using Git Bash, the build still relies on the native Windows compiler and may fail if the SDK is missing or misconfigured.

For reliable installation and reproduction of results, we strongly recommend running RepGen on:
- Linux (Ubuntu 20.04+),
- macOS, or
- Windows via **WSL2 (Ubuntu)**.