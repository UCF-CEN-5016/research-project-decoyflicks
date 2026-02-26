# Imitation Game: Reproducing Deep Learning Bugs Leveraging an Intelligent Agent

**Authors**: Mehil B. Shah, Mohammad Masudur Rahman, Foutse Khomh

**ACM Artifact Evaluation Badges:** Available, Functional, Reusable

<p align="center">
  <img src="figures/PaperHeader.png" alt="ICSE 2026 paper header with ACM artifact badges" width="100%">
</p>

## Abstract

Despite their wide adoption in various domains (e.g., healthcare, finance, and software engineering), deep learning (DL)–based applications suffer from numerous bugs, failures, and vulnerabilities. Reproducing these bugs is essential for their resolution, but it is extremely challenging due to the inherent nondeterminism of DL models and their tight coupling with hardware and software environments. Prior studies report that only about 3% of DL bugs can be reliably reproduced using manual approaches. To address these challenges, we present **RepGen**, a novel automated and intelligent approach for reproducing deep learning bugs. RepGen constructs a learning-enhanced context from a project, develops a comprehensive reproduction plan, and employs an iterative generate–validate–refine mechanism in which a large language model produces executable code that reproduces the target bug. We evaluate RepGen on 106 real-world deep learning bugs and achieve a reproduction rate of 80.19%, representing a 19.81% improvement over the state of the art. A controlled developer study with 27 participants shows that RepGen improves reproduction success by 23.35%, reduces time-to-reproduce by 56.8%, and lowers participants’ cognitive load.

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

* **Python:** Version **3.8+** (Recommended: **3.10+**)
  RepGen depends on several Python libraries (e.g., **ANNOY**) that include native C++ extensions. While Python 3.12 is supported, we recommend Python 3.10 for better compatibility with some older deep learning libraries. Installation on **native Windows environments** may still trigger source builds that require a fully configured Windows SDK. To ensure reproducibility and avoid platform-specific compilation issues, we recommend Linux, macOS, or Windows via **WSL2**.

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
# This process takes approximately 5-10 minutes for a small number of bugs.
# WARNING: Running with a large range of bugs (e.g., 1-106) will clone multiple repositories
# and may take significant time and bandwidth. Plan accordingly.
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
* **Option A (Local - Recommended):** Install [Ollama](https://ollama.ai) and run `ollama serve` in a separate terminal.
  
  **Ollama Setup Details:**
  1. **Install Ollama:** Follow instructions at [ollama.ai](https://ollama.ai).
  2. **Pull and Serve Models:** RepGen uses `qwen2.5:7b` and `qwen2.5-coder:7b` by default. Run:
     ```bash
     ollama pull qwen2.5:7b
     ollama pull qwen2.5-coder:7b
     ollama serve
     ```
  3. **Configure:** No additional configuration is needed if running on default port 11434.

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
bash scripts/experimental/baseline.sh --bugs 1-106
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
├── retrieval/           # Retrieval Augmented Generation (RAG) module
│   ├── core/            # Core analysis modules (indexing, dependency analysis)
│   ├── models/          # Embedding models and search logic
│   └── pipeline.py      # Main retrieval pipeline implementation
├── scripts/             # Automation for experiments
│   ├── setup.sh         # Environment setup and dependency installation
│   ├── quick-start/     # Scripts for running individual bugs
│   └── pipeline/        # Replication scripts for paper results
├── src/                 # RepGen Python source code
├── requirements.txt     # Python dependencies
└── README.md
```

## 7. Reusing RepGen for New Bugs

RepGen is designed to be reusable for reproducing bugs in other deep learning projects. To use RepGen on your own dataset:

### 1. Preparing the Data
Create a directory structure for your new bug (e.g., `my_dataset/new_bug_id/`) with the following structure:
```plaintext
my_dataset/
└── new_bug_id/
    ├── bug_report/
    │   └── new_bug_id.txt      # Text file containing the bug report/issue description
    └── code/
        └── ...                 # The source code of the project to analyze
```

### 2. Running RepGen
Use the `--ae_dataset_path` flag to point to your custom dataset root.

**Local (Ollama):**
```bash
python src/tool.py --bug_id new_bug_id --ae_dataset_path /path/to/my_dataset
```

**Remote (OpenAI/DeepSeek):**
```bash
python src/tool_openai.py --bug_id new_bug_id --ae_dataset_path /path/to/my_dataset
```

## 8. License

This project is licensed under the **MIT License**. See the `LICENSE` file for details. The dataset is derived from public repositories; original licenses for specific projects are respected.

## 9. Customizing RepGen

RepGen is modular, allowing you to customize key components:

### 1. Configuration (`retrieval/config.py`)
Modify this file to change:
- **Embedding Models**: `EMBEDDING_MODEL` (Default: `flax-sentence-embeddings/st-codesearch-distilroberta-base`)
- **Reranker Models**: `RERANKER_MODEL` (Default: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`)
- **Search Parameters**: `SEARCH_TOP_K`, `RERANK_TOP_K`, `ALPHA` (Weight for hybrid search).

### 2. Prompts (`src/tool.py` & `src/tool_openai.py`)
The logic for prompt generation is contained within the main script files. You can modify these functions to change the agent's behavior:
- `create_prompt_refinement`: Modifies how the bug report is summarized/refined.
- `create_prompt_plan`: Changes how the reproduction plan is generated.
- `_build_prompt` (in main loop): Controls the code generation prompt structure.
- `check_relevance`: Adjusts the criteria for relevance checking.
- `calculate_probability_of_reproduction`: Modifies the validaton logic and feedback generation.

### 3. Retrieval Logic (`retrieval/models/hybrid_search.py`)
Modify the `search` method in `HybridSearchIndex` to change how BM25 and Semantic Search scores are combined or to implement a different retrieval algorithm.

## 10. Known Issues

On native Windows installations, some dependencies (notably **ANNOY**) include C++ extensions that may be built from source if a compatible prebuilt wheel is unavailable. This requires a fully configured Windows C++ toolchain, including the Windows SDK. Even when using Git Bash, the build still relies on the native Windows compiler and may fail if the SDK is missing or misconfigured.

For reliable installation and reproduction of results, we strongly recommend running RepGen on:
- Linux (Ubuntu 20.04+),
- macOS, or
- Windows via **WSL2 (Ubuntu)**.
