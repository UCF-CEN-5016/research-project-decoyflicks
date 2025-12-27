## Imitation Game: Reproducing Deep Learning Bugs Leveraging Intelligent Agent (ICSE'26)

This work has been accepted for publication in the 48th ACM/IEEE International Conference on Software Engineering. The preprint can be found at https://arxiv.org/abs/2512.14990.

### Abstract
Despite their wide adoption in various domains (e.g., healthcare, finance, software engineering), Deep Learning (DL)-based applications suffer from many bugs, failures, and vulnerabilities. Reproducing these bugs is essential for their resolution, but it is extremely challenging due to the inherent nondeterminism of DL models and their tight coupling with hardware and software environments. According to recent studies, only about 3% of DL bugs can be reliably reproduced using manual approaches. To address these challenges, we present RepGen, a novel, automated, and intelligent approach for reproducing deep learning bugs. RepGen constructs a learning-enhanced context from a project, develops a comprehensive plan for bug reproduction, employs an iterative generate-validate-refine mechanism, and thus generates such code using an LLM that reproduces the bug at hand. We evaluate RepGen on 106 real-world deep learning bugs and achieve a reproduction rate of 80.19%, a 19.81% improvement over the state-of-the-art measure. A developer study involving 27 participants shows that RepGen improves the success rate of DL bug reproduction by 23.35%, reduces the time to reproduce by 56.8%, and also lowers the cognitive load of the participants.

# Folder Structure
Main Project Files:
-   `tool.py`: RepGen's base implementation (uses local Ollama models)
-   `tool_openai.py`: GPT-based variant of RepGen (uses OpenAI API)
-   `run_ablations.py`: Script for orchestrating ablation studies
-   `script.sh`: SLURM batch script generator for `run_ablations.py`
-   `baseline_script.sh`: Script for running baseline comparisons
-   `dataset_creation.py`: Script for creating the dataset structure
-   `requirements.txt`: Project dependencies

Dataset Structure (`/dataset/[bug_id]/`):

-   `bug_report/`: Contains original bug reports
-   `context/`: Stores relevant context information
-   `plan/`: Generated reproduction plans
-   `refined_bug_report/`: Processed bug reports
-   `reproduction_code/`: Generated reproduction code
-   `Dataset.csv`: Main dataset metadata

Retrieval Module (`/retrieval/`):
-   `core/`: Core functionality
    -   `code_indexer.py`: Code indexing implementation
    -   `dependency_analyzer.py`: Dependency analysis
    -   `module_analyzer.py`: Module analysis
    -   `training_code_detector.py`: Training code detection
    -   `utils.py`: Utility functions
-   `models/`: Model implementations
    -   `hybrid_search.py`: Hybrid search implementation
-   `config.py`: Configuration settings
-   `pipeline.py`: Main retrieval pipeline

Results and Figures:
-   `/figures/`: Visualization and diagrams
    -   `parameter-tuning/`: Parameter tuning results
    -   Various result plots and framework diagrams
-   `/results/`: Experimental results
    -   `Dev Study Results/`: Developer study data
    -   Statistical significance tests, results for RQ1, RQ2, and RQ3.

# Dependencies
The dependencies can be installed using,
```bash
pip3 install -r requirements.txt
````

## Running and Analysis

To run these scripts, you need to set up your environment, including installing Python dependencies and configuring API keys if you plan to use external models like OpenAI, Llama, or DeepSeek, or setting up Ollama for local models.

### Prerequisites

  - Python 3.12
  - Ollama (if using local models like `qwen2.5:7b`, `qwen2.5-coder:7b`, `llama3-8b`, `qwen3-8b`, `deepseek-r1-7b`)
  - API keys for Llama, DeepSeek, or OpenAI (if using their respective models)

### Environment Variables

Set the following environment variables for API access:

  - `GROQ_API_KEY` (for Llama models)
  - `DEEPSEEK_API_KEY` (for DeepSeek models)
  - `OPENAI_API_KEY` (for OpenAI models like `gpt-4.1`)

### Ollama Setup

If using local models, ensure Ollama is installed and running. Download the required models:

```
ollama pull qwen2.5:7b
ollama pull qwen2.5-coder:7b
ollama pull llama3-8b
ollama pull qwen3-8b
ollama pull deepseek-r1-7b
```

## Scripts

### `dataset_creation.py`

This script is used to create a dataset of bug reports by fetching information from GitHub issues and cloning repositories.

**Usage:**

```
python dataset_creation.py [OPTIONS]
```

**Arguments:**

  - `--start-id`: Starting bug ID
  - `--start-row`: Starting row in CSV
  - `--end-row`: Ending row in CSV (default: `None`, process all rows)
  - `--csv-file`: CSV file with issues data (default: `issues.csv`)
  - `--dataset-dir`: Base directory for the dataset (default: `dataset`)

**Example:**

```
python dataset_creation.py --csv-file my_issues.csv --start-row 10 --end-row 50
```

### `baselines.py`

This Python script generates bug reproduction code using various language models and prompting techniques (zero-shot, few-shot, Chain-of-Thought).

**Usage:**

```
python baselines.py --bug_id <BUG_ID> --model <MODEL_NAME> --technique <TECHNIQUE> [OPTIONS]
```

**Arguments:**

  - `--bug_id` (required): Bug ID to reproduce (e.g., `001`)
  - `--model`: Model to use. Options include `llama3-8b`, `qwen3-8b`, `deepseek-r1-7b`, `llama3-70b`, `gpt-4.1`, `qwen2.5-7b`, `deepseek-r1-685b`, `qwen2.5-coder`. Default is `llama3-8b`.
  - `--technique`: Prompting technique. Options include `zero_shot`, `few_shot`, `cot`. Default is `zero_shot`.
  - `--examples`: Number of examples for few-shot learning (default: `3`)

**Supported Models & APIs:**

  - **Local Ollama Models:** `llama3-8b`, `qwen3-8b`, `deepseek-r1-7b`, `qwen2.5-7b`, `qwen2.5-coder`
  - **DeepSeek API:** Models starting with `deepseek`
  - **Llama API:** Models starting with `llama3.3-70b`
  - **OpenAI API:** Models starting with `gpt-4.1`

**Example:**

```
python baselines.py --bug_id 001 --model gpt-4.1 --technique cot
python baselines.py --bug_id 002 --model qwen2.5-7b --technique few_shot --examples 5
```

### `baseline_script.sh`

This bash script automates the execution of `baselines.py` over a range of bug IDs, models, and techniques.

**Usage:**

```
./baseline_script.sh <start_bug_id> <end_bug_id>
```

**Arguments:**

  - `start_bug_id`: The starting bug ID (integer)
  - `end_bug_id`: The ending bug ID (integer)

**Example:**

```
./baseline_script.sh 1 10
```

### `tool.py`

This Python script performs code generation for bug reproduction using RepGen's full pipeline. It utilizes **local Ollama models** (specifically `qwen2.5:7b` and `qwen2.5-coder:7b`) for its operations. It supports ablation studies by allowing you to disable specific pipeline components.

**Usage:**

```
python tool.py --bug_id <BUG_ID> [OPTIONS]
```

**Arguments:**

  - `--bug_id` (required): Bug ID to analyze (e.g., `001`)
  - `--retrieval_ablation`: Name of retrieval config.
      - **Choices:** `full_system`, `NO_BM25`, `NO_ANN`, `NO_RERANKER`, `NO_TRAINING_LOOP_EXTRACTION`, `NO_TRAINING_LOOP_RANKING`, `NO_MODULE_PARTITIONING`, `NO_DEPENDENCY_EXTRACTION`
  - `--generation_ablation`: Name of generation config.
      - **Choices:** `all_steps`, `no_refine`, `no_plan`, `no_compilation`, `no_relevance`, `no_static_analysis`, `no_runtime_feedback`
  - `--max-attempts`: Maximum attempts for code generation (default: `5`)

**Example:**

```
# Run full pipeline on bug 003
python tool.py --bug_id 003

# Run on bug 004, disabling the Reranker and the Plan Generation steps
python tool.py --bug_id 004 --retrieval_ablation NO_RERANKER --generation_ablation no_plan
```

### `tool_openai.py`

This script is similar to `tool.py` but uses **OpenAI's GPT models** (e.g., `gpt-4.1`) for bug report refinement, plan generation, and code generation. It requires an `OPENAI_API_KEY`.

**Usage:**

```
python tool_openai.py --bug_id <BUG_ID> [OPTIONS]
```

**Arguments:**

  - `--bug_id` (required): Bug ID to analyze (e.g., `001`)
  - `--retrieval_ablation`: Name of retrieval config. (See `tool.py` for choices)
  - `--generation_ablation`: Name of generation config. (See `tool.py` for choices)
  - `--max-attempts`: Maximum attempts for code generation (default: `5`)

**Example:**

```
python tool_openai.py --bug_id 005
```

### `run_ablations.py`

This Python script systematically orchestrates ablation studies by repeatedly calling `tool_openai.py` over a range of bug IDs, each time with a different ablation setting.

**Usage:**

```
python run_ablations.py --start_bug_id <ID> --end_bug_id <ID> [OPTIONS]
```

**Arguments:**

  - `--start_bug_id` (required): The first bug ID to process (integer)
  - `--end_bug_id` (required): The last bug ID to process (integer)
  - `--max-gen-attempts`: Max attempts passed to `tool_openai.py` (default: `5`)
  - `--max-run-attempts`: Max retry attempts for each script execution (default: `3`)

**Example:**

```
# Run all retrieval ablations for bugs 1 through 10
python run_ablations.py --start_bug_id 1 --end_bug_id 10
```

### `script.sh`

This bash script generates 106 separate SLURM batch scripts (e.g., `run_bug_001.sh`, `run_bug_002.sh`, etc.) and submits them to the SLURM scheduler. Each generated script sets up its own environment and runs `run_ablations.py` for a single, specific bug ID.

**Usage:**

```
./script.sh
```

*(Note: This script has no arguments; it is hard-coded to run from bug ID 1 to 106).*

### Disclaimer
Some code snippets in this replication package contain hard-coded file paths, as the code was generated using the actual bug reports and code snippets. These paths are preserved for authenticity but are not associated with the authors of this study. We have carefully reviewed all code and the artifacts to ensure no identifying information has been included.