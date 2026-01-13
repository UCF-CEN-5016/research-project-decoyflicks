#!/bin/bash
cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║              RepGen - ICSE'26 Paper Replication Artifact                  ║
║                    Complete, Organized Package                            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝


🚀 QUICK START (Copy & Paste)
═════════════════════════════════════════════════════════════════════════════

  export OPENAI_API_KEY="sk-..." && bash scripts/replicate.sh


📁 ORGANIZED DIRECTORY STRUCTURE
═════════════════════════════════════════════════════════════════════════════

  ICSE26-RepGen/
  │
  ├── 📚 DOCUMENTATION (Start Here)
  │   ├── START_HERE.md                 ⭐ Entry point (2 min)
  │   ├── REPLICATE_ME.md               Easy guide (10 min)
  │   ├── REPLICATION_GUIDE.md          Complete guide (30+ min)
  │   ├── ARTIFACT_README.md            Full overview
  │   ├── DELIVERY_SUMMARY.md           Summary of what's included
  │   ├── INDEX.md                      Quick reference
  │   ├── README.md                     Original paper info
  │   └── requirements.txt              Python dependencies
  │
  ├── 📜 SCRIPTS (Executable)
  │   ├── scripts/
  │   │   ├── replicate.sh              ⭐ Main replication pipeline
  │   │   ├── quick_start.sh            Quick test wrapper
  │   │   ├── script.sh                 SLURM batch generator
  │   │   └── baseline_script.sh        Baseline runner
  │   │
  │   └── Usage: bash scripts/replicate.sh [options]
  │
  ├── 🛠️  TOOLS & SOURCE CODE
  │   ├── src/
  │   │   ├── tool_openai.py            ⭐ Main engine (GPT-4.1)
  │   │   ├── tool.py                   Main engine (Ollama/local)
  │   │   ├── run_ablations.py          Ablation studies
  │   │   ├── baselines.py              Baseline methods
  │   │   └── dataset_creation.py       Dataset creator
  │   │
  │   └── Usage: python src/tool_openai.py --bug_id="001"
  │
  ├── 📦 RETRIEVAL MODULE
  │   ├── retrieval/
  │   │   ├── __init__.py
  │   │   ├── config.py                 Configuration
  │   │   ├── pipeline.py               Main pipeline
  │   │   ├── core/                     Core functionality
  │   │   │   ├── code_indexer.py
  │   │   │   ├── dependency_analyzer.py
  │   │   │   ├── module_analyzer.py
  │   │   │   ├── training_code_detector.py
  │   │   │   └── utils.py
  │   │   └── models/                   Model implementations
  │   │       └── hybrid_search.py
  │   │
  │   └── Auto-imported by main tools
  │
  ├── 📊 DATA & RESULTS
  │   ├── dataset/
  │   │   ├── Dataset.csv               Metadata (106 bugs)
  │   │   ├── 001/ ... 106/             Bug directories
  │   │   │   ├── bug_report/           Original bug report
  │   │   │   ├── code/                 Project source code
  │   │   │   ├── context/              Retrieved context
  │   │   │   ├── plan/                 Generation plan
  │   │   │   ├── refined_bug_report/   Processed report
  │   │   │   ├── reproduction_code/    ✨ Generated output
  │   │   │   └── ablations/            Ablation outputs
  │   │   └── ...
  │   │
  │   ├── results/                      Auto-generated results
  │   │   ├── run_YYYYMMDD_HHMMSS/     Results from each run
  │   │   │   ├── summary.txt           Success metrics
  │   │   │   ├── bug_001.log           Log per bug
  │   │   │   └── ...
  │   │   ├── ControlGroup.csv
  │   │   ├── ExperimentalGroup.csv
  │   │   └── Statistical_Tests_ICSE26.ipynb
  │   │
  │   └── figures/                      Paper figures
  │       └── parameter-tuning/
  │
  └── 🎯 THIS FILE
      └── STRUCTURE.md (this file)


🎯 WHAT TO DO NEXT
═════════════════════════════════════════════════════════════════════════════

  1️⃣  READ THE DOCS
      cat START_HERE.md                 (2 minutes)

  2️⃣  SET API KEY
      export OPENAI_API_KEY="sk-..."

  3️⃣  TEST FIRST (Recommended)
      bash scripts/quick_start.sh 1 5 5

  4️⃣  RUN FULL REPLICATION
      bash scripts/quick_start.sh 1 106 5
      OR
      bash scripts/replicate.sh


📚 DOCUMENTATION GUIDE (Pick Your Level)
═════════════════════════════════════════════════════════════════════════════

  ⏱️  2 MINUTES
  ├─ START_HERE.md
  │  Quick overview, commands, common issues
  │
  ⏱️  10 MINUTES
  ├─ REPLICATE_ME.md
  │  Step-by-step guide with examples
  │
  ⏱️  30+ MINUTES
  ├─ REPLICATION_GUIDE.md
  │  Complete guide with full details
  │
  📚 REFERENCE
  ├─ ARTIFACT_README.md  (Full artifact description)
  ├─ DELIVERY_SUMMARY.md (What was created)
  ├─ INDEX.md            (Quick reference)
  └─ README.md           (Original paper info)


🚀 COMMAND REFERENCE
═════════════════════════════════════════════════════════════════════════════

  FASTEST (1 command):
  ────────────────────
  export OPENAI_API_KEY="sk-..." && bash scripts/replicate.sh

  SAFEST (Test first):
  ────────────────────
  export OPENAI_API_KEY="sk-..."
  bash scripts/quick_start.sh 1 5 5       # Test (5 min)
  bash scripts/quick_start.sh 1 106 5     # Full (2-4 hours)

  CUSTOM RANGE:
  ─────────────
  bash scripts/replicate.sh --bug-start 1 --bug-end 50

  CUSTOM EVERYTHING:
  ──────────────────
  bash scripts/replicate.sh \
    --bug-start 1 \
    --bug-end 106 \
    --max-attempts 10

  HELP:
  ────
  bash scripts/replicate.sh --help


🔧 RUNNING INDIVIDUAL TOOLS
═════════════════════════════════════════════════════════════════════════════

  Main Reproduction Tool (OpenAI):
  ────────────────────────────────
  python src/tool_openai.py --bug_id="001" --max-attempts=5

  Main Reproduction Tool (Local):
  ───────────────────────────────
  python src/tool.py --bug_id="001" --max-attempts=5

  Ablation Studies:
  ────────────────
  python src/run_ablations.py --start_bug_id 1 --end_bug_id 106

  Baseline Methods:
  ────────────────
  python src/baselines.py --bug_id "001" --model "qwen2.5-7b"

  Create Dataset:
  ───────────────
  python src/dataset_creation.py --start-id 1 --csv-file issues.csv


📊 EXPECTED RESULTS
═════════════════════════════════════════════════════════════════════════════

  Success Rate:       ~80.19% (85/106 bugs)
  Time per Bug:       1-2 minutes
  Total Runtime:      2-4 hours
  Generated Scripts:  85+
  Detailed Logs:      106+
  Summary Report:     Auto-generated


🔗 REQUIREMENTS & SETUP
═════════════════════════════════════════════════════════════════════════════

  Required:
  ────────
  • Python 3.12
  • OpenAI API key (from platform.openai.com/api-keys)
  • Internet connection

  Optional:
  ────────
  • Ollama (for local models, no API key needed)
  • Groq API key (for Llama models)
  • DeepSeek API key (for DeepSeek models)


📋 ALL FILES AT A GLANCE
═════════════════════════════════════════════════════════════════════════════

  DOCUMENTATION FILES (6):
  ├── START_HERE.md            5.3 KB  ⭐ Entry point
  ├── REPLICATE_ME.md          9.3 KB  Easy guide
  ├── REPLICATION_GUIDE.md     8.9 KB  Complete guide
  ├── ARTIFACT_README.md       10  KB  Full overview
  ├── DELIVERY_SUMMARY.md      10  KB  Delivery info
  ├── INDEX.md                 8.3 KB  Quick reference
  └── README.md               (original) Paper info

  SCRIPTS (4):
  ├── scripts/replicate.sh     9.2 KB  ⭐ Main pipeline
  ├── scripts/quick_start.sh   1.6 KB  Quick wrapper
  ├── scripts/script.sh        1.0 KB  SLURM generator
  └── scripts/baseline_script.sh 1.5 KB Baseline runner

  TOOLS (5):
  ├── src/tool_openai.py       871 lines ⭐ Main engine
  ├── src/tool.py              823 lines Main engine (local)
  ├── src/run_ablations.py     162 lines Ablations
  ├── src/baselines.py         1126 lines Baselines
  └── src/dataset_creation.py  117 lines Dataset creator

  OTHER:
  ├── requirements.txt          (dependencies)
  ├── retrieval/                (retrieval module)
  ├── dataset/                  (106 bugs)
  ├── results/                  (generated results)
  └── figures/                  (paper figures)


💡 KEY POINTS
═════════════════════════════════════════════════════════════════════════════

  ✓ Fully Organized - Scripts, tools, and docs separated
  ✓ Fully Documented - 5+ comprehensive guides
  ✓ Fully Automated - One command to replicate everything
  ✓ Production Ready - 600+ lines of robust code
  ✓ Easy to Extend - Clear directory structure

  ✓ Main entry point: scripts/replicate.sh
  ✓ Main documentation: START_HERE.md
  ✓ Main tools: src/tool_openai.py, src/tool.py


🎯 NEXT STEPS (In Order)
═════════════════════════════════════════════════════════════════════════════

  1. Read:  cat START_HERE.md
  2. Setup: export OPENAI_API_KEY="sk-..."
  3. Test:  bash scripts/quick_start.sh 1 5 5
  4. Run:   bash scripts/replicate.sh
  5. Check: cat results/run_*/summary.txt


╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    Ready to Replicate! 🚀                                 ║
║                                                                            ║
║              bash scripts/replicate.sh                                     ║
║         OR                                                                 ║
║              cat START_HERE.md                                             ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

EOF
