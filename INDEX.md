#!/bin/bash

# This file is an executable guide - you can also just read it as text

cat << 'EOF'

╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║         RepGen - ICSE'26 Paper Replication Complete Package             ║
║                                                                            ║
║  "Imitation Game: Reproducing Deep Learning Bugs                         ║
║   Leveraging Intelligent Agent"                                           ║
║                                                                            ║
║  🚀 One-line replication:                                                 ║
║     export OPENAI_API_KEY="sk-..." && bash replicate.sh                  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

📚 DOCUMENTATION (Read in this order):

1. ⭐ START_HERE.md (2 minutes)
   └─ Quick overview and fastest path to replication

2. 📋 REPLICATE_ME.md (10 minutes)
   └─ Easy-to-follow guide with examples

3. 📖 REPLICATION_GUIDE.md (30+ minutes)
   └─ Comprehensive guide with all details

4. 🏗️ ARTIFACT_README.md (reference)
   └─ This file - complete artifact overview

5. README.md (for context)
   └─ Original paper description and background


🚀 QUICK START (Choose one):

   Test first (recommended):
   ─────────────────────────
   $ export OPENAI_API_KEY="sk-..."
   $ bash quick_start.sh 1 5 5

   Run full paper (after testing):
   ───────────────────────────────
   $ export OPENAI_API_KEY="sk-..."
   $ bash replicate.sh

   Or with custom options:
   ──────────────────────
   $ bash replicate.sh --bug-start 1 --bug-end 20 --max-attempts 5


🔧 SCRIPTS INCLUDED:

   replicate.sh (606 lines) ⭐
   ├─ Main replication script
   ├─ Handles environment setup
   ├─ Installs dependencies
   ├─ Configures API keys
   ├─ Verifies dataset
   ├─ Runs experiments
   └─ Generates reports
   
   quick_start.sh (40 lines)
   ├─ Simplified wrapper around replicate.sh
   ├─ Perfect for first-time users
   └─ Usage: bash quick_start.sh START END ATTEMPTS

   tool_openai.py (871 lines) ⭐
   ├─ Main reproduction engine
   ├─ Uses OpenAI API (GPT-4.1)
   └─ Usage: python tool_openai.py --bug_id="001"

   tool.py (823 lines)
   ├─ Local version using Ollama
   ├─ No API key needed
   └─ Usage: python tool.py --bug_id="001"

   run_ablations.py (162 lines) ⭐
   ├─ Systematic ablation runner
   └─ Usage: python run_ablations.py --start_bug_id 1 --end_bug_id 106

   baselines.py (1126 lines)
   ├─ Baseline comparison methods
   └─ Usage: python baselines.py --bug_id "001" --model "qwen2.5-7b"


📊 WHAT GETS GENERATED:

   For each of the 106 bugs:
   ├─ Reproduction code (Python scripts)
   ├─ Detailed execution logs
   ├─ Retrieved context
   ├─ Generation plans
   └─ Performance metrics

   Aggregated results:
   ├─ Summary report (success rates, statistics)
   ├─ Log files for each bug
   ├─ CSV results (for statistical analysis)
   └─ Expected: ~80% success rate (85/106 bugs)


✅ PREREQUISITES:

   [ ] Python 3.12 (install from https://www.python.org/downloads/)
   [ ] OpenAI API key (from https://platform.openai.com/api-keys)
   [ ] ~5-10 GB disk space
   [ ] 8+ GB RAM (16GB+ recommended)
   [ ] Internet connection


🎯 TYPICAL WORKFLOW:

   1. Read START_HERE.md (2 min)
   2. Install Python 3.12 if needed (1 min)
   3. Get OpenAI API key (2 min)
   4. Test with: bash quick_start.sh 1 5 5 (5 min)
   5. Run full: bash replicate.sh (2-4 hours)
   6. Check results: cat results/run_*/summary.txt (1 min)
   
   Total setup time: ~10 minutes
   Total experiment time: ~2-4 hours


📁 KEY FILES:

   ├── START_HERE.md ⭐ (READ THIS FIRST!)
   │   └─ 2-minute overview
   │
   ├── REPLICATE_ME.md
   │   └─ Easy guide with examples
   │
   ├── REPLICATION_GUIDE.md
   │   └─ Full detailed guide
   │
   ├── replicate.sh ⭐ (RUN THIS!)
   │   └─ Main replication script
   │
   ├── quick_start.sh
   │   └─ Quick wrapper
   │
   ├── tool_openai.py ⭐ (Main engine)
   │   └─ Reproduction tool
   │
   ├── requirements.txt
   │   └─ Python dependencies
   │
   └── dataset/
       ├── 001/ ... 106/
       │   ├── bug_report/
       │   ├── code/
       │   └── reproduction_code/ ⭐ (Generated here)
       └── Dataset.csv


🌟 HIGHLIGHTS:

   ✨ Fully automated - One command replicates the entire paper
   ✨ Well documented - 4 comprehensive guides included
   ✨ Flexible - Works with OpenAI, Groq, DeepSeek, or local models
   ✨ Production-ready - 606-line robust replication script
   ✨ Scalable - Handles all 106 bugs
   ✨ Monitored - Automatic logging and result aggregation
   ✨ Reproducible - Exact same workflow as paper


💡 TIPS:

   • Start with: bash quick_start.sh 1 5 5 (test with 5 bugs)
   • For full: bash quick_start.sh 1 106 5 (all 106 bugs)
   • Check setup: python3 --version (must be 3.12.x)
   • Verify API key: echo $OPENAI_API_KEY (should show sk-...)
   • View results: cat dataset/001/reproduction_code/001.py
   • Check logs: tail -f results/run_*/bug_001.log


❓ COMMON QUESTIONS:

   Q: What if I don't have an API key?
   A: Use local models with Ollama (free, no API needed)
      See REPLICATION_GUIDE.md for setup

   Q: How long does it take?
   A: Setup: ~10 minutes | Full replication: ~2-4 hours

   Q: Will it work on my computer?
   A: If you have Python 3.12 and internet, yes!
      Tested on macOS, Linux, WSL

   Q: What if something goes wrong?
   A: Check REPLICATION_GUIDE.md → Troubleshooting
      (Most issues are solved in that section)

   Q: Can I run on GPU?
   A: Yes, automatically uses GPU if available
      Falls back to CPU if GPU not available


🚀 ONE-LINER REPLICATION:

   export OPENAI_API_KEY="sk-..." && bash replicate.sh


📞 NEED HELP?

   1. Quick question? → START_HERE.md
   2. How to...? → REPLICATE_ME.md  
   3. Troubleshooting? → REPLICATION_GUIDE.md
   4. Background? → README.md


🎓 PAPER INFORMATION:

   Title: Imitation Game: Reproducing Deep Learning Bugs
          Leveraging Intelligent Agent
   
   Venue: 48th ACM/IEEE International Conference on
          Software Engineering (ICSE'26)
   
   Preprint: https://arxiv.org/abs/2512.14990

   Citation:
   @inproceedings{RepGen2026,
     title={Imitation Game: Reproducing Deep Learning Bugs...},
     author={Shah, Mehil and others},
     booktitle={ICSE'26},
     year={2026}
   }


📈 EXPECTED RESULTS:

   Success Rate: ~80.19% (85/106 bugs)
   Time per Bug: 1-2 minutes
   Total Time: ~2-4 hours
   Generated Files: 85+ Python scripts
   Logs Generated: 106+ detailed logs


🎯 NEXT STEPS:

   1. Read: START_HERE.md
   2. Run: bash quick_start.sh 1 5 5
   3. Test: bash quick_start.sh 1 106 5
   4. Analyze: cat results/run_*/summary.txt


═══════════════════════════════════════════════════════════════════════════════

                    Ready? Let's replicate! 🚀

                 bash quick_start.sh 1 5 5    (test first)
                 bash quick_start.sh 1 106 5  (full paper)

                     Questions? → START_HERE.md

═══════════════════════════════════════════════════════════════════════════════

EOF

# If run as executable, show path to START_HERE.md
echo ""
echo "Next: Read $(pwd)/START_HERE.md"
echo ""
