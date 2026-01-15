import subprocess
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

# --- 1. Define Retrieval Ablations ---
# These map to the --retrieval_ablation flag in main.py
RETRIEVAL_ABLATIONS = [
    # "full_system",
    "NO_BM25",
    "NO_ANN",
    "NO_RERANKER",
    "NO_TRAINING_LOOP_EXTRACTION",
    "NO_TRAINING_LOOP_RANKING",
    "NO_MODULE_PARTITIONING",
    "NO_DEPENDENCY_EXTRACTION",
]

# --- 2. Define Generation Ablations ---
# These map to the --generation_ablation flag in main.py
GENERATION_ABLATIONS = [
    "all_steps",
    "no_refine",
    "no_plan",
    "no_compilation",
    "no_relevance",
    "no_static_analysis",
    "no_runtime_feedback", 
]

# --- 3. Setup Logging ---
script_logger = logging.getLogger()
script_logger.setLevel(logging.INFO)
if not script_logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    script_logger.addHandler(handler)

# --- 4. Retry Function ---
def run_command_with_retry(cmd: List[str], log_file: Path, max_attempts: int = 3):
    """Runs a command with retries and logs stdout/stderr to a file."""
    for attempt in range(1, max_attempts + 1):
        script_logger.info(f"Attempt {attempt} of {max_attempts}: {' '.join(cmd)}")
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.run(
                    cmd,
                    check=True, 
                    text=True,
                    stdout=f,
                    stderr=f
                )
            script_logger.info(f"Command succeeded. Log saved to: {log_file}")
            return 0 # Success
        except subprocess.CalledProcessError as e:
            script_logger.warning(f"Command failed with exit code {e.returncode}. Retrying in 2s...")
            time.sleep(2)
        except FileNotFoundError:
            script_logger.error(f"Error: 'python' command not found. Cannot continue.")
            return 127 # Command not found
        except Exception as e:
            script_logger.error(f"An unexpected error occurred: {e}. Retrying in 2s...")
            time.sleep(2)
    
    script_logger.error(f"Command failed after {max_attempts} attempts. See log: {log_file}")
    return 1 # Failure

# --- 5. Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Systematic ablation runner for RepGen")
    parser.add_argument("--start_bug_id", required=True, type=int, help="First bug ID to process (e.g., 1)")
    parser.add_argument("--end_bug_id", required=True, type=int, help="Last bug ID to process (e.g., 10)")
    parser.add_argument("--max-gen-attempts", type=int, default=5, 
                        help="Max attempts for *each* code generation run (passed to main.py)")
    parser.add_argument("--max-run-attempts", type=int, default=3,
                        help="Max retry attempts for *each script execution*")
    args = parser.parse_args()

    if args.start_bug_id > args.end_bug_id:
        script_logger.error("Error: Start bug ID must be less than or equal to end bug ID")
        sys.exit(1)

    base_dir = Path(__file__).parent
    logs_base_dir = base_dir / "logs"

    # Loop through bug IDs
    for i in range(args.start_bug_id, args.end_bug_id + 1):
        bug_id = f"{i:03d}" # Format as "001", "002", etc.
        script_logger.info(f"\n===== Processing bug_id: {bug_id} =====")
        
        logs_dir = logs_base_dir / f"bug_{bug_id}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # --- STUDY 1: Retrieval Ablations (ACTIVE) ---
        # Runs each retrieval ablation with the FULL generation pipeline.
        script_logger.info("========== RUNNING STUDY 1: RETRIEVAL ABLATIONS ==========")
        
        for ret_ablation in RETRIEVAL_ABLATIONS:
            
            if ret_ablation == "full_system":
                log_name = f"bug_{bug_id}_baseline_retrieval.log"
                script_logger.info("\nRunning baseline retrieval (full_system):")
            else:
                log_name = f"bug_{bug_id}_{ret_ablation}.log"
                script_logger.info(f"\nRunning with {ret_ablation}:")
                
            log_file_path = logs_dir / log_name

            cmd = [
                "python", "tool_openai.py",
                f"--bug_id={bug_id}",
                f"--max-attempts={args.max_gen_attempts}",
                f"--retrieval_ablation={ret_ablation}",
                "--generation_ablation=all_steps" # Always use full generation pipeline
            ]
            
            run_command_with_retry(cmd, log_file_path, args.max_run_attempts)

        # --- STUDY 2: Generation Ablations (DISABLED) ---
        # To run this study, uncomment the lines below.
        # This will use the "full_system" context and create outputs
        # in the "ablations/" sub-folder for each generation step.
        # -------------------------------------------------------------
        
        script_logger.info("========== SKIPPING STUDY 2: GENERATION ABLATIONS ==========")

        # script_logger.info("========== RUNNING STUDY 2: GENERATION ABLATIONS ==========")
        # # We skip the first item ("all_steps") because it was run in Study 1
        # for gen_ablation in GENERATION_ABLATIONS[1:]:
        #     script_logger.info(f"\nRunning with {gen_ablation}:")
        #     log_name = f"bug_{bug_id}_{gen_ablation}.log"
        #     log_file_path = logs_dir / log_name
            
        #     cmd = [
        #         "python", "main.py",
        #         f"--bug_id={bug_id}",
        #         f"--max-attempts={args.max_gen_attempts}",
        #         "--retrieval_ablation=full_system", # Use full_system context
        #         f"--generation_ablation={gen_ablation}"
        #     ]
        #     run_command_with_retry(cmd, log_file_path, args.max_run_attempts)

        # -------------------------------------------------------------

        script_logger.info(f"\nAll logs for bug_id {bug_id} have been saved to: {logs_dir}")
        try:
            if sys.platform == "win32":
                subprocess.run(["cmd", "/c", "dir", str(logs_dir)], check=True)
            else:
                subprocess.run(["ls", "-l", str(logs_dir)], check=True)
        except Exception as e:
            script_logger.warning(f"Could not list log directory contents: {e}")

    script_logger.info(f"\n===== Completed processing all bug IDs from {args.start_bug_id} to {args.end_bug_id} =====")

if __name__ == "__main__":
    main()