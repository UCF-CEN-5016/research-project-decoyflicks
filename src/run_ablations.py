import subprocess
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

# --- 1. Define Retrieval Ablations ---
# These map to the --retrieval_ablation flag in the tool script
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
# These map to the --generation_ablation flag in the tool script
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
            # Capture both stdout and stderr to the log file
            with open(log_file, 'w', encoding='utf-8') as f:
                process = subprocess.run(
                    cmd,
                    check=True, 
                    text=True,
                    stdout=f,
                    stderr=subprocess.STDOUT
                )
            script_logger.info(f"Command succeeded. Log saved to: {log_file}")
            return 0 # Success
        except subprocess.CalledProcessError as e:
            script_logger.warning(f"Command failed with exit code {e.returncode}. Retrying in 2s...")
            time.sleep(2)
        except FileNotFoundError:
            script_logger.error(f"Error: 'python' command not found or script missing. Cannot continue.")
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
    
    # New configuration arguments
    parser.add_argument("--tool_script", type=str, default="tool_openai.py", 
                        help="The script to run (e.g., tool.py or tool_openai.py)")
    parser.add_argument("--max-gen-attempts", type=int, default=5, 
                        help="Max attempts for *each* code generation run (passed to tool)")
    parser.add_argument("--max-run-attempts", type=int, default=3,
                        help="Max retry attempts for *each script execution*")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Optional custom dataset path to pass to the tool")

    args = parser.parse_args()

    if args.start_bug_id > args.end_bug_id:
        script_logger.error("Error: Start bug ID must be less than or equal to end bug ID")
        sys.exit(1)

    # Verify the tool script exists
    if not os.path.exists(args.tool_script):
        script_logger.error(f"Error: Tool script '{args.tool_script}' not found.")
        sys.exit(1)

    base_dir = Path(__file__).parent
    logs_base_dir = base_dir / "logs"

    # Loop through bug IDs
    for i in range(args.start_bug_id, args.end_bug_id + 1):
        bug_id = f"{i:03d}" # Format as "001", "002", etc.
        script_logger.info(f"\n===== Processing bug_id: {bug_id} using {args.tool_script} =====")
        
        logs_dir = logs_base_dir / f"bug_{bug_id}"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # --- STUDY 1: Retrieval Ablations (ACTIVE) ---
        script_logger.info("========== RUNNING STUDY 1: RETRIEVAL ABLATIONS ==========")
        
        for ret_ablation in RETRIEVAL_ABLATIONS:
            if ret_ablation == "full_system":
                log_name = f"bug_{bug_id}_baseline_retrieval.log"
                script_logger.info("\nRunning baseline retrieval (full_system):")
            else:
                log_name = f"bug_{bug_id}_{ret_ablation}.log"
                script_logger.info(f"\nRunning with {ret_ablation}:")
                
            log_file_path = logs_dir / log_name

            # Construct command dynamically
            cmd = [
                "python", args.tool_script,
                f"--bug_id={bug_id}",
                f"--max-attempts={args.max_gen_attempts}",
                f"--retrieval_ablation={ret_ablation}",
                "--generation_ablation=all_steps" # Always use full generation pipeline for retrieval study
            ]
            
            # Pass dataset path if provided (assuming the tool script accepts --ae_dataset_path)
            if args.dataset_path:
                cmd.append(f"--ae_dataset_path={args.dataset_path}")
            
            run_command_with_retry(cmd, log_file_path, args.max_run_attempts)

        script_logger.info("========== RUNNING STUDY 2: GENERATION ABLATIONS ==========")
        # We skip the first item ("all_steps") because it was run in Study 1 (as full_system)
        for gen_ablation in GENERATION_ABLATIONS[1:]:
            script_logger.info(f"\nRunning with {gen_ablation}:")
            log_name = f"bug_{bug_id}_{gen_ablation}.log"
            log_file_path = logs_dir / log_name
            
            cmd = [
                "python", args.tool_script,
                f"--bug_id={bug_id}",
                f"--max-attempts={args.max_gen_attempts}",
                "--retrieval_ablation=full_system", 
                f"--generation_ablation={gen_ablation}"
            ]
            
            if args.dataset_path:
                cmd.append(f"--ae_dataset_path={args.dataset_path}")

            run_command_with_retry(cmd, log_file_path, args.max_run_attempts)

    script_logger.info(f"\n===== Completed processing all bug IDs from {args.start_bug_id} to {args.end_bug_id} =====")

if __name__ == "__main__":
    main()