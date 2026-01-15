import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import RetrievalPipeline
import argparse
import os
import subprocess
import json
import logging
from typing import Tuple, Dict, Any
import ast

import openai
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

# ==========================================
# PRODUCTION LOGGING CONFIGURATION
# ==========================================
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

logging.Logger.success = success

class ProductionFormatter(logging.Formatter):
    """
    Produces aligned, colored logs similar to Nginx or sophisticated CLI tools.
    Format: [HH:MM:SS] [LEVEL  ] Message
    """
    # ANSI Colors
    GREY = "\x1b[38;5;240m"
    BLUE = "\x1b[38;5;39m"
    GREEN = "\x1b[38;5;82m"
    YELLOW = "\x1b[38;5;226m"
    RED = "\x1b[38;5;196m"
    BOLD_RED = "\x1b[31;1m"
    MAGENTA = "\x1b[38;5;213m"
    CYAN = "\x1b[38;5;51m"
    RESET = "\x1b[0m"

    def format(self, record):
        # Timestamp in Grey
        dt = self.formatTime(record, "%H:%M:%S")
        timestamp = f"{self.GREY}{dt}{self.RESET}"

        # Level with color - removed spaces inside brackets
        if record.levelno == logging.INFO:
            level_fmt = f"{self.BLUE}[INFO]{self.RESET}"
        elif record.levelno == SUCCESS_LEVEL_NUM:
            level_fmt = f"{self.GREEN}[SUCCESS]{self.RESET}"
        elif record.levelno == logging.WARNING:
            level_fmt = f"{self.YELLOW}[WARNING]{self.RESET}"
        elif record.levelno == logging.ERROR:
            level_fmt = f"{self.RED}[ERROR]{self.RESET}"
        elif record.levelno == logging.CRITICAL:
            level_fmt = f"{self.BOLD_RED}[CRITICAL]{self.RESET}"
        elif record.levelno == logging.DEBUG:
            level_fmt = f"{self.GREY}[DEBUG]{self.RESET}"
        else:
            level_fmt = f"{self.CYAN}[CUSTOM]{self.RESET}"

        # Add context information if available
        context = ""
        if hasattr(record, 'bug_id'):
            context = f"{self.MAGENTA}[Bug {record.bug_id}]{self.RESET} "
        if hasattr(record, 'context_num'):
            context += f"{self.CYAN}[Ctx {record.context_num}]{self.RESET} "
        if hasattr(record, 'attempt'):
            context += f"{self.YELLOW}[Att {record.attempt}]{self.RESET} "

        return f"{timestamp} {level_fmt} {context}{record.getMessage()}"

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up production-ready logging with optional file output.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional file path for log output
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ProductionFormatter())
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger

logger = setup_logging()
# ==========================================

# Set environment variables at the start
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

RETRIEVAL_ABLATION_CONFIGS = {
    "full_system": {},
    "NO_BM25": {"NO_BM25": True},
    "NO_ANN": {"NO_ANN": True},
    "NO_RERANKER": {"NO_RERANKER": True},
    "NO_TRAINING_LOOP_EXTRACTION": {"NO_TRAINING_LOOP_EXTRACTION": True},
    "NO_TRAINING_LOOP_RANKING": {"NO_TRAINING_LOOP_RANKING": True},
    "NO_MODULE_PARTITIONING": {"NO_MODULE_PARTITIONING": True},
    "NO_DEPENDENCY_EXTRACTION": {"NO_DEPENDENCY_EXTRACTION": True},
}

GENERATION_ABLATION_MAP = {
    "all_steps": {},
    "no_refine": {"no_refine": True},
    "no_plan": {"no_plan": True},
    "no_compilation": {"no_compilation": True},
    "no_relevance": {"no_relevance": True},
    "no_static_analysis": {"no_static_analysis": True},
    "no_runtime_feedback": {"no_runtime_feedback": True},
}

def query_openai_api(prompt: str, model: str = "gpt-4.1", temperature: float = 0.0, max_retries: int = 3) -> str:
    """
    Sends a prompt to the OpenAI API and returns the text response with retry logic.

    Args:
        prompt: The input prompt string.
        model: The model to use (e.g., "gpt-4.1").
        temperature: The sampling temperature.
        max_retries: Maximum number of retry attempts for transient errors.

    Returns:
        The content of the AI's response, or an empty string on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return ""
    
    # Sanitize API key for logging (show only last 4 chars)
    sanitized_key = f"...{api_key[-4:]}" if len(api_key) > 4 else "****"
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} for OpenAI API call")
            
            client = OpenAI(api_key=api_key)
            
            logger.debug(f"Calling OpenAI API: model={model}, temp={temperature}, prompt_len={len(prompt)}")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature, 
                max_tokens=4096,
            )
            
            content = response.choices[0].message.content
            if content:
                logger.debug(f"Received response: {len(content)} chars")
                return content.strip()
            else:
                logger.warning(f"API response missing 'content' field: {response}")
                return ""
        
        except openai.AuthenticationError as e:
            logger.error(f"Authentication failed with API key {sanitized_key}: {e}")
            return ""  # Don't retry auth errors
        
        except RateLimitError as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"Rate limit exceeded. Waiting {wait_time}s before retry...")
            if attempt < max_retries - 1:
                import time
                time.sleep(wait_time)
            else:
                logger.error(f"Rate limit exceeded after {max_retries} attempts: {e}")
                return ""
        
        except APIConnectionError as e:
            logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to connect after {max_retries} attempts")
                return ""
        
        except APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return ""  # Don't retry API errors
        
        except Exception as e:
            logger.error(f"Unexpected error in query_openai_api: {type(e).__name__}: {e}", exc_info=True)
            return ""
    
    return ""

def main():
    parser = argparse.ArgumentParser(description="Code generation for bug reproduction")
    parser.add_argument("--bug_id", required=True, help="Bug ID to analyze (e.g., 001)")
    
    parser.add_argument(
        "--retrieval_ablation", 
        type=str, 
        default="full_system", 
        choices=RETRIEVAL_ABLATION_CONFIGS.keys(),
        help="Name of retrieval ablation config to use"
    )
    
    parser.add_argument(
        "--generation_ablation",
        type=str,
        default="all_steps",
        choices=GENERATION_ABLATION_MAP.keys(),
        help="Name of generation ablation config to use"
    )

    parser.add_argument("--max-attempts", type=int, default=5,
                          help="Maximum attempts for code generation")
    
    parser.add_argument("--ae_dataset_path", type=str, default=None,
                          help="Optional path to ae_dataset (uses dataset by default)")
    
    parser.add_argument("--log-level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                          help="Logging level")
    
    parser.add_argument("--log-file", type=str, default=None,
                          help="Optional log file path")
    
    args = parser.parse_args()
    
    # Re-initialize logger with arguments
    global logger
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logging(log_level=log_level, log_file=args.log_file)
    
    logger.info("=" * 60)
    logger.info("RepGen Code Generation Pipeline - Starting")
    logger.info("=" * 60)
    logger.info(f"Bug ID: {args.bug_id}", extra={'bug_id': args.bug_id})
    logger.info(f"Retrieval Ablation: {args.retrieval_ablation}")
    logger.info(f"Generation Ablation: {args.generation_ablation}")
    logger.info(f"Max Attempts: {args.max_attempts}")
    if args.ae_dataset_path:
        logger.info(f"Custom Dataset Path: {args.ae_dataset_path}")
    logger.info("=" * 60)

    # Setup pipeline and flags
    ret_ablation_name = args.retrieval_ablation
    gen_ablation_name = args.generation_ablation
    
    ret_ablation_dict = RETRIEVAL_ABLATION_CONFIGS.get(ret_ablation_name, {})
    gen_flags = GENERATION_ABLATION_MAP.get(gen_ablation_name, {})
    
    logger.info("Initializing retrieval pipeline...", extra={'bug_id': args.bug_id})
    try:
        pipeline = RetrievalPipeline(
            bug_id=args.bug_id, 
            ablation_config=ret_ablation_dict, 
            retrieval_ablation_name=ret_ablation_name,
            generation_ablation_name=gen_ablation_name,
            dataset_dir=args.ae_dataset_path
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True, extra={'bug_id': args.bug_id})
        return
    
    logger.info("Running retrieval pipeline...", extra={'bug_id': args.bug_id})
    try:
        result = pipeline.run_pipeline(args.bug_id)
        logger.success(f"Retrieval pipeline completed: {result['status']}", extra={'bug_id': args.bug_id})
    except Exception as e:
        logger.error(f"Retrieval pipeline failed: {e}", exc_info=True, extra={'bug_id': args.bug_id})
        return
 
    # Get all paths from config
    config = pipeline.config
    logger.debug(f"Context Directory: {config.CONTEXT_DIR_IN}")
    logger.debug(f"Output Directory: {config.REPRODUCTION_DIR_OUT}")
    
    # Bug report handling
    bug_report_path = config.BUG_REPORTS_DIR / f"{args.bug_id}.txt"
    try:
        with open(bug_report_path, 'r', encoding='utf-8') as f:
            bug_report_content = f.read()
        logger.info(f"Loaded bug report: {len(bug_report_content)} chars", extra={'bug_id': args.bug_id})
    except FileNotFoundError:
        logger.error(f"Bug report not found: {bug_report_path}", extra={'bug_id': args.bug_id})
        return
    except Exception as e:
        logger.error(f"Failed to read bug report: {e}", exc_info=True, extra={'bug_id': args.bug_id})
        return
    
    refined_report_dir = config.REFINED_BUG_REPORT_DIR_OUT
    refined_report_path = refined_report_dir / f"{args.bug_id}.txt"
    
    # STEP 1: Bug Report Refinement
    logger.info("─" * 60)
    logger.info("STEP 1/3: Bug Report Refinement", extra={'bug_id': args.bug_id})
    logger.info("─" * 60)
    
    if not gen_flags.get("no_refine", False):
        logger.info("Refining bug report with LLM...", extra={'bug_id': args.bug_id})
        prompt_refinement = create_prompt_refinement(bug_report_content)
        
        accumulated_output = query_openai_api(prompt_refinement, model="gpt-4.1", temperature=0.5)
        
        if accumulated_output:
            try:
                with open(refined_report_path, 'w', encoding='utf-8') as f:
                    f.write(accumulated_output)
                logger.success(f"Refined bug report saved: {refined_report_path.name} ({len(accumulated_output)} chars)", 
                             extra={'bug_id': args.bug_id})
            except Exception as e:
                logger.error(f"Failed to save refined report: {e}", extra={'bug_id': args.bug_id})
                return
        else:
            logger.warning("LLM returned empty response, using original report", extra={'bug_id': args.bug_id})
            with open(refined_report_path, 'w', encoding='utf-8') as f:
                f.write(bug_report_content)
    else:
        logger.info("Refinement skipped (ablation active). Using original report.", extra={'bug_id': args.bug_id})
        with open(refined_report_path, 'w', encoding='utf-8') as f:
            f.write(bug_report_content)
    
    # STEP 2: Plan Generation
    logger.info("─" * 60)
    logger.info("STEP 2/3: Plan Generation", extra={'bug_id': args.bug_id})
    logger.info("─" * 60)
    
    context_dir = config.CONTEXT_DIR_IN 
    plan_dir = config.PLANS_DIR_OUT
    
    try:
        context_files_list = os.listdir(context_dir)
        logger.info(f"Found {len(context_files_list)} context files", extra={'bug_id': args.bug_id})
    except Exception as e:
        logger.error(f"Failed to list context files: {e}", extra={'bug_id': args.bug_id})
        return

    for idx, context_file in enumerate(context_files_list, 1):
        context_path = os.path.join(context_dir, context_file)
        
        logger.info(f"Processing context {idx}/{len(context_files_list)}: {context_file}", 
                   extra={'bug_id': args.bug_id})
        
        try:
            with open(context_path, 'r', encoding='utf-8') as f:
                context_content = json.loads(f.read())
            logger.debug(f"Loaded context: {len(str(context_content))} chars")
        except Exception as e:
            logger.error(f"Failed to load context file {context_file}: {e}", extra={'bug_id': args.bug_id})
            continue
        
        prompt_plan = create_prompt_plan(bug_report_content, context_content)
        
        if not gen_flags.get("no_plan", False):
            output = query_openai_api(prompt_plan, model="gpt-4.1", temperature=0.0)
            if not output:
                logger.warning(f"Empty plan generated for {context_file}", extra={'bug_id': args.bug_id})
                output = "[]"
        else:
            logger.debug("Plan generation skipped (ablation active)")
            output = "[]"
    
        plan_path = plan_dir / f"plan_{context_file.split('.')[0]}.json"
        if 'module' in plan_path.name:
            plan_path = plan_path.with_name(plan_path.name.replace('module_', ''))
        
        try:
            with open(plan_path, 'w', encoding='utf-8') as f:
                f.write(output)
            logger.success(f"Plan saved: {plan_path.name}", extra={'bug_id': args.bug_id})
        except Exception as e:
            logger.error(f"Failed to save plan: {e}", extra={'bug_id': args.bug_id})

    # STEP 3: Code Generation
    logger.info("─" * 60)
    logger.info("STEP 3/3: Code Generation & Verification", extra={'bug_id': args.bug_id})
    logger.info("─" * 60)
    
    refined_report_path = config.REFINED_BUG_REPORT_DIR_IN / f"{args.bug_id}.txt"
    try:
        with open(refined_report_path, 'r', encoding='utf-8') as f:
            refined_bug_report = f.read()
    except Exception as e:
        logger.error(f"Failed to load refined bug report: {e}", extra={'bug_id': args.bug_id})
        return
    
    try:
        context_files = sorted(config.CONTEXT_DIR_IN.glob("*.json"), 
                              key=lambda x: int(x.stem.split('_')[-1]))
        logger.info(f"Processing {len(context_files)} contexts", extra={'bug_id': args.bug_id})
    except Exception as e:
        logger.error(f"Failed to enumerate context files: {e}", extra={'bug_id': args.bug_id})
        return
    
    reproduction_dir = config.REPRODUCTION_DIR_OUT
    plan_dir = config.PLANS_DIR_IN 
    
    for ctx_idx, context_file in enumerate(context_files, 1):
        context_num = context_file.stem.split('_')[-1]
        
        logger.info("=" * 60, extra={'bug_id': args.bug_id, 'context_num': context_num})
        logger.info(f"Processing Context {ctx_idx}/{len(context_files)} (ID: {context_num})", 
                   extra={'bug_id': args.bug_id, 'context_num': context_num})
        logger.info("=" * 60, extra={'bug_id': args.bug_id, 'context_num': context_num})
        
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context_content = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load context: {e}", 
                        extra={'bug_id': args.bug_id, 'context_num': context_num})
            continue
        
        plan_file = plan_dir / f"plan_{args.bug_id}_{context_num}.json"
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                plan_content = f.read()
        except Exception as e:
            logger.warning(f"Failed to load plan, using empty: {e}", 
                          extra={'bug_id': args.bug_id, 'context_num': context_num})
            plan_content = "[]"
        
        prompt_code = _build_prompt(
            refined_bug_report,
            context_content,
            plan_content
        )
        
        max_attempts = args.max_attempts
        success = False
        
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt + 1}/{max_attempts}", 
                       extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
            
            stdout = query_openai_api(prompt_code, model="gpt-4.1", temperature=0.0)
            
            if not stdout:
                logger.error("LLM returned empty response", 
                           extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
                continue
            
            output = stdout
            start_marker = "```python"
            end_marker = "```"
            start_idx = output.find(start_marker)
            end_idx = output.find(end_marker, start_idx + len(start_marker)) if start_idx != -1 else -1

            if start_idx != -1 and end_idx != -1:
                extracted_code = output[start_idx + len(start_marker):end_idx].strip()
                logger.debug("Extracted code from markdown block")
            else:
                extracted_code = output
                logger.warning("No markdown code block found, using full output", 
                             extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})

            output_file = reproduction_dir / f"reproduce_{args.bug_id}_{context_num}_attempt{attempt + 1}.py"
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_code)
                logger.debug(f"Saved attempt to: {output_file.name}")
            except Exception as e:
                logger.error(f"Failed to save code: {e}", 
                           extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
                continue

            # Structural correctness check
            if not gen_flags.get("no_compilation", False):
                is_correct, feedback = check_structural_correctness(extracted_code)
                
                if not is_correct:
                    logger.warning(f"Structural check failed: {feedback.splitlines()[0][:80]}...", 
                                 extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
                    prompt_code = _build_prompt(
                        refined_bug_report,
                        context_content,
                        plan_content,
                        feedback=f"Code failed structural check:\n{feedback}\nPlease fix these issues."
                    )
                    continue
                else:
                    logger.success("Structural check passed", 
                                 extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
            
            # Relevance check
            if not gen_flags.get("no_relevance", False):
                relevance_check = check_relevance(refined_bug_report, extracted_code)
                if not relevance_check:
                    logger.warning("Relevance check failed", 
                                 extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
                    prompt_code = _build_prompt(
                        refined_bug_report,
                        context_content,
                        plan_content,
                        feedback="Generated code is not relevant to the bug report."
                    )
                    continue
                else:
                    logger.success("Relevance check passed", 
                                 extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
            
            # Static analysis
            if not gen_flags.get("no_static_analysis", False):
                pylint_output = analyze_with_pylint(extracted_code, output_file)
                if pylint_output:
                    logger.info(f"Static analysis found {len(pylint_output)} critical issues", 
                              extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
                    for issue in pylint_output[:3]:  # Show first 3
                        logger.debug(f"  - {issue}")
                    
                    refactor_prompt = _build_refactor_prompt(
                        extracted_code,
                        "\n".join(pylint_output),
                        refined_bug_report
                    )
                    refactored_code = _run_llm_refactor(refactor_prompt)
                    if refactored_code:
                        extracted_code = refactored_code
                        logger.success("Code refactored", 
                                     extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
                else:
                    logger.success("Static analysis passed", 
                                 extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
            
            # Runtime feedback
            if not gen_flags.get("no_runtime_feedback", False):
                score, feedback = calculate_probability_of_reproduction(extracted_code, refined_bug_report)
                logger.info(f"Confidence score: {score:.2f}", 
                          extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
            
                if score > 0.7:
                    final_output_file = reproduction_dir / f"reproduce_{args.bug_id}.py"
                    try:
                        with open(final_output_file, 'w', encoding='utf-8') as f:
                            f.write(extracted_code)
                        logger.success(f"✓ Valid reproduction found! Saved to: {final_output_file.name}", 
                                     extra={'bug_id': args.bug_id, 'context_num': context_num})
                        
                        analysis = analyze_bug_reproduction(extracted_code, refined_bug_report)
                        if analysis:
                            logger.info("Analysis Summary:", extra={'bug_id': args.bug_id})
                            for line in analysis.split('\n')[:10]:  # First 10 lines
                                logger.info(f"  {line}")
                        
                        success = True
                        return
                    except Exception as e:
                        logger.error(f"Failed to save final code: {e}", 
                                   extra={'bug_id': args.bug_id, 'context_num': context_num})
                else:
                    logger.warning(f"Low confidence ({score:.2f}), retrying...", 
                                 extra={'bug_id': args.bug_id, 'context_num': context_num, 'attempt': attempt + 1})
                    logger.debug(f"Feedback: {feedback[:200]}")
                    prompt_code = _build_prompt(
                        refined_bug_report,
                        context_content,
                        plan_content,
                        feedback=f"Code failed to reproduce the bug with confidence score {score:.2f}. Feedback: {feedback}"
                    )
            else:
                final_output_file = reproduction_dir / f"reproduce_{args.bug_id}.py"
                try:
                    with open(final_output_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_code)
                    logger.success(f"Code generated (no feedback mode): {final_output_file.name}", 
                                 extra={'bug_id': args.bug_id, 'context_num': context_num})
                    success = True
                    break
                except Exception as e:
                    logger.error(f"Failed to save code: {e}", 
                               extra={'bug_id': args.bug_id, 'context_num': context_num})
        
        if not success:
            logger.error(f"Failed to generate valid code after {max_attempts} attempts", 
                        extra={'bug_id': args.bug_id, 'context_num': context_num})
    
    logger.info("=" * 60)
    logger.info("Pipeline Completed", extra={'bug_id': args.bug_id})
    logger.info("=" * 60)

def analyze_with_pylint(code: str, file_path: Path):
    """Run pylint analysis on code and return critical errors."""
    result = []
    try:
        from pylint.lint import Run
        from pylint.reporters.text import TextReporter
        from io import StringIO
        
        file_path.write_text(code, encoding='utf-8')
        
        pylint_output = StringIO()
        reporter = TextReporter(pylint_output)
        Run([str(file_path)], reporter=reporter, exit=False)
        raw_output = pylint_output.getvalue()
        
        lines = raw_output.splitlines()
        for line in lines:
            if "e0" in line.lower() and "import-error" not in line.lower():
                result.append(line.strip())
        
        logger.debug(f"Pylint found {len(result)} critical errors")
        return result
    except ImportError:
        logger.warning("Pylint not available, skipping static analysis")
        return []
    except Exception as e:
        logger.warning(f"Pylint analysis failed: {e}")
        return []

def analyze_bug_reproduction(code: str, bug_report: str) -> str:
    """Analyze which parts of the code are most likely to reproduce the bug."""
    prompt = f"""
    Analyze the following bug reproduction code and identify which sections are most likely
    to be responsible for reproducing the reported bug. Provide line numbers and explanations.

    Bug Report:
    {bug_report}

    Reproduction Code:
    {code}

    Instructions:
    1. Identify the key sections that match the bug description
    2. Highlight specific lines that likely trigger the bug
    3. Explain why these sections are likely responsible
    4. Note any error-prone patterns
    5. Provide confidence level (High/Medium/Low) for each identified section

    Format your response as:
    - Lines X-Y: [Description] (Confidence: High/Medium/Low)
      [Explanation]
    """
    
    output = query_openai_api(prompt, model="gpt-4.1", temperature=0.0)
    return output if output else "Analysis not available"

def _build_refactor_prompt(code: str, issues: str, bug_report: str) -> str:
    """Build a prompt for code refactoring based on feedback."""
    return f"""
    Refactor the following Python code to fix the identified issues while maintaining 
    its ability to reproduce the reported bug. Preserve all functionality related to the bug.

    Bug Report:
    {bug_report}

    Original Code:
    {code}

    Identified Issues:
    {issues}

    Instructions:
    1. Fix all syntax and major style issues
    2. Preserve all bug reproduction logic
    3. Maintain the same imports and core functionality
    4. Keep the same variable names where possible
    5. Add comments if changes might affect bug reproduction

    Output only the refactored code in a Python code block:
    ```python
    [refactored code]
    ```
    """

def _run_llm_refactor(prompt: str) -> str:
    """Execute the LLM to refactor code based on PyLint feedback and bug report."""
    logger.debug("Requesting code refactoring from LLM")
    stdout = query_openai_api(prompt, model="gpt-4.1", temperature=0.0)
    
    if not stdout:
        logger.warning("LLM returned empty refactoring response")
        return ""
    
    output = stdout.strip()
    
    start_marker = "```python"
    end_marker = "```"
    start_idx = output.find(start_marker)
    
    if start_idx != -1:
        end_idx = output.find(end_marker, start_idx + len(start_marker))
        if end_idx != -1:
            extracted_code = output[start_idx + len(start_marker):end_idx].strip()
            logger.debug("Extracted refactored code from markdown block")
            return extracted_code
    
    logger.warning("No code block markers found in refactoring output")
    return output

def check_relevance(bug_report: str, code: str) -> bool:
    """Check if generated code is relevant to the bug report."""
    prompt = f"""System:
            You are a helpful AI software engineer specializing in identifying
            relevant code segments given a bug report. Analyze the provided bug
            report and the code segment to determine if the code segment is
            relevant to the bug described in the bug report.

            There are two possible outputs:
            - 'yes': The code is relevant to the bug described in the bug report.
            - 'no': The code is NOT relevant to the bug described in the bug report.

            Provide your output in JSON format like this sample: {{"relevance": "yes"}}.

            Bug Report:
            {bug_report}

            Code Segment:
            {code}

            Output only the JSON response with no additional commentary:"""

    logger.debug("Checking code relevance with LLM")
    stdout = query_openai_api(prompt, model="gpt-4.1", temperature=0.0)

    if not stdout:
        logger.warning("Empty response from relevance check, assuming not relevant")
        return False

    try:
        json_str = extract_json_content(stdout)
        if not json_str:
            json_str = stdout
            
        response = json.loads(json_str)
        is_relevant = response.get('relevance', '').lower() == 'yes'
        logger.debug(f"Relevance check result: {is_relevant}")
        return is_relevant
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON for relevance check: {e}")
        logger.debug(f"Raw response: {stdout[:200]}")
        return 'yes' in stdout.lower()

def extract_json_content(text):
    """Extract JSON content from markdown code blocks or raw text."""
    import re
    pattern = r'```(?:json)?\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1)
    
    if text.strip().startswith("{") and text.strip().endswith("}"):
        return text.strip()
        
    return None

def calculate_probability_of_reproduction(code: str, bug_report: str) -> Tuple[float, str]:
    """Calculate probability that code will reproduce the bug."""
    
    # Stage 1: Code Behavior Prediction
    code_analysis_prompt = f"""Analyze this deep learning code systematically:

<Code>
{code}
</Code>

Perform detailed execution simulation with fault taxonomy checking:

1. Execution Path Analysis:
   - List all possible execution paths through the code
   - For each path:
     * Program states at key points (tensor shapes, values)
     * Variable value ranges and distributions
     * Conditions that trigger each path

2. Taxonomy Fault Detection:
MODEL FAULTS

Model Type & Properties: Structure, initialization, depth
Layers:
Missing/Redundant/Wrong Layer
Layer Properties (shape, size, neurons)
Activation Functions
Tensors & Inputs:
Tensor Shape (padding, indexing, orientation)
Input Format (datatype, dimensions, structure)
TRAINING FAULTS

Hyperparameters (learning rate, batch size, epochs)
Loss Function (selection, implementation)
Validation/Testing (metrics, data splits)
Preprocessing of Training Data:
Missing preprocessing steps
Wrong preprocessing implementation
Optimizer (selection, parameter tuning)
Training Data Quality:
Data quantity
Class balance
Label accuracy
Data consistency
Training Process:
Memory management
Checkpoints
Data augmentation
GPU USAGE FAULTS

Device references
Parallelism implementation
Process state sharing
Data transfer operations
API FAULTS

API usage conformity
API call positioning
Missing API calls

TENSORS & INPUTS FAULT DETECTION

Wrong Tensor Shape

Check for incompatible tensor shapes during operations
Verify output padding configurations
Confirm correct tensor indexing
Ensure proper tensor orientation (normal vs transposed)
Wrong Input

Verify data type compatibility (e.g., float vs string)
Check input shape dimensions (e.g., 5x5 vs 5x10)
Validate input format correctness
Confirm proper channel ordering (e.g., channel-first vs channel-last)

3. Failure Prediction:
   - For each path, predict possible failure symptoms based on the taxonomy below (Silent Bugs in Deep Learning Frameworks: An Empirical Study of Keras and TensorFlow), and (Demystifying and Detecting Misuses of Deep Learning APIs):
    SYMPTOMS OF DL SYSTEM FAULTS

Program Crashes (Most Common API Misuse)
Immediate program failures
Version update incompatibilities
Code refactoring issues
Explicit error messages with line numbers
Silent/Unexpected Output Issues
A. Wrong Calculation (29.9% of silent bugs)

Incorrect gradient computations
Incorrect loss metric calculations
No error messages thrown
B. Wrong Parameter Setting (20.8% of silent bugs)

Incorrect learning rate behavior
Parameter setting not taking effect
Functionality continues with wrong parameters
C. Wrong Displayed Message (19.5% of silent bugs)

UI/console message issues
Misleading progress indicators
Incorrect logging information
D. Wrong Save/Reload (13% of silent bugs)

Accuracy loss after model reload
Missing components after restoration
Inconsistent model behavior
E. Wrong Resulting Shape (7.79% of silent bugs)

Incorrect tensor shapes
Shape mismatch without errors
Silent dimension issues
Performance Issues
A. Low Efficiency (API Misuse)

Slow execution speed
GPU configuration issues
Resource utilization problems
B. Performance Degradation (5.19% of silent bugs)

Memory usage issues
Execution speed degradation
Resource management problems
Return Warnings (Least Common API Misuse)

Deprecated API usage warnings
Backward compatibility issues
Version mismatch notifications
Wrong Resulting Structure (3.9% of silent bugs)
Unexpected model architecture changes
Duplicate layer creation
Framework handling inconsistencies


Format your response as JSON with:
{{
    "execution_paths": [
        {{
            "path_description": str,
            "program_states": [str],
            "outputs": [str],
            "potential_symptoms": [str]
        }}
    ],
}}"""

    logger.debug("Analyzing code behavior with LLM")
    analysis_result_raw = query_openai_api(code_analysis_prompt, model="gpt-4.1", temperature=0.0)
    analysis_result = extract_json_content(analysis_result_raw)
    if not analysis_result:
        logger.warning("Could not extract JSON from behavior prediction")
        analysis_result = "{}"

    # Stage 2: Symptom Extraction from Bug Report
    symptom_extraction_prompt = f"""Extract technical symptoms from this bug report:
    
<Bug Report>
{bug_report}
</Bug Report>

Identify and list:
1. Exact error messages/text
2. System behavior observations
3. Pre-failure conditions
4. Post-failure states

Format as JSON with:
{{
    "reported_symptoms": [str],
    "required_trigger_conditions": [str],
    "error_manifestations": [str]
}}"""

    logger.debug("Extracting symptoms from bug report")
    extraction_result_raw = query_openai_api(symptom_extraction_prompt, model="gpt-4.1", temperature=0.0)
    extraction_result = extract_json_content(extraction_result_raw)
    if not extraction_result:
        logger.warning("Could not extract JSON from symptom extraction")
        extraction_result = "{}"

    # Stage 3: Comparison
    COMPARISON_PROMPT = f"""
Analyze the code, symptoms from the execution analysis, bug report and the actual bug report to determine if the code will reproduce the bug:

### SYMPTOMS FROM THE EXECUTION ANALYSIS:
{analysis_result}

### SYMPTOMS FROM THE BUG REPORT:
{extraction_result}

### ACTUAL CODE:
{code}

FULL BUG REPORT:
{bug_report}

Provide JSON output with ONLY these 2 fields:
    "score": A number 0-1 (probability of reproducing the bug)
    "feedback": A string explaining why it might/might not reproduce

Consider these factors for matching:

1. Symptom Alignment:
   - Do reported symptoms match potential symptoms in analysis?
   - Are error manifestations consistent?
   - Do error timings/triggers align?
   - Are failure modes similar?

2. Execution Flow:
   - Do code paths contain reported problematic sections?
   - Are critical operations present and similar?
   - Do program states match bug conditions?
   - Are execution sequences comparable?

3. Resource Patterns:
   - Do resource usage patterns align?
   - Are memory/computation requirements similar?
   - Do hardware dependencies match?
   - Are performance characteristics comparable?

4. Error Triggers: Do trigger conditions exist in code?

Return ONLY valid JSON with NO additional text:
"""
    logger.debug("Calculating final reproduction probability")
    comparison_result_raw = query_openai_api(COMPARISON_PROMPT, model="gpt-4.1", temperature=0.0)
    comparison_result_json = extract_json_content(comparison_result_raw)
    
    if not comparison_result_json:
        logger.error("Could not extract JSON from comparison")
        return 0.0, "Failed to parse LLM response for score."

    try:
        comparison_result = json.loads(comparison_result_json)
        score = comparison_result.get('score', 0.0)
        feedback = comparison_result.get('feedback', "")
        logger.debug(f"Calculated score: {score}")
        return float(score), feedback
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {comparison_result_json[:200]}")
        return 0.0, "Failed to parse JSON response from LLM."
    except Exception as e:
        logger.error(f"Error in calculate_probability: {e}", exc_info=True)
        return 0.0, f"Error: {e}"

def parse_json_response(response: str, default: dict = None) -> dict:
    """Safe JSON parsing with fallback"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return default or {}

def check_structural_correctness(code: str) -> tuple[bool, str]:
    """
    Check if Python code is structurally correct and return (success, error_feedback).
    Uses AST parsing for safety (no code execution).
    """
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax Error: {e.msg}\nLine {e.lineno}: {e.text.strip() if e.text else 'N/A'}"
        return False, error_msg
    except Exception as e:
        return False, f"Structural Error: {str(e)}"

def _build_prompt(bug_report: str, code_context: str, plan: str, feedback: str = "") -> str:
    """Build the prompt for code generation, including feedback if provided"""
    prompt = f"""You are a senior software engineer fluent in reproducing deep learning bugs. Generate a code snippet to reproduce this bug:

            Bug Report:
            {bug_report}

            Relevant Code Context:
            {code_context}

            Reproduction Plan:
            {plan}"""

    if feedback:
        prompt += f"""

            Previous Attempt Feedback:
            {feedback}"""

    prompt += """

            Requirements:
            1. Minimal Python script
            2. Include necessary setup
            3. Do not add any explanation comments, pure code, nothing else.
            4. Use standard libraries where possible
            5. Mention dependencies in comments if needed
            6. Do not generate any code for the module, use the existing imports.
            7. Use the existing imports and their respective methods from the main file to generate the code snippet, as we will be using the code in the respective module for reproduction. Prioritize completness of the code, it should compile, so you can use the imports. 
            8. Main file and module snippets are the most important parts of the context to be used for generating the code, so, use them to reconstruct the code snippet for reproduction.
            9. The provided plan is a reference sequence of steps to be followed, so use that to generate the code snippet.
            Output ONLY the code without explanation:"""
    return prompt
 
def create_prompt_plan(bug_report_content, context):
    """Create a prompt for generating a reproduction plan."""
    if 'main_file' in context:
        file_paths = [context['main_file']['path']]
        file_contents = [context['main_file']['content']]
        dependencies = context.get('dependencies', [])
        dep_string = "\n\nImport Dependencies:\n" + json.dumps(dependencies, indent=2) if dependencies else ""
    elif 'module' in context:
        file_paths = [file['path'] for file in context['module']['files']]
        file_contents = [snippet['code'] for file in context['module']['files'] for snippet in file['snippets']]
        dep_string = ""

    file_paths_string = "\n\n".join([f"Module Path {i+1}:\n{json.dumps(path, indent=2)}" for i, path in enumerate(file_paths)])
    file_contents_string = "\n\n".join([f"Code Context {i+1}:\n{json.dumps(content, indent=2)}" for i, content in enumerate(file_contents)])

    return f"""You are a code generation planner. Create a detailed step-by-step plan to reproduce this bug. Focus on concrete, technical steps with specific values and assertions.
 
Bug Report:
{bug_report_content}
 
{file_paths_string}
 
{file_contents_string}{dep_string}
 
Your task is to create a precise technical plan that an LLM can follow to generate code that reproduces this bug. Each step should be specific and actionable.
 
Requirements:
- Include specific technical details (e.g., dimensions, batch sizes, function parameters)
- Focus only on reproducing the bug, not fixing it
- Include setup steps (imports, data preparation)
- Include validation steps to verify the bug occurs
- Make steps granular and specific
 
Output must be a valid JSON array of strings, formatted like this example:
[
    "Import TensorFlow and the inception module from inception_test.py",
    "Define a batch size of 5 and image dimensions of 299x299",
    "Create random uniform input data with shape (batch_size, height, width, 3)",
    "Call inception_v3 function with num_classes=1000",
    "Verify output contains NaN values in loss calculation",
    "Monitor GPU memory usage during execution",
    "Assert GPU memory exceeds expected threshold"
]
 
Generate plan steps:"""
 
def create_prompt_refinement(bug_report_content):
    """Create a prompt for refining the bug report."""
    return f"""You are a software development assistant. Analyze and restructure this bug report.
 
Original bug report:
{bug_report_content}
 
Provide your analysis in exactly this format:
 
TITLE
[One-line summary of the core issue]
 
SYMPTOMS
• [List each observed problem]
• [Include error messages exactly as shown]
• [Include all reported unexpected behaviors]
 
EXPECTED BEHAVIOR
[Describe what should happen when the software works correctly]
 
REPRODUCTION STEPS
1. [First step to reproduce]
2. [Next step]
3. [Continue until complete]
 
Begin your structured analysis:"""
 
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)