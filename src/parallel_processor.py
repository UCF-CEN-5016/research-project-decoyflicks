"""
Parallel Processing Module

Implements multiprocessing for Plan Generation and Code Generation phases.
Enables concurrent execution of independent tasks to significantly reduce runtime.
"""

import logging
import json
from pathlib import Path
from typing import Callable, List, Dict, Any, Tuple, Optional
from multiprocessing import Pool, cpu_count
import os

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    Manages parallel execution of tasks using multiprocessing.
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        Initialize the parallel processor.
        
        Args:
            num_workers: Number of worker processes (default: CPU count)
        """
        self.num_workers = num_workers or cpu_count()
        logger.info(f"Parallel processor initialized with {self.num_workers} workers")
    
    @staticmethod
    def process_plan_generation(args: Tuple) -> Dict[str, Any]:
        """
        Static method for plan generation (must be picklable for multiprocessing).
        
        Args:
            args: Tuple containing (context_idx, context_file, context_content, bug_report, 
                                   create_prompt_plan_fn, query_ollama_fn, gen_flags, 
                                   bug_id, plan_dir)
        
        Returns:
            Dictionary with results
        """
        (context_idx, context_file, context_content, bug_report, 
         create_prompt_plan_fn, query_ollama_fn, gen_flags, bug_id, plan_dir) = args
        
        context_num = Path(context_file).stem.split('_')[-1]
        
        try:
            prompt_plan = create_prompt_plan_fn(bug_report, context_content)
            
            if not gen_flags.get("no_plan", False):
                output = query_ollama_fn(prompt_plan, model='qwen2.5-coder:7b')
                if not output:
                    output = "[]"
            else:
                output = "[]"
            
            plan_path = Path(plan_dir) / f"plan_{Path(context_file).stem}.json"
            if 'module' in plan_path.name:
                plan_path = plan_path.with_name(plan_path.name.replace('module_', ''))
            
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            with open(plan_path, 'w', encoding='utf-8') as f:
                f.write(output)
            
            return {
                'status': 'success',
                'context_idx': context_idx,
                'context_num': context_num,
                'plan_path': str(plan_path),
                'message': f"Plan saved: {plan_path.name}"
            }
        except Exception as e:
            return {
                'status': 'error',
                'context_idx': context_idx,
                'context_num': context_num if 'context_num' in locals() else 'unknown',
                'error': str(e),
                'message': f"Failed to generate plan: {e}"
            }
    
    @staticmethod
    def process_code_generation(args: Tuple) -> Dict[str, Any]:
        """
        Static method for code generation (must be picklable for multiprocessing).
        
        Args:
            args: Tuple containing necessary parameters for code generation
        
        Returns:
            Dictionary with results
        """
        (context_idx, context_file, context_content, refined_bug_report,
         plan_content, bug_id, gen_flags, max_attempts, 
         query_ollama_fn, build_prompt_fn, check_structural_correctness_fn,
         check_relevance_fn, analyze_with_pylint_fn, build_refactor_prompt_fn,
         run_llm_refactor_fn, calculate_probability_of_reproduction_fn,
         reproduction_dir) = args
        
        context_num = Path(context_file).stem.split('_')[-1]
        
        try:
            prompt_code = build_prompt_fn(
                refined_bug_report,
                context_content,
                plan_content
            )
            
            for attempt in range(max_attempts):
                stdout = query_ollama_fn(prompt_code, model='qwen2.5-coder:7b')
                
                if not stdout:
                    continue
                
                output = stdout
                start_marker = "```python"
                end_marker = "```"
                start_idx = output.find(start_marker)
                end_idx = output.find(end_marker, start_idx + len(start_marker)) if start_idx != -1 else -1

                if start_idx != -1 and end_idx != -1:
                    extracted_code = output[start_idx + len(start_marker):end_idx].strip()
                else:
                    extracted_code = output

                output_file = Path(reproduction_dir) / f"reproduce_{bug_id}_{context_num}_attempt{attempt + 1}.py"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_code)

                # Structural Check
                if not gen_flags.get("no_compilation", False):
                    is_correct, feedback = check_structural_correctness_fn(extracted_code)
                    
                    if not is_correct:
                        prompt_code = build_prompt_fn(
                            refined_bug_report,
                            context_content,
                            plan_content,
                            feedback=f"Code failed structural check:\n{feedback}\nPlease fix these issues."
                        )
                        continue

                # Relevance Check
                if not gen_flags.get("no_relevance", False):
                    relevance_check = check_relevance_fn(refined_bug_report, extracted_code)
                    if not relevance_check:
                        prompt_code = build_prompt_fn(
                            refined_bug_report,
                            context_content,
                            plan_content,
                            feedback="Generated code is not relevant to the bug report."
                        )
                        continue

                # Static Analysis
                if not gen_flags.get("no_static_analysis", False):
                    pylint_output = analyze_with_pylint_fn(extracted_code, output_file)
                    if pylint_output:
                        refactor_prompt = build_refactor_prompt_fn(
                            extracted_code,
                            "\n".join(pylint_output),
                            refined_bug_report
                        )
                        refactored_code = run_llm_refactor_fn(refactor_prompt)
                        if refactored_code:
                            extracted_code = refactored_code

                # Runtime Feedback
                if not gen_flags.get("no_runtime_feedback", False):
                    score, feedback = calculate_probability_of_reproduction_fn(extracted_code, refined_bug_report)
                    
                    if score > 0.7:
                        final_output_file = Path(reproduction_dir) / f"reproduce_{bug_id}.py"
                        with open(final_output_file, 'w', encoding='utf-8') as f:
                            f.write(extracted_code)
                        
                        return {
                            'status': 'success',
                            'context_idx': context_idx,
                            'context_num': context_num,
                            'output_file': str(final_output_file),
                            'confidence_score': score,
                            'attempt': attempt + 1,
                            'message': f"Valid reproduction found with score {score:.2f}"
                        }
                else:
                    final_output_file = Path(reproduction_dir) / f"reproduce_{bug_id}.py"
                    with open(final_output_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_code)
                    
                    return {
                        'status': 'success',
                        'context_idx': context_idx,
                        'context_num': context_num,
                        'output_file': str(final_output_file),
                        'attempt': attempt + 1,
                        'message': f"Code generated (no feedback mode)"
                    }

            return {
                'status': 'failed',
                'context_idx': context_idx,
                'context_num': context_num,
                'message': f"Failed to generate valid code after {max_attempts} attempts"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'context_idx': context_idx,
                'context_num': context_num if 'context_num' in locals() else 'unknown',
                'error': str(e),
                'message': f"Error during code generation: {e}"
            }
    
    def process_plans_parallel(self, context_data: List[Tuple[int, str, Any]], bug_report: str,
                              create_prompt_plan_fn: Callable,
                              query_ollama_fn: Callable, gen_flags: Dict, bug_id: str,
                              plan_dir: str) -> List[Dict[str, Any]]:
        """
        Process plan generation in parallel for multiple contexts.
        
        Args:
            context_data: List of tuples (context_idx, context_file, context_content)
            bug_report: The full bug report used to generate plans
            create_prompt_plan_fn: Function to create plan prompts
            query_ollama_fn: Function to query Ollama
            gen_flags: Generation flags
            bug_id: Bug ID
            plan_dir: Output directory for plans
        
        Returns:
            List of results from all processes
        """
        logger.info(f"Starting parallel plan generation with {len(context_data)} contexts")
        
        # Prepare arguments for each worker
        args_list = [
            (idx, ctx_file, ctx_content, bug_report, create_prompt_plan_fn, query_ollama_fn, 
             gen_flags, bug_id, plan_dir)
            for idx, (ctx_file, ctx_content) in enumerate(context_data)
        ]
        
        results = []
        with Pool(self.num_workers) as pool:
            for result in pool.imap_unordered(self.process_plan_generation, args_list):
                results.append(result)
                if result['status'] == 'success':
                    logger.success(result['message'])
                else:
                    logger.error(result['message'])
        
        logger.info(f"Parallel plan generation completed: {len([r for r in results if r['status'] == 'success'])}/{len(results)} successful")
        return results
    
    def process_code_parallel(self, context_data: List[Tuple], refined_bug_report: str,
                             bug_id: str, gen_flags: Dict, max_attempts: int,
                             query_ollama_fn: Callable, build_prompt_fn: Callable,
                             check_structural_correctness_fn: Callable, check_relevance_fn: Callable,
                             analyze_with_pylint_fn: Callable, build_refactor_prompt_fn: Callable,
                             run_llm_refactor_fn: Callable, calculate_probability_of_reproduction_fn: Callable,
                             reproduction_dir: str) -> List[Dict[str, Any]]:
        """
        Process code generation in parallel for multiple contexts.
        
        Args:
            context_data: List of tuples with context information
            refined_bug_report: Refined bug report content
            bug_id: Bug ID
            gen_flags: Generation flags
            max_attempts: Maximum attempts per context
            query_ollama_fn: Function to query Ollama
            build_prompt_fn: Function to build prompts
            check_structural_correctness_fn: Function for structural checks
            check_relevance_fn: Function for relevance checks
            analyze_with_pylint_fn: Function for pylint analysis
            build_refactor_prompt_fn: Function to build refactor prompts
            run_llm_refactor_fn: Function to run LLM refactoring
            calculate_probability_of_reproduction_fn: Function to calculate reproduction probability
            reproduction_dir: Output directory for reproduction code
        
        Returns:
            List of results from all processes
        """
        logger.info(f"Starting parallel code generation with {len(context_data)} contexts")
        
        # Prepare arguments for each worker
        args_list = [
            (idx, ctx_file, ctx_content, refined_bug_report, plan_content, bug_id,
             gen_flags, max_attempts, query_ollama_fn, build_prompt_fn,
             check_structural_correctness_fn, check_relevance_fn, analyze_with_pylint_fn,
             build_refactor_prompt_fn, run_llm_refactor_fn, calculate_probability_of_reproduction_fn,
             reproduction_dir)
            for idx, (ctx_file, ctx_content, plan_content) in enumerate(context_data)
        ]
        
        results = []
        with Pool(self.num_workers) as pool:
            for result in pool.imap_unordered(self.process_code_generation, args_list):
                results.append(result)
                if result['status'] == 'success':
                    logger.info(result['message'])
                else:
                    logger.warning(result['message'])
        
        logger.info(f"Parallel code generation completed: {len([r for r in results if r['status'] == 'success'])}/{len(results)} successful")
        return results


def get_cpu_count() -> int:
    """Get the number of available CPUs."""
    return cpu_count()
