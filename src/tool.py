import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval import RetrievalPipeline
import argparse
import os
import subprocess
import json
import logging
from typing import Tuple
import ast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
    args = parser.parse_args()
    
    logger.info(f"Processing bug ID: {args.bug_id}")
    logger.info(f"Using Retrieval Ablation: {args.retrieval_ablation}")
    logger.info(f"Using Generation Ablation: {args.generation_ablation}")

    # --- Setup pipeline and flags ---
    ret_ablation_name = args.retrieval_ablation
    gen_ablation_name = args.generation_ablation
    
    ret_ablation_dict = RETRIEVAL_ABLATION_CONFIGS.get(ret_ablation_name, {})
    gen_flags = GENERATION_ABLATION_MAP.get(gen_ablation_name, {})
    
    logger.info("Initializing Retrieval Pipeline")
    pipeline = RetrievalPipeline(
        bug_id=args.bug_id, 
        ablation_config=ret_ablation_dict, 
        retrieval_ablation_name=ret_ablation_name,
        generation_ablation_name=gen_ablation_name
    )
    
    result = pipeline.run_pipeline(args.bug_id)
    logger.info(f"Retrieval pipeline completed: {result}")
 
    # --- Get all paths from config ---
    config = pipeline.config
    logger.info(f"Reading context from: {config.CONTEXT_DIR_IN}")
    logger.info(f"Writing outputs to: {config.REPRODUCTION_DIR_OUT}")
    
    # Bug report handling
    bug_report_path = config.BUG_REPORTS_DIR / f"{args.bug_id}.txt"
    logger.info(f"Loading bug report from: {bug_report_path}")
    with open(bug_report_path, 'r', encoding='utf-8') as f:
        bug_report_content = f.read()
    
    refined_report_dir = config.REFINED_BUG_REPORT_DIR_OUT
    refined_report_path = refined_report_dir / f"{args.bug_id}.txt"
    
    if not gen_flags.get("no_refine", False):
        logger.info("Performing bug report refinement")
        prompt_refinement = create_prompt_refinement(bug_report_content)
        
        logger.info("Executing LLM for bug report refinement")
        cmd = ['ollama', 'run', 'qwen2.5:7b', prompt_refinement]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        if process.stderr:
            for line in process.stderr:
                logger.error(f"LLM error: {line.strip()}")
        accumulated_output = []
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            logger.debug(f"LLM output: {line}")
            accumulated_output.append(line)
     
        process.stdout.close()
        return_code = process.wait()
        
        logger.info(f"Saving refined bug report to: {refined_report_path}")
        with open(refined_report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(accumulated_output))
    else:
        logger.info("Skipping refinement - copying original bug report")
        with open(refined_report_path, 'w', encoding='utf-8') as f:
            f.write(bug_report_content)
        logger.info(f"Original bug report copied to: {refined_report_path}")
    
    # Plan generation
    context_dir = config.CONTEXT_DIR_IN 
    plan_dir = config.PLANS_DIR_OUT
    
    logger.info(f"Processing context files from: {context_dir}")
    for context_file in os.listdir(context_dir):
        context_path = os.path.join(context_dir, context_file)
        logger.info(f"Processing context file: {context_file}")
        
        with open(context_path, 'r', encoding='utf-8') as f:
            context_content = json.loads(f.read())
        
        prompt_plan = create_prompt_plan(bug_report_content, context_content)
        cmd = ['ollama', 'run', 'qwen2.5-coder:7b', prompt_plan]
        
        if not gen_flags.get("no_plan", False):
            logger.info("Executing LLM for plan generation")
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if process.stderr:
                for line in process.stderr:
                    logger.error(f"LLM error: {line.strip()}")
            output, _ = process.communicate()
        else:
            # Keep an empty string "No plan" for ablation
            output = "No plan"
    
        plan_path = plan_dir / f"plan_{context_file.split('.')[0]}.json"
        # Remove module from the path if it exists
        if 'module' in plan_path.name:
            plan_path = plan_path.with_name(plan_path.name.replace('module_', ''))
        logger.info(f"Saving plan to: {plan_path}")
        with open(plan_path, 'w', encoding='utf-8') as f:
            f.write(output)

    # Code generation
    logger.info("Starting code generation phase")
    refined_report_path = config.REFINED_BUG_REPORT_DIR_IN / f"{args.bug_id}.txt"
    with open(refined_report_path, 'r', encoding='utf-8') as f:
        refined_bug_report = f.read()
    
    context_files = sorted(config.CONTEXT_DIR_IN.glob("*.json"), key=lambda x: int(x.stem.split('_')[-1]))
    reproduction_dir = config.REPRODUCTION_DIR_OUT
    plan_dir = config.PLANS_DIR_IN 
    
    for context_file in context_files:
        context_num = context_file.stem.split('_')[-1]
        logger.info(f"Generating code for context {context_num}")
        
        with open(context_file, 'r', encoding='utf-8') as f:
            context_content = json.load(f)
        
        plan_file = plan_dir / f"plan_{args.bug_id}_{context_num}.json"
        with open(plan_file, 'r', encoding='utf-8') as f:
            plan_content = f.read()
        
        prompt_code = _build_prompt(
            refined_bug_report,
            context_content,
            plan_content
        )
        
        max_attempts = args.max_attempts
        for attempt in range(max_attempts):
            logger.info(f"Attempt {attempt + 1} for context {context_num}")
            
            cmd = ['ollama', 'run', 'qwen2.5-coder:7b']
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(input=prompt_code)
            
            output = stdout
            start_marker = "```python"
            end_marker = "```"
            start_idx = output.find(start_marker)
            end_idx = output.find(end_marker, start_idx + len(start_marker)) if start_idx != -1 else -1

            if start_idx != -1 and end_idx != -1:
                extracted_code = output[start_idx + len(start_marker):end_idx].strip()
            else:
                extracted_code = output
                logger.warning("No code block markers found in LLM output")

            output_file = reproduction_dir / f"reproduce_{args.bug_id}_{context_num}_attempt{attempt + 1}.py"
            logger.debug(f"Saving attempt code to: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(extracted_code)

            if not gen_flags.get("no_compilation", False):
                is_correct, feedback = check_structural_correctness(extracted_code)
                
                if not is_correct:
                    logger.warning(f"Structural issues found in attempt {attempt + 1}: {feedback}")
                    prompt_code = _build_prompt(
                        refined_bug_report,
                        context_content,
                        plan_content,
                        feedback=f"Code failed structural check:\n{feedback}\nPlease fix these issues."
                    )
                    continue
            
            if not gen_flags.get("no_relevance", False):
                relevance_check = check_relevance(refined_bug_report, extracted_code)
                if not relevance_check:
                    logger.warning("Generated code is not relevant to the bug report")
                    prompt_code = _build_prompt(
                        refined_bug_report,
                        context_content,
                        plan_content,
                        feedback="Generated code is not relevant to the bug report."
                    )
                    continue
            
            if not gen_flags.get("no_static_analysis", False):
                pylint_output = analyze_with_pylint(extracted_code, output_file)
                if pylint_output:
                    refactor_prompt = _build_refactor_prompt(
                        extracted_code,
                        "\n".join(pylint_output),
                        refined_bug_report
                    )
                    logger.info("Executing LLM for code refactoring and fixing the issues raised by static analysis")
                    refactored_code = _run_llm_refactor(refactor_prompt)
                    extracted_code = refactored_code
            
            if not gen_flags.get("no_runtime_feedback", False):
                score, feedback = calculate_probability_of_reproduction(extracted_code, refined_bug_report)
                logger.info(f"Reproduction score for attempt {attempt + 1}: {score:.2f}")
            
                if score > 0.7:
                    final_output_file = reproduction_dir / f"reproduce_{args.bug_id}.py"
                    with open(final_output_file, 'w', encoding='utf-8') as f:
                        f.write(extracted_code)
                    logger.info(f"Successfully generated reproduction code: {final_output_file}")
                    analysis = analyze_bug_reproduction(extracted_code, refined_bug_report)
                    logger.info(f"Bug reproduction analysis:\n{analysis}")
                    return extracted_code, analysis
                else:
                    prompt_code = _build_prompt(
                        refined_bug_report,
                        context_content,
                        plan_content,
                        feedback=f"Code failed to reproduce the bug with the confidence score {score:.2f}. Feedback: {feedback}"
                    )
            else:
                final_output_file = reproduction_dir / f"reproduce_{args.bug_id}.py"
                with open(final_output_file, 'w', encoding='utf-8') as f:
                    f.write(extracted_code)          
                return extracted_code, "No runtime analysis performed."
    else:
        logger.warning(f"Failed to generate valid code for context {context_num} after {max_attempts} attempts")

def analyze_with_pylint(code: str, file_path: Path):
    result = []
    try:
        from pylint.lint import Run
        from pylint.reporters.text import TextReporter
        from io import StringIO
        
        # Write code to temporary file
        file_path.write_text(code, encoding='utf-8')
        
        # Run pylint analysis
        pylint_output = StringIO()
        reporter = TextReporter(pylint_output)
        Run([str(file_path)], reporter=reporter, exit=False)
        raw_output = pylint_output.getvalue()
        # Split the raw output into lines
        lines = raw_output.splitlines()
        for line in lines:
            if "e0" in line.lower() and "import-error" not in line.lower():
                result.append(line.strip())
        return result
    except Exception as e:
        result = []
    return result


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
    
    cmd = ['ollama', 'run', 'qwen2.5-coder:7b', prompt]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output, _ = process.communicate()
    return output

def _build_refactor_prompt(code: str, issues: str, bug_report: str) -> str:
    """Build a prompt for code refactoring based on ` feedback."""
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
    """Execute the LLM to refactor code based on PyLint feedback and bug report.
    
    Args:
        prompt: The complete refactoring prompt with code and issues
        
    Returns:
        The refactored code extracted from LLM's response
    """
    
    cmd = ['ollama', 'run', 'qwen2.5-coder:7b']
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send the prompt and get response
    stdout, stderr = process.communicate(input=prompt)
    output = stdout.strip()
    
    # Extract code block if present
    start_marker = "```python"
    end_marker = "```"
    start_idx = output.find(start_marker)
    
    if start_idx != -1:
        end_idx = output.find(end_marker, start_idx + len(start_marker))
        if end_idx != -1:
            extracted_code = output[start_idx + len(start_marker):end_idx].strip()
            logger.debug("Extracted refactored code from code block")
            return extracted_code
    
    # Fallback: return entire output if no markers found
    logger.warning("No code block markers found in refactoring output")
    return output

def check_relevance(bug_report: str, code: str) -> bool:
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

    cmd = ['ollama', 'run', 'qwen2.5-coder:7b']
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(input=prompt)
    try:
        response = json.loads(stdout.strip())
        return response.get('relevance', '').lower() == 'yes'
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return 'yes' in stdout.lower()

def extract_json_content(text):
    # Find content between ```json and ``` markers
    import re
    pattern = r'```json\s*(.*?)\s*```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1)
    return None

def calculate_probability_of_reproduction(code: str, bug_report: str) -> Tuple[float, str]:   
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

    analysis_result = query_llm(code_analysis_prompt)
    analysis_result = extract_json_content(analysis_result)

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

    extraction_result = query_llm(symptom_extraction_prompt)
    extraction_result = extract_json_content(extraction_result)

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
    comparison_result = query_llm(COMPARISON_PROMPT)
    comparison_result = extract_json_content(comparison_result)
    try:
        comparison_result = json.loads(comparison_result)
        score = comparison_result.get('score', 0.0)
        feedback = comparison_result.get('feedback', "")
        logger.info(f"Reproduction score: {score}, Feedback: {feedback}")
        return float(score), feedback
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON response from LLM")

def query_llm(prompt: str) -> str:
    """Executes ollama query with deepseek-r1:7b"""
    process = subprocess.Popen(
        ['ollama', 'run', 'qwen2.5-coder:7b'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, _ = process.communicate(prompt)
    return stdout.strip()

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
        return True, ""  # Code is structurally correct
    except SyntaxError as e:
        # Capture syntax error details
        error_msg = f"Syntax Error: {e.msg}\nLine {e.lineno}: {e.text.strip()}"
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

    # Add feedback section if provided
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
    # Determine structure type based on keys in context
    if 'main_file' in context:  # Old structure
        file_paths = [context['main_file']['path']]
        file_contents = [context['main_file']['content']]
        dependencies = context.get('dependencies', [])
        dep_string = "\n\nImport Dependencies:\n" + json.dumps(dependencies, indent=2) if dependencies else ""
    elif 'module' in context:  # New structure
        file_paths = [file['path'] for file in context['module']['files']]
        file_contents = [snippet['code'] for file in context['module']['files'] for snippet in file['snippets']]
        dep_string = ""

    # Format file paths and contents for output
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
    main()