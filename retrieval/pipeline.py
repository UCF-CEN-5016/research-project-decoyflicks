from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from .config import Config
from .core.code_indexer import CodeIndexer
from .core.module_analyzer import ModuleAnalyzer
from .core.training_code_detector import TrainingCodeDetector
from .core.dependency_analyzer import DependencyAnalyzer
from .core.utils import setup_logger, load_bug_report, save_json
import json

logger = setup_logger(__name__)

class RetrievalPipeline:
    def __init__(self, bug_id: str = "001", config: Config = None, ablation_config: Dict[str, Any] = None, retrieval_ablation_name: str = "full_system", generation_ablation_name: str = "all_steps"):
        self.config = config or Config(bug_id=bug_id, ablation_config=ablation_config, retrieval_ablation_name=retrieval_ablation_name, generation_ablation_name=generation_ablation_name)
        self.bug_id = bug_id
        self.code_indexer = CodeIndexer(self.config)
        self.module_analyzer = ModuleAnalyzer(self.config)
        self.training_detector = TrainingCodeDetector(self.config)
        self.dependency_analyzer = DependencyAnalyzer(self.config)

    def run_pipeline(self, bug_id: str) -> Optional[Dict[str, Any]]:
        """Run full pipeline for a given bug ID."""
        try:
            # 1. Setup paths
            bug_report_path = self.config.BUG_REPORTS_DIR / f"{bug_id}.txt"
            code_dir = self.config.CODE_DIR
            
            if not bug_report_path.exists():
                raise FileNotFoundError(f"Bug report not found: {bug_report_path}")
            if not code_dir.exists():
                raise FileNotFoundError(f"Code directory not found: {code_dir}")

            # 2. Index codebase
            logger.info(f"Indexing codebase for bug {bug_id}")
            hybrid_index = self.code_indexer.index_codebase(code_dir)

            # 3. Find relevant code
            logger.info(f"Finding relevant code for bug {bug_id}")
            bug_report = load_bug_report(bug_report_path)
            relevant_code = self.code_indexer.find_relevant_code(bug_report, hybrid_index)
            if not relevant_code:
                logger.warning(f"No relevant code found for bug {bug_id}")
                return None

            # 4. Analyze modules
            logger.info(f"Analyzing modules for bug {bug_id}")
            module_report = {
                'bug_report': bug_id,
                'modules': self.module_analyzer.analyze_modules(relevant_code)
            }
            # 5. Detect training code
            logger.info(f"Detecting training code for bug {bug_id}")
            training_report = self.training_detector.detect_training_code(module_report, bug_report_path)
  
            # # 6. Analyze dependencies
            logger.info(f"Analyzing dependencies for bug {bug_id}")
            dependency_report = self.dependency_analyzer.analyze_training_dependencies(training_report)

            # 7. Generate final context
            logger.info(f"Generating final context for bug {bug_id}")
            self.create_context_files(bug_id, module_report, dependency_report)
            return {"status": "success", "context_dir": str(self.config.CONTEXT_DIR_OUT)}
        except Exception as e:
            logger.error(f"Pipeline failed for bug {bug_id}: {str(e)}")
            raise

    def create_context_files(self, bug_id, module_report, dependency_report):
        bug_report = dependency_report.get('bug_report')

        # Create the context directory if it doesn't exist
        context_dir = self.config.CONTEXT_DIR_OUT

        dependencies = dependency_report.get('dependencies', [])
        if not dependencies:
            for i, module in enumerate(module_report['modules'], 1):
                context = {
                    "bug_report": bug_report,
                    "module": module,
                    "module_snippets": [],
                }

                # Find relevant module snippets for this file
                for file in module['files']:
                    context['module_snippets'].append({
                        "file": file['path'],
                        "snippets": file['snippets']
                    })

                # Create the output file path
                output_filename = f"{bug_id}_module_{i}.json"
                filename = context_dir / output_filename

                # Write to file
                with open(filename, 'w') as f:
                    json.dump(context, f, indent=2)

        else:
            for i, dep in enumerate(dependencies[:5], 1):
                training_file_path = self.config.CODE_DIR / dep['file']
                try:
                    with open(training_file_path, 'r') as f:
                        training_file_content = f.read()
                except FileNotFoundError:
                    print(f"Warning: File not found - {training_file_path}")
                    training_file_content = f"Content not available for {training_file_path}"
                
                # Create the context structure
                context = {
                    "bug_report": bug_report,
                    "rank": i,
                    "score": dep['score'],
                    "main_file": {
                        "path": dep['file'],
                        "content": training_file_content
                    },
                    "module_snippets": [],
                    "dependencies": dep['external_dependencies']
                }
                
                # Find relevant module snippets for this file
                for module in module_report['modules']:
                    for file in module['files']:
                        if file['path'] == dep['file']:
                            context['module_snippets'].append({
                                "file": file['path'],
                                "snippets": file['snippets']
                            })
                        else:
                            # Check if this module file is a dependency
                            for ext_dep in dep['external_dependencies']:
                                if ext_dep['module'].replace('/', '_') in file['path']:
                                    context['module_snippets'].append({
                                        "file": file['path'],
                                        "snippets": file['snippets']
                                    })
                
                # Create the output file path
                output_filename = f"{bug_id}_{i}.json"
                filename = context_dir / output_filename

                # Write to file
                with open(filename, 'w') as f:
                    json.dump(context, f, indent=2)