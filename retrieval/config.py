import os
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, 
                 bug_id: str, 
                 ablation_config: Dict[str, Any] = None, 
                 retrieval_ablation_name: str = "full_system", 
                 generation_ablation_name: str = "all_steps",
                 dataset_dir: str = None):
        
        self.BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
        
        # Use custom dataset_dir if provided (for ae_dataset), otherwise use default
        if dataset_dir:
            self.DATASET_DIR = Path(dataset_dir)
        else:
            self.DATASET_DIR = self.BASE_DIR / "dataset"
        
        self.PROJECT_DIR = self.DATASET_DIR / bug_id
        
        # --- Base paths for full_system (default) ---
        self.BASE_REFINED_BUG_REPORT_DIR = self.PROJECT_DIR / "refined_bug_report"
        self.BASE_CONTEXT_DIR = self.PROJECT_DIR / "context"
        self.BASE_PLANS_DIR = self.PROJECT_DIR / "plan"
        self.BASE_REPRODUCTION_DIR = self.PROJECT_DIR / "reproduction_code"
        
        # --- Base path for all ablation outputs ---
        self.ABLATION_OUTPUT_DIR = self.PROJECT_DIR / "ablations"

        # --- Base paths for shared data ---
        self.BUG_REPORTS_DIR = self.PROJECT_DIR / "bug_report"
        self.CODE_DIR = self.PROJECT_DIR / "code"

        # --- 1. Set INPUT Paths (Where to READ from) ---
        # A generation ablation (e.g., "no_plan") READS from the "full_system" context.
        # A retrieval ablation (e.g., "NO_BM25") READS from its *own* context.
        read_ablation_name = retrieval_ablation_name
        
        if read_ablation_name == "full_system":
            self.REFINED_BUG_REPORT_DIR_IN = self.BASE_REFINED_BUG_REPORT_DIR
            self.CONTEXT_DIR_IN = self.BASE_CONTEXT_DIR
            self.PLANS_DIR_IN = self.BASE_PLANS_DIR
        else:
            self.REFINED_BUG_REPORT_DIR_IN = self.ABLATION_OUTPUT_DIR / read_ablation_name / "refined_bug_report"
            self.CONTEXT_DIR_IN = self.ABLATION_OUTPUT_DIR / read_ablation_name / "context"
            self.PLANS_DIR_IN = self.ABLATION_OUTPUT_DIR / read_ablation_name / "plan"

        # --- 2. Set OUTPUT Paths (Where to WRITE to) ---
        # The output dir is the generation ablation name if it exists,
        # otherwise it's the retrieval ablation name.
        if generation_ablation_name != "all_steps":
            output_ablation_name = generation_ablation_name
        else:
            output_ablation_name = retrieval_ablation_name
            
        if output_ablation_name == "full_system":
            self.REFINED_BUG_REPORT_DIR_OUT = self.BASE_REFINED_BUG_REPORT_DIR
            self.CONTEXT_DIR_OUT = self.BASE_CONTEXT_DIR
            self.PLANS_DIR_OUT = self.BASE_PLANS_DIR
            self.REPRODUCTION_DIR_OUT = self.BASE_REPRODUCTION_DIR
        else:
            self.REFINED_BUG_REPORT_DIR_OUT = self.ABLATION_OUTPUT_DIR / output_ablation_name / "refined_bug_report"
            self.CONTEXT_DIR_OUT = self.ABLATION_OUTPUT_DIR / output_ablation_name / "context"
            self.PLANS_DIR_OUT = self.ABLATION_OUTPUT_DIR / output_ablation_name / "plan"
            self.REPRODUCTION_DIR_OUT = self.ABLATION_OUTPUT_DIR / output_ablation_name / "reproduction_code"

        # Model configurations
        self.EMBEDDING_MODEL = "flax-sentence-embeddings/st-codesearch-distilroberta-base"
        self.RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        
        # Search parameters
        self.SEARCH_TOP_K = 200
        self.RERANK_TOP_K = 20
        self.ALPHA = 0.55
        
        # --- Ablation Settings ---
        ablations = ablation_config or {}
        self.AB_NO_BM25 = ablations.get("NO_BM25", False)
        self.AB_NO_ANN = ablations.get("NO_ANN", False)
        self.AB_NO_RERANKER = ablations.get("NO_RERANKER", False)
        self.AB_NO_TRAINING_LOOP_EXTRACTION = ablations.get("NO_TRAINING_LOOP_EXTRACTION", False)
        self.AB_NO_TRAINING_LOOP_RANKING = ablations.get("NO_TRAINING_LOOP_RANKING", False)
        self.AB_NO_MODULE_PARTITIONING = ablations.get("NO_MODULE_PARTITIONING", False)
        self.AB_NO_DEPENDENCY_EXTRACTION = ablations.get("NO_DEPENDENCY_EXTRACTION", False)
        
        # Ensure directories exist
        self._setup_directories()
    
    def _setup_directories(self):
        # Shared dirs
        self.BUG_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.CODE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Base (full_system) output dirs
        self.BASE_REFINED_BUG_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        self.BASE_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
        self.BASE_PLANS_DIR.mkdir(parents=True, exist_ok=True)
        self.BASE_REPRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Ablation output dirs (if different from base)
        self.REFINED_BUG_REPORT_DIR_OUT.mkdir(parents=True, exist_ok=True)
        self.CONTEXT_DIR_OUT.mkdir(parents=True, exist_ok=True)
        self.PLANS_DIR_OUT.mkdir(parents=True, exist_ok=True)
        self.REPRODUCTION_DIR_OUT.mkdir(parents=True, exist_ok=True)