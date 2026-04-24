"""
Profiling Module

Provides profiling capabilities for the RepGen program using py-spy.
Generates flamegraphs to identify performance bottlenecks.
"""

import subprocess
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PySpyProfiler:
    """
    Manages profiling using py-spy to generate performance flamegraphs.
    """
    
    def __init__(self, output_dir: str = ".profiling"):
        """
        Initialize the profiler.
        
        Args:
            output_dir: Directory to store profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.is_py_spy_available = self._check_py_spy()
        
    def _check_py_spy(self) -> bool:
        """
        Check if py-spy is installed.
        
        Returns:
            True if py-spy is available, False otherwise
        """
        try:
            result = subprocess.run(['py-spy', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"py-spy detected: {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            logger.warning("py-spy not found. Install with: pip install py-spy")
            return False
        
        return False
    
    def profile_script(self, script_path: str, args: list, output_prefix: str = "profile") -> Optional[str]:
        """
        Profile a Python script using py-spy.
        
        Args:
            script_path: Path to the Python script to profile
            args: Command-line arguments to pass to the script
            output_prefix: Prefix for output files
        
        Returns:
            Path to the generated SVG flamegraph, or None if profiling failed
        """
        if not self.is_py_spy_available:
            logger.warning("py-spy is not available. Profiling skipped.")
            return None
        
        svg_file = self.output_dir / f"{output_prefix}.svg"
        
        try:
            logger.info(f"Profiling {script_path} with py-spy...")
            cmd = [
                'py-spy', 'record',
                '-o', str(svg_file),
                '--format=flamegraph',
                '--',
                sys.executable,
                script_path
            ] + args
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"py-spy profiling failed: {result.stderr}")
                return None
            
            logger.success(f"Flamegraph generated: {svg_file}")
            return str(svg_file)
        except Exception as e:
            logger.error(f"Profiling error: {e}")
            return None
    
    def _generate_flamegraph(self, profile_file: str, svg_file: str) -> bool:
        """
        Generate a flamegraph from a profiling file.
        
        Args:
            profile_file: Path to the .prof file from py-spy
            svg_file: Output path for the SVG flamegraph
        
        Returns:
            True if generation succeeded, False otherwise
        """
        logger.warning("_generate_flamegraph is deprecated; use profile_script directly.")
        return False
    
    def profile_function(self, func, *args, **kwargs):
        """
        Profile a single function call.
        
        Args:
            func: Function to profile
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        
        Returns:
            Result of the function call
        """
        if not self.is_py_spy_available:
            logger.warning("py-spy is not available. Running without profiling.")
            return func(*args, **kwargs)
        
        # For now, we'll just run the function normally
        # Full function-level profiling would require more complex setup
        logger.debug(f"Running {func.__name__} (profiling not available at function level)")
        return func(*args, **kwargs)
    
    def get_stats(self) -> dict:
        """
        Get profiling statistics.
        
        Returns:
            Dictionary with profiling statistics
        """
        profiles = list(self.output_dir.glob("*.prof"))
        svgs = list(self.output_dir.glob("*.svg"))
        
        return {
            'profile_files': len(profiles),
            'flamegraph_files': len(svgs),
            'output_directory': str(self.output_dir),
            'py_spy_available': self.is_py_spy_available
        }
    
    def print_stats(self) -> None:
        """Print profiling statistics."""
        stats = self.get_stats()
        logger.info("=" * 50)
        logger.info("Profiling Statistics")
        logger.info("=" * 50)
        logger.info(f"Profile files: {stats['profile_files']}")
        logger.info(f"Flamegraph files: {stats['flamegraph_files']}")
        logger.info(f"Output directory: {stats['output_directory']}")
        logger.info(f"py-spy available: {stats['py_spy_available']}")
        logger.info("=" * 50)


class ProfileContext:
    """Context manager for profiling a code block."""
    
    def __init__(self, name: str, profiler: PySpyProfiler):
        """
        Initialize the profile context.
        
        Args:
            name: Name of the code block being profiled
            profiler: PySpyProfiler instance
        """
        self.name = name
        self.profiler = profiler
        self.start_time = None
    
    def __enter__(self):
        """Enter the context."""
        import time
        self.start_time = time.time()
        logger.debug(f"Started profiling: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        import time
        elapsed = time.time() - self.start_time
        logger.info(f"Profiled '{self.name}': {elapsed:.2f} seconds")
        return False


# Global profiler instance
_profiler_instance: Optional[PySpyProfiler] = None


def get_profiler(output_dir: str = ".profiling") -> PySpyProfiler:
    """
    Get or create the global profiler instance.
    
    Args:
        output_dir: Directory to store profiling results
    
    Returns:
        PySpyProfiler instance
    """
    global _profiler_instance
    if _profiler_instance is None:
        _profiler_instance = PySpyProfiler(output_dir)
    return _profiler_instance


def init_profiler(output_dir: str = ".profiling") -> PySpyProfiler:
    """
    Initialize the global profiler.
    
    Args:
        output_dir: Directory to store profiling results
    
    Returns:
        PySpyProfiler instance
    """
    global _profiler_instance
    _profiler_instance = PySpyProfiler(output_dir)
    return _profiler_instance


def profile_context(name: str) -> ProfileContext:
    """
    Create a profiling context for timing a code block.
    
    Args:
        name: Name of the code block
    
    Returns:
        ProfileContext instance
    """
    profiler = get_profiler()
    return ProfileContext(name, profiler)
