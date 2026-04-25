"""
LLM Query Cache Module

Provides caching functionality for LLM queries to avoid redundant API calls.
Identical queries return cached results, significantly reducing processing time.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMCache:
    """
    File-based cache for LLM queries.
    Uses SHA256 hash of queries as keys to store and retrieve responses.
    """
    
    def __init__(self, cache_dir: str = ".llm_cache"):
        """
        Initialize the LLM cache.
        
        Args:
            cache_dir: Directory to store cache files (default: .llm_cache)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_metadata_file = self.cache_dir / "cache_metadata.json"
        self._load_metadata()
        
    def _load_metadata(self) -> None:
        """Load cache metadata from file."""
        if self.cache_metadata_file.exists():
            try:
                with open(self.cache_metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache metadata: {e}. Starting fresh.")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata to file."""
        try:
            with open(self.cache_metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except IOError as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_query_hash(self, query: str, model: str) -> str:
        """
        Generate a unique hash for a query and model combination.
        
        Args:
            query: The LLM query/prompt
            model: The model name (e.g., 'qwen2.5:7b')
            
        Returns:
            SHA256 hash of the combined query and model
        """
        combined = f"{model}:::{query}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def get(self, query: str, model: str) -> Optional[str]:
        """
        Retrieve a cached response for a query.
        
        Args:
            query: The LLM query/prompt
            model: The model name
            
        Returns:
            Cached response if found, None otherwise
        """
        query_hash = self._get_query_hash(query, model)
        cache_file = self.cache_dir / f"{query_hash}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                logger.debug(f"Cache HIT for query hash {query_hash[:8]}... (model: {model})")
                return cache_data.get('response')
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read cache file {query_hash}: {e}")
            return None
    
    def put(self, query: str, model: str, response: str) -> bool:
        """
        Store a response in the cache.
        
        Args:
            query: The LLM query/prompt
            model: The model name
            response: The LLM response to cache
            
        Returns:
            True if successfully cached, False otherwise
        """
        query_hash = self._get_query_hash(query, model)
        cache_file = self.cache_dir / f"{query_hash}.json"
        
        try:
            cache_data = {
                'query_hash': query_hash,
                'model': model,
                'response': response,
                'cached_at': datetime.now().isoformat(),
                'query_length': len(query),
                'response_length': len(response)
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            # Update metadata
            if query_hash not in self.metadata:
                self.metadata[query_hash] = {
                    'model': model,
                    'created': datetime.now().isoformat(),
                    'hits': 0,
                    'query_length': len(query)
                }
            
            self.metadata[query_hash]['hits'] = self.metadata[query_hash].get('hits', 0) + 1
            self.metadata[query_hash]['last_accessed'] = datetime.now().isoformat()
            self._save_metadata()
            
            logger.debug(f"Cache MISS - stored response for query hash {query_hash[:8]}... (model: {model})")
            return True
            
        except IOError as e:
            logger.warning(f"Failed to write cache file: {e}")
            return False
    
    def clear(self) -> None:
        """Clear all cached responses."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.metadata = {}
            self._save_metadata()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_hits = sum(item.get('hits', 0) for item in self.metadata.values())
        total_entries = len(self.metadata)
        
        cache_size = 0
        if self.cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))
        
        return {
            'total_entries': total_entries,
            'total_hits': total_hits,
            'cache_size_bytes': cache_size,
            'cache_size_mb': cache_size / (1024 * 1024)
        }
    
    def print_stats(self) -> None:
        """Print cache statistics to logger."""
        stats = self.get_stats()
        logger.info("=" * 50)
        logger.info("LLM Cache Statistics")
        logger.info("=" * 50)
        logger.info(f"Total cache entries: {stats['total_entries']}")
        logger.info(f"Total cache hits: {stats['total_hits']}")
        logger.info(f"Cache size: {stats['cache_size_mb']:.2f} MB")
        logger.info("=" * 50)


# Global cache instance
_cache_instance: Optional[LLMCache] = None


def get_cache(cache_dir: str = ".llm_cache") -> LLMCache:
    """
    Get or create the global LLM cache instance.
    
    Args:
        cache_dir: Directory to store cache files
        
    Returns:
        LLMCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LLMCache(cache_dir)
    return _cache_instance


def init_cache(cache_dir: str = ".llm_cache") -> LLMCache:
    """
    Initialize the global LLM cache.
    
    Args:
        cache_dir: Directory to store cache files
        
    Returns:
        LLMCache instance
    """
    global _cache_instance
    _cache_instance = LLMCache(cache_dir)
    return _cache_instance


def clear_cache() -> None:
    """Clear the global LLM cache."""
    cache = get_cache()
    cache.clear()
