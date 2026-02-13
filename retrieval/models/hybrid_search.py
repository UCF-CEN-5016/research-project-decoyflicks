"""
Hybrid search model combining BM25 and dense retrieval.

This module implements the `HybridSearchIndex` class, which combines sparse keyword search (BM25)
with dense semantic search (SentenceTransformers + Annoy) to retrieve relevant code chunks.
It also supports re-ranking using a CrossEncoder.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from annoy import AnnoyIndex
from transformers import AutoTokenizer
import torch
from sklearn.preprocessing import normalize
from ..core.utils import tokenize
import logging

# Suppress logs from transformers and sentence_transformers
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

class HybridSearchIndex:
    """
    Implements hybrid search using BM25 and Approximate Nearest Neighbors (Annoy).    
    """
    def __init__(
        self,
        embedding_model: str,
        reranker_model: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        config: Optional[Any] = None
    ):
        self.device = device
        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.cross_encoder = CrossEncoder(reranker_model, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(reranker_model)
        self.max_seq_length = 512
        self.bm25 = None
        self.embeddings = None
        self.code_chunks = None
        self.annoy_index = None
        self.config = config

    def build_index(self, code_chunks: List[dict], corpus: List[List[str]]) -> None:
        """
        Build the hybrid search index from code chunks.

        Computes BM25 frequencies and generates dense embeddings for all chunks.
        Builds the Annoy index for fast nearest neighbor search.

        Args:
            code_chunks: List of code snippet dictionaries.
            corpus: Tokenized corpus for BM25.
        """
        self.code_chunks = code_chunks
        self.bm25 = BM25Okapi(corpus)
        
        texts = [chunk['page_content'] for chunk in code_chunks]
        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        self.embeddings = normalize(self.embeddings.cpu().numpy())
        
        self._build_annoy_index()

    def _build_annoy_index(self) -> None:
        """
        Build Annoy index for approximate nearest neighbor search.
        
        Uses Angular distance metric.
        """
        dim = self.embeddings.shape[1]
        self.annoy_index = AnnoyIndex(dim, 'angular')
        for i, vec in enumerate(self.embeddings):
            self.annoy_index.add_item(i, vec)
        self.annoy_index.build(n_trees=50)

    def save_index(self, index_dir: Path) -> None:
        """
        Save the index components to disk.

        Saves embeddings (npy), code chunks (json), and the Annoy index (ann).

        Args:
            index_dir: Directory path to save the index.
        """
        index_dir.mkdir(exist_ok=True)
        
        with open(index_dir / "documents.json", 'w') as f:
            json.dump(self.code_chunks, f)
            
        np.save(index_dir / "embeddings.npy", self.embeddings)
        self.annoy_index.save(str(index_dir / "annoy_index.ann"))

    def load_index(self, index_dir: Path) -> None:
        """
        Load the index components from disk.

        Args:
            index_dir: Directory path containing the saved index files.
        """
        with open(index_dir / "documents.json", 'r') as f:
            self.code_chunks = json.load(f)
            
        self.embeddings = np.load(index_dir / "embeddings.npy")
        
        corpus = [tokenize(doc['page_content']) for doc in self.code_chunks]
        self.bm25 = BM25Okapi(corpus)
        
        self.annoy_index = AnnoyIndex(self.embeddings.shape[1], 'angular')
        self.annoy_index.load(str(index_dir / "annoy_index.ann"))

    def semantic_search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform semantic search using the Annoy index.

        Args:
            query_embedding: Embedding vector of the query.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of (indices, scores).
        """
        indices, distances = self.annoy_index.get_nns_by_vector(
            query_embedding.flatten(), top_k, include_distances=True
        )
        return np.array(indices), 1 - np.array(distances)

    def search(
        self,
        query: str,
        top_k: int = 200,
        alpha: float = 0.55,
        rerank_top_k: int = 20,
        ann_top_k: int = 200
    ) -> List[dict]:
        """
        Perform hybrid search query.

        Combines BM25 and Semantic search scores using the formula:
        score = (1 - alpha) * bm25_score + alpha * semantic_score

        Args:
            query: Search query string.
            top_k: Number of results to consider from each method (not fully used in logic but kept for interface).
            alpha: Weight for semantic search (0.0 to 1.0).
            rerank_top_k: Number of results to re-rank using CrossEncoder.
            ann_top_k: Number of results to retrieve from Annoy.

        Returns:
            List of unique code chunks sorted by relevance.
        """
        if not query or not isinstance(query, str):
            return []

        if self.config:
            alpha = self.config.ALPHA
            rerank_top_k = self.config.RERANK_TOP_K
        
        # Ablation: No BM25
        if self.config and self.config.AB_NO_BM25:
            alpha = 1.0  # Fully semantic
        
        # Ablation: No ANN
        if self.config and self.config.AB_NO_ANN:
            alpha = 0.0  # Fully sparse (BM25)

        query = query[:100000]  # Truncate very long queries
        
        # BM25 search
        query_tokens = tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(query_tokens))
        bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-6)

        # Semantic search
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        query_embedding = normalize(query_embedding.cpu().numpy().reshape(1, -1))
        ann_indices, ann_scores = self.semantic_search(query_embedding, ann_top_k)
        ann_indices = np.array(ann_indices, dtype=int)

        if len(ann_indices) == 0:
            return []

        # Combine scores
        combined_scores = (1 - alpha) * bm25_scores[ann_indices] + alpha * ann_scores
        combined_indices_sorted = ann_indices[np.argsort(combined_scores)[::-1]]
        top_combined_indices = combined_indices_sorted[:rerank_top_k]

        # Prepare for cross-encoder
        top_chunks = [self.code_chunks[i] for i in top_combined_indices]
        
        if self.config and self.config.AB_NO_RERANKER:
            return top_chunks
        
        # Tokenize with proper truncation
        features = self.tokenizer(
            [query]*len(top_chunks),
            [chunk['page_content'][:100000] for chunk in top_chunks],
            padding=True,
            truncation='longest_first',
            max_length=self.max_seq_length,
            return_tensors="pt"
        ).to(self.device)

        # Run cross-encoder
        with torch.no_grad():
            rerank_scores = self.cross_encoder.model(**features).logits.squeeze()
        
        # Sort by cross-encoder scores
        reranked_indices = np.argsort(rerank_scores.cpu().numpy())[::-1]
        return [top_chunks[i] for i in reranked_indices]