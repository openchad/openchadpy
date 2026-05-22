"""
Base Embedding Module
Abstract base class for Embedding backends.
"""
from abc import abstractmethod
from typing import List, Tuple, Optional, Union
import numpy as np
from .base_backend import BaseBackend 
class BaseEmbedding(BaseBackend):
    """
    Abstract base class for Embedding backends.
    Provides text embedding and reranking capabilities.
    """
    @property
    def dimension(self) -> int:
        """Return the embedding dimension. Override if known at runtime."""
        raise NotImplementedError("Subclass must implement dimension property")
    
    @abstractmethod
    def embed(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Embed texts into vectors.
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2 normalize embeddings
            batch_size: Batch size for processing
        Returns:
            numpy array of shape (n_texts, dimension)
        """
        pass
    
    @abstractmethod
    def create_embedding(
        self,
        texts: Union[str, List[str]],
        task: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Create embeddings with optional task-specific prefixes.
        Args:
            texts: Single text or list of texts
            task: Task type ('query', 'document', 'code', etc.)
            normalize: Whether to L2 normalize
            batch_size: Batch size for processing
        Returns:
            numpy array of embeddings
        """
        pass
    
    @abstractmethod
    def embed_query(
        self, 
        query: str,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Embed a query for retrieval.
        Args:
            query: Query text
            normalize: Whether to normalize
        Returns:
            1D numpy array of shape (dimension,)
        """
        pass           
    
    @abstractmethod
    def embed_documents(
        self, 
        documents: List[str],
        titles: Optional[List[str]] = None,
        normalize: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Embed documents for retrieval.
        Args:
            documents: List of document texts
            titles: Optional list of document titles
            normalize: Whether to normalize
        Returns:
            numpy array of shape (n_docs, dimension)
        """
        pass
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        query_task: Optional[str] = None,
        document_task: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[int, float, str]]:
        """
        Rerank documents by relevance to query.
        Args:
            query: Query text
            documents: List of documents to rerank
            top_k: Return only top k results
            query_task: Task prefix for query
            document_task: Task prefix for documents
        Returns:
            List of (index, score, text) tuples sorted by score descending
        """
        pass
