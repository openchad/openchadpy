"""
Base LLM Module
Abstract base class for Language Model backends.
"""
from abc import abstractmethod
from typing import Dict, Generator, Union, List, Any, Optional
from .base_backend import BaseBackend
class BaseLM(BaseBackend):
    """
    Abstract base class for LLM backends.
    Provides text generation and chat completion capabilities.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict, Generator]:
        """
        Generate text completion based on a prompt.
        Args:
            prompt: The input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            stop: Stop sequences
            stream: Whether to stream the response
        Returns:
            Dict with completion or Generator for streaming
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict, Generator]:
        """
        Generate chat completion from messages.
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            stream: Whether to stream the response
        Returns:
            Dict with completion or Generator for streaming
        """
        pass
