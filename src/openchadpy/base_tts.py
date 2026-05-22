"""
Base TTS (Text-to-Speech) Module
Abstract base class for Text-to-Speech backends.
"""
from abc import abstractmethod
from typing import Union, Generator, Optional
import numpy as np
from .base_backend import BaseBackend
class BaseTTS(BaseBackend):
    """
    Abstract base class for TTS (Text-to-Speech) backends.
    Provides text-to-speech synthesis capabilities.
    """
    @property
    def sampling_rate(self) -> int:
        """Return the audio sampling rate. Override in subclass."""
        return 24000
    
    @abstractmethod
    def speech(
        self, 
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Generator]:
        """
        Generate speech from text.
        Args:
            text: Text to synthesize
            voice: Voice name/id to use
            speed: Speaking speed multiplier
            stream: If True, yield audio chunks
        Returns:
            numpy array (float32) or Generator for streaming
        """
        pass
    
    def generate(
        self,
        text: str,
        **kwargs
    ) -> Union[np.ndarray, Generator]:
        """Alias for speech() to match common patterns."""
        return self.speech(text, **kwargs)