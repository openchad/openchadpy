"""
Base STT (Speech-to-Text) Module
Abstract base class for Speech Recognition backends.
"""
from abc import abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
from .base_backend import BaseBackend
class BaseSTT(BaseBackend):
    """
    Abstract base class for STT (Speech-to-Text) backends.
    Provides audio transcription capabilities.
    """
    @abstractmethod
    def transcription(
        self, 
        audio_path: str,
        **kwargs
    ) -> str:
        """
        Transcribe audio file to text.
        Args:
            audio_path: Path to audio file
        Returns:
            Transcribed text
        """
        pass

    @abstractmethod
    def transcription_from_bytes(
        self, 
        audio_bytes: bytes,
        **kwargs
    ) -> str:
        """
        Transcribe audio bytes to text.
        Args:
            audio_bytes: Raw audio bytes (16-bit PCM, 16kHz, mono)
        Returns:
            Transcribed text
        """
        pass

    def process_chunk(
        self, 
        audio_bytes: bytes,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process audio chunk for streaming recognition.
        Args:
            audio_bytes: Chunk of raw audio bytes
        Returns:
            Dict with partial results and metadata
        """
        raise NotImplementedError("Subclass may implement process_chunk for streaming")
    
    def finalize(
        self, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Finalize streaming recognition and get final result.
        Returns:
            Dict with final transcription result
        """
        raise NotImplementedError("Subclass may implement finalize for streaming")
    
    def get_audio_buffer(
        self, 
        **kwargs
    ) -> Optional[np.ndarray]:
        """Get accumulated audio buffer."""
        return None
    
    def get_full_transcript(
        self, 
        **kwargs
    ) -> str:
        """Get complete transcript of all processed audio."""
        return ""
    
    def clear_buffer(
        self, 
        **kwargs
    ):
        """Clear audio buffer."""
        pass
    
    def reset(
        self, 
        **kwargs
    ):
        """Reset processor state."""
        pass