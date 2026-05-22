"""
Base Model Provider Module
Provides the foundation for all model provider implementations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from .settings import Settings

@dataclass
class ProviderMetadata:
    """Metadata describing a model provider."""
    name: str
    version: str
    description: str
    author: str = ""
    requirements: List[str] = field(default_factory=list)
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "requirements": self.requirements
        }

class BaseModelProvider(ABC):
    """
    Abstract base class for all model providers.
    A model provider is responsible for scanning and discovering models
    from a specific source (e.g., local files, Hugging Face, etc.).
    """
    # Must be overridden by subclasses
    provider_id: str = ""
    settings_manager : Optional["Settings"]
    def __init__(self):
        self.settings_manager = None

    @abstractmethod
    async def scan(self) -> List[Dict[str, Any]]:
        """
        Scan for available models.
        Returns:
            A list of dictionaries, each representing a discovered model.
            The dictionary should contain at least:
            - id: Unique identifier for the model
            - name: Human-readable name
            - backend: The backend to use for this model
            - model_type: Type of model (e.g., 'llm', 'embedding')
        """
        pass

    async def close(self):
        """
        Cleanup resources (e.g., stop watchers, close connections).
        Should be called when the provider is unloaded.
        """
        pass
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} provider_id='{self.provider_id}'>"
    
    @staticmethod
    def format_model_name(m_name):
        words = str(m_name).split('/')[-1].replace('-', ' ').replace(':', ' ').replace('_', ' ').split()
        result = []
        for word in words:
            letter_count = sum(c.isalpha() for c in word)
            if letter_count <= 3:
                result.append(word.upper())
            else:
                result.append(word.title())
        return ' '.join(result)
