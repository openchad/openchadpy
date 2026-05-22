"""
Base Backend Module
Provides the foundation for all backend implementations including:
- BackendMetadata dataclass for backend information
- BaseBackend abstract base class
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Set, List
@dataclass
class BackendMetadata:
    """Metadata describing a backend."""
    name: str
    version: str
    description: str
    capabilities: List[str]
    author: str = ""
    requirements: List[str] = field(default_factory=list)
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "capabilities": [cap for cap in self.capabilities],
            "author": self.author,
            "requirements": self.requirements
        }
    
class BaseBackend(ABC):
    """
    Abstract base class for all backends.
    All backend implementations must:
    1. Set a unique `backend` class attribute
    2. Implement the `metadata` property
    """
    # Must be overridden by subclasses
    backend: str = ""
    use_lock: bool = False
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} backend='{self.backend}'>"