"""
Backend Registry Module
Provides plugin discovery and registration for backend implementations.
Automatically discovers backends from the Backend/ directory.
"""
import os
import sys
import json
import logging
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Set
from .base_backend import BaseBackend, BackendMetadata 
logger = logging.getLogger(__name__)
class BackendRegistry:
    """
    Discovers and manages backend plugins from the Backend/ directory.
    Each backend must have:
    - main.py with a class inheriting from BaseBackend (or its subclasses)
    - Optionally: manifest.json with metadata
    Example usage:
        registry = BackendRegistry("../Backend")
        await registry.discover()
        # List all backends
        for meta in registry.list_backends():
            print(meta.name, meta.capabilities)
        # Create instance
        model = registry.create_instance("SentenceTransformers", model_path="...")
    """
    
    def __init__(self, backends_dir: str):
        """
        Initialize the registry.
        Args:
            backends_dir: Path to the Backend/ directory
        """
        if os.path.isabs(backends_dir):
             self._backends_dir = Path(backends_dir).resolve()
        else:
             self._backends_dir = Path(backends_dir).resolve()
        self._backends: Dict[str, Type[BaseBackend]] = {}
        self._metadata: Dict[str, BackendMetadata] = {}
        self._backend_paths: Dict[str, Path] = {} # New: store paths for lazy loading
        self._discovered = False
    
    async def discover(self) -> Dict[str, BackendMetadata]:
        """
        Scan backends directory and identify available backends without fully loading them.
        Uses manifest.json if available for metadata.
        Directory structure: Backend/{publisher}/{plugin}/main.py
        Keys use format: {publisher}/{plugin} to prevent naming conflicts.
        """
        if not self._backends_dir.exists():
            logger.warning(f"Backends directory not found: {self._backends_dir}")
            return {}
        logger.info(f"Discovering backends in: {self._backends_dir}")
        # Iterate through publisher directories
        for publisher_dir in self._backends_dir.iterdir():
            try:
                if not publisher_dir.is_dir():
                    continue
                # Skip directories starting with underscore or dot
                if publisher_dir.name.startswith(('_', '.')):
                    continue
                publisher_name = publisher_dir.name
                # Iterate through plugin directories within publisher
                for plugin_dir in publisher_dir.iterdir():
                    try:
                        if not plugin_dir.is_dir():
                            continue
                        # Skip directories starting with underscore or dot
                        if plugin_dir.name.startswith(('_', '.')):
                            continue
                        main_py = plugin_dir / "main.py"
                        manifest_path = plugin_dir / "manifest.json"
                        if not main_py.exists():
                            logger.debug(f"Skipping {publisher_name}/{plugin_dir.name}: no main.py found")
                            continue
                        plugin_name = plugin_dir.name
                        # 1. Try to get identity/metadata from manifest first (fast)
                        metadata = None
                        if manifest_path.exists():
                            try:
                                metadata = self._load_metadata_from_manifest(manifest_path, plugin_name)
                            except Exception as e:
                                logger.warning(f"Failed to read manifest for {publisher_name}/{plugin_name}: {e}")
                        # 2. If no manifest, provide minimal metadata until loaded
                        if metadata is None:
                            metadata = BackendMetadata(
                                name=plugin_name,
                                version="1.0.0",
                                description=f"Lazy-loaded backend: {publisher_name}/{plugin_name}",
                                capabilities=[],  
                                author=publisher_name
                            )
                        # Discovery key uses publisher/plugin format to prevent conflicts
                        bid = f"{publisher_name}/{plugin_name}".lower()
                        self._backend_paths[bid] = plugin_dir
                        self._metadata[bid] = metadata
                    except Exception as e:
                        logger.error(f"Error discovering plugin in {publisher_dir.name}: {e}")
            except Exception as e:
                logger.error(f"Error scanning publisher directory {publisher_dir.name}: {e}")
        self._discovered = True
        logger.info(f"Discovered {len(self._backend_paths)} potential backends.")
        return self._metadata.copy()
    
    def _load_metadata_from_manifest(self, manifest_path: Path, fallback_name: str) -> BackendMetadata:
        """Helper to read manifest without loading class."""
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        caps = []
        for cap_name in data.get('capabilities', []):
            try:
                caps.append(cap_name)
            except KeyError:
                logger.warning(f"Unknown capability: {cap_name}")
        return BackendMetadata(
            name=data.get('name', fallback_name),
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            capabilities=caps,
            author=data.get('author', ''),
            requirements=data.get('requirements', [])
        )
    
    async def _load_backend_by_name(self, name: str):
        """Perform actual module load and class extraction.
        Args:
            name: Backend key in format 'publisher/plugin'
        """
        backend_dir = self._backend_paths.get(name.lower())
        if not backend_dir:
            raise ValueError(f"Backend '{name}' not found in discovery map.")
        # Extract publisher and plugin names from path
        plugin_name = backend_dir.name
        publisher_name = backend_dir.parent.name
        main_py = backend_dir / "main.py"
        manifest_path = backend_dir / "manifest.json"
        # Ensure plugin paths are in sys.path
        plugin_path = str(backend_dir.resolve())
        if plugin_path not in sys.path:
            logger.debug(f"Adding plugin path to sys.path: {plugin_path}")
            sys.path.insert(0, plugin_path)
        # Add project python directory
        python_path = str(Path(os.environ.get("OPENCHAD_UV_PROJECT_DIR")).resolve()) #pyrefly: ignore
        if python_path not in sys.path:
            logger.debug(f"Adding python path to sys.path: {python_path}")
            sys.path.insert(0, python_path)
        logger.info(f"Lazy-loading backend code: {publisher_name}/{plugin_name}")
        # Use publisher_plugin format for module name to prevent conflicts
        module_name = f"backend_{publisher_name}_{plugin_name}"
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                module_name, 
                main_py
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {main_py}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            # Use a thread for exec_module as it might be slow, though here we want it to be part of the flow
            spec.loader.exec_module(module)
            # Find the backend class
            backend_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) 
                    and hasattr(attr, 'backend') 
                    and getattr(attr, 'backend', '') != ''
                    and attr.__module__ == module.__name__
                ):
                    backend_class = attr
                    break
            if backend_class is None:
                raise ValueError(f"No valid backend class found in {main_py}")
            # Extract class backend id
            actual_bid = getattr(backend_class, 'backend')
            # Update metadata if it was minimal or needs class verification
            metadata = self._get_metadata(backend_class, manifest_path, plugin_name)
            # Register in final maps using the discovery key (publisher/plugin)
            self._backends[name.lower()] = backend_class
            self._metadata[name.lower()] = metadata
            logger.info(f"Registered backend: {name} (class id: {actual_bid})")
        except Exception as e:
            logger.error(f"Failed to load backend module {module_name}: {e}", exc_info=True)
            # Clean up if partially loaded
            if module_name in sys.modules:
                del sys.modules[module_name]
            raise RuntimeError(f"Backend module loading failed: {e}") from e
    
    async def get_backend_class(self, name: str) -> Optional[Type[BaseBackend]]:
        """Get a backend class by name, loading it if necessary."""
        # 1. Try already loaded
        if name in self._backends:
            return self._backends[name]
        # 2. Try loading from discovery map
        search_name = name.lower()
        if search_name in self._backend_paths:
            try:
                await self._load_backend_by_name(search_name)
                # Return by original name or the normalized search name
                return self._backends.get(name) or self._backends.get(search_name)
            except Exception as e:
                logger.error(f"Failed to lazy-load backend {name}: {e}", exc_info=True)
                return None
        return None
    
    def get_metadata(self, name: str) -> Optional[BackendMetadata]:
        """Get metadata for a backend."""
        search_name = name.lower()
        return self._metadata.get(search_name) or self._metadata.get(name)
    
    def list_backends(self) -> List[BackendMetadata]:
        """List all registered or discovered backends."""
        return list(self._metadata.values())
    
    def get_backends_by_capability(self, capability: str) -> List[str]:
        """Get all backends that support a given capability."""
        result = []
        for name, meta in self._metadata.items():
            if capability in meta.capabilities:
                result.append(name)
        return list(set(result)) # Unique names
    
    async def create_instance(self, name: str, **kwargs) -> BaseBackend:
        """
        Create an instance of a backend, loading it if necessary.
        """
        backend_class = await self.get_backend_class(name)
        if backend_class is None:
            raise ValueError(f"Unknown or failed to load backend: {name}")
        return backend_class(**kwargs)
    
    def has_backend(self, name: str) -> bool:
        """Check if a backend is registered or discoverable."""
        search_name = name.lower()
        return search_name in self._backends or search_name in self._backend_paths or name in self._backends
    
    def _get_metadata(
        self, 
        backend_class: Type, 
        manifest_path: Path, 
        fallback_name: str
    ) -> BackendMetadata:
        """Extract metadata from class or manifest file."""
        # Try manifest.json first
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                # Parse capabilities
                caps = []
                for cap_name in data.get('capabilities', []):
                    try:
                        caps.append(cap_name)
                    except KeyError:
                        logger.warning(f"Unknown capability: {cap_name}")
                return BackendMetadata(
                    name=data.get('name', fallback_name),
                    version=data.get('version', '1.0.0'),
                    description=data.get('description', ''),
                    capabilities=caps,
                    author=data.get('author', ''),
                    requirements=data.get('requirements', [])
                )
            except Exception as e:
                logger.warning(f"Failed to read manifest: {e}")
        # Try class metadata property (if implemented)
        if hasattr(backend_class, 'metadata') and not callable(getattr(backend_class, 'metadata', None)):
            # It's a property, we'd need an instance - skip for now
            pass
        # Generate minimal metadata from class
        return BackendMetadata(
            name=getattr(backend_class, 'backend', fallback_name),
            version='1.0.0',
            description=f"Backend: {backend_class.__name__}",
            capabilities=[],
            author='',
            requirements=[]
        )    
    
    async def reload_backend(self, name: str) -> bool:
        """
        Reload a backend module.
        Args:
            name: Backend key in format 'publisher:plugin' or class backend id
        Returns:
            True if successful
        """
        search_name = name.lower()
        # Check if using discovery path key
        if search_name not in self._backend_paths:
            return False
        backend_dir = self._backend_paths[search_name]
        # Extract publisher and plugin names from path
        plugin_name = backend_dir.name
        publisher_name = backend_dir.parent.name
        module_name = f"backend_{publisher_name}_{plugin_name}"
        try:
            if search_name in self._backends:
                del self._backends[search_name]
            if search_name in self._metadata:
                del self._metadata[search_name]
            
            await self._load_backend_by_name(search_name)
            return True
        except Exception as e:
            logger.error(f"Failed to reload {name}: {e}")
            return False
    
    def unload_backend(self, name: str) -> bool:
        """
        Unload a backend module from memory.
        Args:
            name: Backend key in format 'publisher:plugin' or class backend id
        Returns:
            True if successful
        """
        search_name = name.lower()
        # Check if in discovery path
        if search_name not in self._backend_paths:
            return False
        backend_dir = self._backend_paths[search_name]
        # Extract publisher and plugin names from path
        plugin_name = backend_dir.name
        publisher_name = backend_dir.parent.name
        module_name = f"backend_{publisher_name}_{plugin_name}"
        try:
            # Remove from backends registry
            if search_name in self._backends:
                del self._backends[search_name]
            # Remove from metadata
            if search_name in self._metadata:
                del self._metadata[search_name]
            # Remove from discovery paths
            if search_name in self._backend_paths:
                del self._backend_paths[search_name]
            # Remove from sys.modules
            if module_name in sys.modules:
                del sys.modules[module_name]
            logger.info(f"Unloaded backend: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload {name}: {e}")
            return False

def issubclass_safe(cls: Type, parent: Type) -> bool:
    """Safe issubclass check that handles import issues."""
    try:
        return issubclass(cls, parent)
    except TypeError:
        return False
