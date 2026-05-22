import os
import sys
import json
import logging
import importlib
import importlib.util
import time
import asyncio
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Any, Set
from .pipeline_base import PipelineBase
logger = logging.getLogger(__name__)
_script_dir = os.path.dirname(os.path.abspath(__file__))

@dataclass
class PipelineMetadata:
    """Metadata describing a backend."""
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

class PipelineManager:
    def __init__(self, config_path: Optional[str] = None, pipelines_dir: Optional[str] = None, config_lock: Optional[asyncio.Lock] = None):
        # Path resolution
        if pipelines_dir:
            self._pipelines_dir = Path(pipelines_dir).resolve()
        else:
            # Fallback for package structure
            self._pipelines_dir = Path(os.path.join(os.path.dirname(_script_dir), "Pipeline")).resolve()        
        self._pipelines: Dict[str, Type[PipelineBase]] = {}
        self._metadata: Dict[str, PipelineMetadata] = {}
        self._pipeline_paths: Dict[str, Path] = {} # New: store paths for lazy loading
        self._discovered = False
        self._errors: Dict[str, Dict[str, Any]] = {}
        self.config_path = os.path.abspath(config_path) if config_path else None
        self.config_lock = config_lock or asyncio.Lock()


    async def record_error(self, name: str, error: Any, context: str):
        """Record an error for a pipeline and persist it."""
        error_msg = str(error)
        logger.error(f"Pipeline error [{name}] in {context}: {error_msg}")
        self._errors[name.lower()] = {
            "error": error_msg,
            "timestamp": datetime.now().isoformat() if 'datetime' in globals() else time.time(),
            "context": context
        }
        await self.save_errors()


    async def save_errors(self):
        """Persist errors to config.json."""
        if not self.config_path or not os.path.exists(self.config_path):
            return
        try:
            async with self.config_lock:
                def _update():
                    with open(self.config_path, 'r') as f:
                        config = json.load(f)
                    config["pipeline_errors"] = self._errors
                    with open(self.config_path, 'w') as f:
                        json.dump(config, f, indent=4)
                await asyncio.to_thread(_update)
        except Exception as e:
            logger.error(f"Failed to save pipeline errors: {e}")

    def _load_metadata_from_manifest(self, manifest_path: Path, fallback_name: str) -> PipelineMetadata:
        """Helper to read manifest without loading class."""
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        return PipelineMetadata(
            name=data.get('name', fallback_name),
            version=data.get('version', '1.0.0'),
            description=data.get('description', ''),
            author=data.get('author', ''),
            requirements=data.get('requirements', [])
        )
    

    async def discover(self) -> Dict[str, PipelineMetadata]:
        """
        Scan pipelines directory and identify available pipelines without fully loading them.
        Uses manifest.json if available for metadata.
        Directory structure: Pipeline/{publisher}/{plugin}/main.py
        Keys use format: {publisher}/{plugin} to prevent naming conflicts.
        """
        if not self._pipelines_dir.exists():
            logger.warning(f"Pipelines directory not found: {self._pipelines_dir}")
            return {}
        logger.info(f"Discovering pipelines in: {self._pipelines_dir}")
        # Iterate through publisher directories
        for publisher_dir in self._pipelines_dir.iterdir():
            if not publisher_dir.is_dir():
                continue
            # Skip directories starting with underscore or dot
            if publisher_dir.name.startswith(('_', '.')):
                continue
            publisher_name = publisher_dir.name
            # Iterate through plugin directories within publisher
            for plugin_dir in publisher_dir.iterdir():
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
                        await self.record_error(f"{publisher_name}/{plugin_name}", e, "read_manifest")
                # 2. If no manifest, provide minimal metadata until loaded
                if metadata is None:
                    metadata = PipelineMetadata(
                        name=plugin_name,
                        version="1.0.0",
                        description=f"Lazy-loaded pipeline: {publisher_name}/{plugin_name}",
                        author=publisher_name
                    )
                # Discovery key uses publisher/plugin format to prevent conflicts
                pid = f"{publisher_name}/{plugin_name}".lower()
                self._pipeline_paths[pid] = plugin_dir
                self._metadata[pid] = metadata
        logger.info(f"Discovered {len(self._pipeline_paths)} potential pipelines.")
        self._discovered = True
        return self._metadata.copy()
    
    def get_pipeline(self, pipeline_name):
        if pipeline_name in self._pipelines:
            return self._pipelines[pipeline_name]
        else:
            return None
        

    async def _load_pipeline_by_name(self, name: str):
        """Perform actual module load and class extraction.
        Args:
            name: Pipeline key in format 'publisher/plugin'
        """
        pipeline_dir = self._pipeline_paths.get(name.lower())
        if not pipeline_dir:
            raise ValueError(f"Pipeline '{name}' not found in discovery map.")
        # Extract publisher and plugin names from path
        plugin_name = pipeline_dir.name
        publisher_name = pipeline_dir.parent.name
        main_py = pipeline_dir / "main.py"
        manifest_path = pipeline_dir / "manifest.json"
        # Ensure plugin paths are in sys.path
        plugin_path = str(pipeline_dir.resolve())
        if plugin_path not in sys.path:
            logger.debug(f"Adding plugin path to sys.path: {plugin_path}")
            sys.path.insert(0, plugin_path)
        # Add project python directory
        project_dir = os.environ.get("OPENCHAD_UV_PROJECT_DIR")
        if project_dir:
            python_path = str(Path(project_dir).resolve())
            if python_path not in sys.path:
                logger.debug(f"Adding python path to sys.path: {python_path}")
                sys.path.insert(0, python_path)
        logger.info(f"Lazy-loading pipeline code: {publisher_name}/{plugin_name}")
        # Use publisher_plugin format for module name to prevent conflicts
        module_name = f"pipeline_{publisher_name}_{plugin_name}"
        # Load the module
        try:
            spec = importlib.util.spec_from_file_location(
                module_name, 
                main_py
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {main_py}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            # Find the pipeline class
            pipeline_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) 
                    and issubclass_safe(attr, PipelineBase)
                    and attr.__module__ == module.__name__
                ):
                    pipeline_class = attr
                    break
            if pipeline_class is None:
                raise ValueError(f"No valid pipeline class found in {main_py}")
            # Register in final maps using the discovery key (publisher:plugin)
            self._pipelines[name.lower()] = pipeline_class
            logger.info(f"Registered pipeline: {name}")
        except Exception as e:
            await self.record_error(name, e, "load_module")
            raise


    async def get_pipeline_class(self, name: str) -> Optional[Type[PipelineBase]]:
        """Get a pipeline class by name, loading it if necessary."""
        # 1. Try already loaded
        if name.lower() in self._pipelines:
            return self._pipelines[name.lower()]
        # 2. Try loading from discovery map
        search_name = name.lower()
        if search_name in self._pipeline_paths:
            try:
                await self._load_pipeline_by_name(search_name)
                # Return by original name or the normalized search name
                return self._pipelines.get(name.lower()) or self._pipelines.get(search_name)
            except Exception as e:
                logger.error(f"Failed to lazy-load pipeline {name}: {e}", exc_info=True)
                return None
        return None
    
    def get_metadata(self, name: str) -> Optional[PipelineMetadata]:
        """Get metadata for a pipeline."""
        return self._metadata.get(name) or self._metadata.get(name.lower())
    
    def list_pipelines(self) -> List[PipelineMetadata]:
        """List all registered or discovered pipelines."""
        return list(self._metadata.values())

    async def create_instance(self, name: str, **kwargs) -> Optional[PipelineBase]:
        """
        Create an instance of a pipeline, loading it if necessary.
        """
        try:
            pipeline_class = await self.get_pipeline_class(name)
            if pipeline_class is None:
                return None
            return pipeline_class(**kwargs)
        except Exception as e:
            await self.record_error(name, e, "instantiate")
            return None

    async def reload_pipeline(self, name: str) -> bool:
        """
        Reload a pipeline module.
        Args:
            name: Pipeline key in format 'publisher/plugin'
        """
        search_name = name.lower()
        if search_name not in self._pipeline_paths:
            return False
        pipeline_dir = self._pipeline_paths[search_name]
        # Extract publisher and plugin names from path
        plugin_name = pipeline_dir.name
        publisher_name = pipeline_dir.parent.name
        module_name = f"pipeline_{publisher_name}_{plugin_name}"
        try:
            if search_name in self._pipelines:
                del self._pipelines[search_name]
            if module_name in sys.modules:
                del sys.modules[module_name]
            await self._load_pipeline_by_name(search_name)
            return True
        except Exception as e:
            logger.error(f"Failed to reload {name}: {e}")
            return False

    def unload_pipeline(self, name: str) -> bool:
        """
        Unload a pipeline module from memory.
        Args:
            name: Pipeline key in format 'publisher/plugin'
        Returns:
            True if successful
        """
        search_name = name.lower()
        if search_name not in self._pipeline_paths:
            return False
        pipeline_dir = self._pipeline_paths[search_name]
        # Extract publisher and plugin names from path
        plugin_name = pipeline_dir.name
        publisher_name = pipeline_dir.parent.name
        module_name = f"pipeline_{publisher_name}_{plugin_name}"
        try:
            # Remove from pipelines registry
            if search_name in self._pipelines:
                del self._pipelines[search_name]
            # Remove from metadata
            if search_name in self._metadata:
                del self._metadata[search_name]
            # Remove from discovery paths
            if search_name in self._pipeline_paths:
                del self._pipeline_paths[search_name]
            # Remove from sys.modules
            if module_name in sys.modules:
                del sys.modules[module_name]
            logger.info(f"Unloaded pipeline: {name}")
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