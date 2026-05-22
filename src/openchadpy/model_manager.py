import os
import sys
import asyncio
import functools
import time
import gc
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, Generator, List, AsyncGenerator, Tuple, Callable
import logging
from enum import Enum
import numpy as np
import json
from datetime import datetime
from .backend_registry import BackendRegistry
from PIL import Image
from pathlib import Path
import psutil
from .vram_checker import check_vram
logger = logging.getLogger(__name__)

@dataclass
class LoadedModel:
    """Represents a loaded model with metadata."""
    model_id: str
    model: Any
    backend: str
    model_type: List[str]  # e.g. ["llm"], ["embedding", "reranker"]
    model_path: str
    name: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    filename: Optional[str] = None
    mmproj_path: Optional[str] = None
    pricing: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    auto_load: bool = False
    is_local: bool = False
    api_base: Optional[str] = None
    credential_key: Optional[str] = None
    

    def touch(self):
        """Update last_used timestamp."""
        self.last_used = time.time()

    @property
    def use_lock(self) -> bool:
        """Check if the model requires a lock."""
        return getattr(self.model, "use_lock", False)
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            "model_id": self.model_id,
            "id": self.model_id, # for available_models compatibility
            "backend": self.backend,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "name": self.name,
            "filename": self.filename,
            "mmproj_path": self.mmproj_path,
            "loaded_at": self.loaded_at,
            "last_used": self.last_used,
            "pricing": self.pricing,
            "auto_load": self.auto_load,
            "is_local": self.is_local,
            "api_base": self.api_base,
            "credential_key": self.credential_key
        }
        if self.params:
            d.update(self.params)
        return d
class ModelManager:  

    def __init__(self, config_path: Optional[str] = None, backends_dir: Optional[str] = None, config_lock: Optional[asyncio.Lock] = None):
        # Model registry: model_id -> LoadedModel
        self._models: Dict[str, LoadedModel] = {}
        # Default model IDs for each type: model_type -> model_id
        self._defaults: Dict[str, Optional[str]] = {}
        # Global lock for registry modifications
        self._registry_lock = asyncio.Lock()
        # Shared lock for config.json access
        self.config_lock = config_lock or asyncio.Lock()
        # Loading status registry: model_id -> status_dict
        self._loading_status: Dict[str, Dict[str, Any]] = {}
        # Config path for persistence
        self.config_path = os.path.abspath(config_path) if config_path else None
        # Initialize project root from environment or script location
        env_project_dir = os.environ.get("OPENCHAD_PROJECT_DIR")
        if env_project_dir:
            self.project_root = os.path.abspath(env_project_dir)
        else:
            # Fallback: openchad/src/openchad -> project_root
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            # Try to find project root by looking for 'Models' or 'Backend'
            current = _script_dir
            for _ in range(4): # Search up 4 levels
                if os.path.exists(os.path.join(current, "Models")) or os.path.exists(os.path.join(current, "Backend")):
                    break
                current = os.path.dirname(current)
            self.project_root = current
        if backends_dir:
            self._backends_dir = os.path.abspath(backends_dir)
        else:
            self._backends_dir = os.path.join(self.project_root, "Backend")
        self._backend_registry = BackendRegistry(self._backends_dir)
        self._backends_discovered = False

    def get_model_name(self, model_id: str) -> Optional[str]:
        """Get the name of a specific model."""
        model = self._models.get(model_id)
        if model and model.name:
            return model.name
        return None

    def get_pricing(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get pricing information for a specific model."""
        model = self._models.get(model_id)
        if model and model.pricing:
            return model.pricing
        return None

    def _check_resources(self) -> Tuple[bool, Optional[str]]:
        """Check if system has enough available memory (RAM and VRAM)."""
        try:
            # 1. Check RAM (require at least 500MB free)
            mem = psutil.virtual_memory()
            min_free_ram = 500 * 1024 * 1024  # 500MB
            if mem.available < min_free_ram:
                return False, f"System RAM critically low ({mem.available // (1024*1024)} MB available)"
            # 2. Check VRAM
            vram_ok, vram_error, free_vram_mb = check_vram()
            if not vram_ok:
                return False, vram_error
            if free_vram_mb > 0:
                logger.debug(f"VRAM check passed: {free_vram_mb:.0f} MB free")
            return True, None
        except Exception as e:
            logger.warning(f"Failed to check system resources: {e}")
            return True, None  # Fail open if check fails

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        """
        Resolve a path to an absolute path.
        Checks relative to CWD first, then relative to project root.
        If the path doesn't exist in either, returns it as is (might be a repo ID).
        """
        if not path or not isinstance(path, str):
            return path
        if os.path.isabs(path):
            return path
        # 1. Try relative to CWD
        if os.path.exists(path):
            return os.path.abspath(path)
        # 2. Try relative to project root
        root_relative = os.path.join(self.project_root, path)
        if os.path.exists(root_relative):
            return os.path.abspath(root_relative)
        # 3. Check if it's a directory (if path ends with separator)
        if path.endswith(os.sep) or path.endswith('/'):
            return os.path.abspath(os.path.join(self.project_root, path))
        return path
    async def get_model_from_config(self, model_id: str): 
        try:
            async with self.config_lock:
            
                def read_config():
                    with open(self.config_path, 'r') as f: # pyrefly: ignore
                        return json.load(f)
                config = await asyncio.to_thread(read_config)
                return config.get('available_models', {}).get(model_id, None)
        except Exception as e:
            logger.error(f"Error getting model from config: {e}")
            return None
    async def load_config(self):
        """
        Load models and set defaults from a JSON configuration file.
        """
        target_path = self.config_path
        if not target_path:
            logger.warning("No config path provided for load_config")
            return
        if not os.path.exists(target_path):
            logger.warning(f"Config file not found: {target_path}")
            return
        self.config_path = os.path.abspath(target_path)
        try:
            async with self.config_lock:
            
                def read_config():
                    with open(self.config_path, 'r') as f: # pyrefly: ignore
                        return json.load(f)
                config = await asyncio.to_thread(read_config)
            # 1. Unload models no longer in config
            models_config = config.get("models", {})
            current_model_ids = list(self._models.keys())
            for model_id in current_model_ids:
                if model_id not in models_config:
                    # Model exists in memory but not in config - unload it
                    logger.info(f"Unloading model '{model_id}' as it is no longer in configuration")
                    try:
                        # Use internal unload to avoid redundant config writes if possible
                        # but standard unload is safer and handled correctly
                        await self.unload(model_id)
                    except Exception as e:
                        logger.error(f"Error unloading stale model '{model_id}': {e}")
            # 2. (Re)Load models from config
            for model_id, params in models_config.items():
                try:
                    filename = params.get("filename")
                    backend = params.get("backend")
                    name = params.get("name", model_id)
                    model_type = params.get("model_type", "llm")
                    model_path = params.get("model_path")
                    # Extract other kwargs
                    kwargs = {k: v for k, v in params.items() if k not in ["backend", "model_type", "model_path", "filename", "name", "last_error"]}
                    # check if already loaded with same content to avoid redundant load
                    if self.is_loaded(model_id):
                        loaded = self._models[model_id]
                        # simplified check: if backend and path match, skip
                        if loaded.backend == backend and loaded.model_path == self._resolve_path(model_path):
                            logger.debug(f"Model '{model_id}' already loaded with same config, skipping")
                            continue
                    logger.info(f"Loading model '{model_id}' from config")
                    await self.load(
                        model_id=model_id,
                        filename=filename,
                        backend=backend,
                        name=name,
                        model_type=model_type,
                        model_path=model_path,
                        set_as_default=False, # We'll set defaults explicitly from the config
                        **kwargs
                    )
                except Exception as e:
                    logger.error(f"Failed to load model '{model_id}' from config: {e}")
            # 2. Set defaults
        
            defaults_config = config.get("defaults", {})
            for m_type, m_id in defaults_config.items():
                if self.is_loaded(m_id) or m_id in self._loading_status:
                    self._set_default_for_type(m_id, m_type)
                    logger.info(f"Set default for '{m_type}' to '{m_id}'")
                else:
                    logger.warning(f"Default model '{m_id}' for type '{m_type}' was not found in registry or loading status")
        except Exception as e:
            logger.error(f"Error loading config from {self.config_path}: {e}", exc_info=True)
    # =========================================================================
    # Backend Discovery
    # =========================================================================
    async def discover_backends(self) -> List[Dict[str, Any]]:
        """
        Discover and register all available plugin backends.
        Returns:
            List of backend metadata dictionaries
        """
        if not self._backends_discovered:
            await self._backend_registry.discover()
            self._backends_discovered = True
        return [meta.to_dict() for meta in self._backend_registry.list_backends()]

    def list_available_backends(self) -> List[str]:
        """
        List all available backend names (both core and plugin).
        Returns:
            List of backend names
        """
        # Core backends (now plugins)
        core = []
        # Plugin backends
        plugin = list(self._backend_registry._metadata.keys())
        return core + plugin

    def get_backends_by_capability(self, capability: str) -> List[str]:
        """
        Get backends that support a given capability.
        Args:
            capability: Capability name (e.g., 'embedding', 'llm')
        Returns:
            List of backend names
        """
        try:
            return self._backend_registry.get_backends_by_capability(capability)
        except KeyError:
            return []
    # =========================================================================
    # Model Registry Operations
    # =========================================================================

    def get_loading_status(self, model_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Args:
            model_id: Optional ID to get status for specific model.
        Returns:
            Status dictionary or list of all active loading statuses.
        """
        if model_id:
            status = self._loading_status.get(model_id)
            if status:
                return status
            if self.is_loaded(model_id):
                return {"status": "loaded", "model_id": model_id}
            return {"status": "not_loading", "model_id": model_id}
        return list(self._loading_status.values())
    async def load(
        self,
        model_id: str,
        backend: str,
        name: str,
        model_type: Union[str, List[str]],
        model_path: Optional[str],
        filename: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        set_as_default: bool = True,
        is_local: bool = False,
        **kwargs
    ) -> Dict[str, Any]: # pyrefly: ignore
        """
        Load a model with a unique ID. Returns immediately while loading in background.
        """        
        # Normalize model_type to List[str]
        model_types = [model_type] if isinstance(model_type, str) else model_type
        try:
            # If model_path is a directory and filename is provided, join them
            effective_path = model_path
            if model_path and filename:
                resolved_tmp = self._resolve_path(model_path)
                if (resolved_tmp and os.path.isdir(resolved_tmp)) or model_path.endswith(os.sep) or model_path.endswith('/'):
                    effective_path = os.path.join(model_path, filename)
            actual_path = self._resolve_path(effective_path)
            actual_mmproj_path = self._resolve_path(mmproj_path)
            if isinstance(actual_path, str) and not Path(actual_path).exists():
                logger.warning(f"Model path not found locally: {actual_path}")
            if isinstance(actual_mmproj_path, str) and not Path(actual_mmproj_path).exists():
                logger.warning(f"mmproj path not found locally: {actual_mmproj_path}")
            # 2. Create model instance based on backend (passing the actual local path)
            model = await self._create_model(
                backend=backend,
                model_path=actual_path,
                model_type=model_types,
                mmproj_path=actual_mmproj_path,
                is_local=is_local,
                **kwargs
            )
            # Create LoadedModel entry
            loaded = LoadedModel(
                model_id=model_id,
                model=model,
                backend=backend,
                model_type=model_types,
                model_path=actual_path, # pyrefly: ignore
                name=name,
                filename=filename,
                mmproj_path=mmproj_path,
                params=kwargs
            )
            # Phase 3: Final registration (Locked)
            async with self._registry_lock:
                # Add to registry
                self._models[model_id] = loaded
                # Set as default if requested
                if set_as_default:
                    self._set_default_for_type(model_id, model_types)
                # Update status to loaded (or remove)
                # We pop it here to avoid race conditions with get_loading_status
                # which checks both _loading_status and _models
                self._loading_status.pop(model_id, None)
            logger.info(f"Model '{model_id}' loaded successfully")
            # Clear any previous error on success and persist model configuration
            await self.save_config(model_id=model_id, loaded_model=loaded)
        except Exception as e:
            logger.error(f"Error loading model '{model_id}' : {e}", exc_info=True)
            # Handle error status (Locked)
            async with self._registry_lock:
                self._loading_status[model_id] = {
                    "model_id": model_id,
                    "status": "error",
                    "error": str(e),
                    "backend": backend,
                    "model_type": model_type,
                    "model_path": model_path,
                    "name": name
                }
            # Record error in config
            await self.save_config(
                model_id=model_id, 
                error=str(e),
                metadata={
                    "id": model_id,
                    "backend": backend,
                    "name": name,
                    "model_type": model_types,
                    "model_path": model_path,
                    "filename": filename,
                    "mmproj_path": mmproj_path,
                    **kwargs
                }
            )
    async def unload(self, model_id: str) -> Dict[str, Any]:
        """
        Unload a specific model by ID.
        Args:
            model_id: ID of the model to unload
        Returns:
            Dict with status and message
        """
        async with self._registry_lock:
            return await self._unload_internal(model_id)
    async def unload_all(self) -> Dict[str, Any]:
        """Unload all models."""
        async with self._registry_lock:
            model_ids = list(self._models.keys())
            results = {}
            for model_id in model_ids:
                results[model_id] = await self._unload_internal(model_id)
            return {"status": "success", "unloaded": results}
    async def _unload_internal(self, model_id: str) -> Dict[str, Any]:
        """Internal unload (must be called under registry lock)."""
        if model_id not in self._models:
            return {"status": "not_found", "message": f"Model '{model_id}' not found"}
        loaded = self._models.pop(model_id)
        # Clear default references
        for m_type, m_id in list(self._defaults.items()):
            if m_id == model_id:
                self._defaults[m_type] = None
        # Cleanup model
        if hasattr(loaded.model, "unload"):
            try:
                if asyncio.iscoroutinefunction(loaded.model.unload):
                    await loaded.model.unload()
                else:
                    await asyncio.to_thread(loaded.model.unload)
            except Exception as e:
                logger.error(f"Error calling unload on model '{model_id}': {e}")
        del loaded.model
        await asyncio.to_thread(gc.collect)
        # Remove from config
        await self.remove_from_config(model_id)
        logger.info(f"Model '{model_id}' unloaded")
        return {"status": "success", "message": f"Model '{model_id}' unloaded"}
    async def _has_last_error(self, model_id: str) -> bool:
        """Check if a model has a last_error in the config file."""
        if not self.config_path or not os.path.exists(self.config_path):
            logger.debug(f"Config path '{self.config_path}' not found, skipping last_error check")
            return False
        try:
            async with self.config_lock:
            
                def read_config():
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                config = await asyncio.to_thread(read_config)
            models_cfg = config.get("models", {})
            model_cfg = models_cfg.get(model_id, {})
            has_error = "last_error" in model_cfg
            if has_error:
                logger.info(f"Detected persistent error for model '{model_id}' in config.json")
            else:
                logger.debug(f"No persistent error for model '{model_id}' in config.json")
            return has_error
        except Exception as e:
            logger.error(f"Error checking last_error for model '{model_id}': {e}")
            return False
    async def _create_model(self,
        backend: str,
        model_path: Optional[str],
        model_type: Union[str, List[str]],
        mmproj_path: Optional[str] = None, 
        is_local: bool = False,
        **kwargs
        ) -> Any:
        """
        Create a model instance based on backend.
        Supports both core backends (llamacpp, mlx, etc.) and plugin backends
        discovered from the Backend/ directory.
        """
        # Normalize path for local models
        target_path = model_path
        if is_local:
            target_path = self._resolve_path(model_path)
        if self._backend_registry.has_backend(backend):
            try:
                backend_class = await self._backend_registry.get_backend_class(backend)
                if not backend_class:
                    raise ValueError(f"Backend class for '{backend}' could not be loaded (check logs for errors)")
                return await asyncio.to_thread(
                    functools.partial(
                        backend_class,
                        model_path=target_path,
                        mmproj_path=mmproj_path,
                        **kwargs
                    )
                )
            except Exception as e:
                logger.error(f"Failed to instantiate backend '{backend}': {e}", exc_info=True)
                raise RuntimeError(f"Backend '{backend}' instantiation failed: {e}") from e
        # 2. Core backends (now handled by plugin system)
        raise ValueError(f"Unknown backend: {backend}. Available: {self.list_available_backends()}")
    async def save_config(self, model_id: Optional[str] = None, error: Optional[str] = None, loaded_model: Optional[LoadedModel] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Save current configuration to disk, optionally recording an error for a model
        or adding/updating a loaded model's configuration.
        """
        if not self.config_path:
            return
    
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(x) for x in obj]
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            if hasattr(obj, "__str__"):
                return str(obj)
            return repr(obj)
        try:
            async with self.config_lock:
            
                def _update():
                    if not os.path.exists(self.config_path):
                        # Create if missing? Existing logic just returns, sticking to it.
                        return
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    if "models" not in config:
                        config["models"] = {}
                    # 1. If we have a newly loaded model, persist its full config
                    if loaded_model:
                        mid = loaded_model.model_id
                        model_data = _sanitize(loaded_model.to_dict())
                        # Preserve existing fields like 'provider' or 'auto_load' if they exist
                        if mid in config["models"]:
                            for k, v in config["models"][mid].items():
                                if k not in model_data:
                                    model_data[k] = v
                        else:
                            # Default to auto_load if new
                            model_data["auto_load"] = True
                        # Clean up serialization specific fields not needed in config.json
                        model_data.pop("loaded_at", None)
                        model_data.pop("last_used", None)
                        model_data.pop("model_id", None)
                        config["models"][mid] = model_data
                        # Clear any previous error
                        config["models"][mid].pop("last_error", None)
                        logger.info(f"Persisted model '{mid}' to configuration")
                    # 2. If we just have an error to report
                    elif model_id:
                        # Record error even if model not in config yet (newly attempted load failure)
                        if model_id not in config["models"]:
                            # Use provided metadata if available to populate the entry
                            if metadata:
                                config["models"][model_id] = _sanitize(metadata)
                            else:
                                config["models"][model_id] = {"id": model_id} # Minimal fallback
                        elif metadata:
                            # Sync metadata even for existing entries if provided
                            for k, v in metadata.items():
                                if k not in ["auto_load", "last_error"]:
                                    config["models"][model_id][k] = _sanitize(v)
                        if error:
                            config["models"][model_id]["last_error"] = {
                                "message": str(error),
                                "timestamp": datetime.now().isoformat()
                            }
                        else:
                            # Clear error if success (though usually called via loaded_model path)
                            config["models"][model_id].pop("last_error", None)
                    with open(self.config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=4)
                await asyncio.to_thread(_update)
        except Exception as e:
            logger.error(f"Failed to save config: {e}", exc_info=True)
    async def remove_from_config(self, model_id: str):
        """Remove a model from config.json."""
        if not self.config_path:
            return
        try:
            async with self.config_lock:
            
                def _update():
                    if not os.path.exists(self.config_path):
                        return
                    with open(self.config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    if "models" in config and model_id in config["models"]:
                        del config["models"][model_id]
                        logger.info(f"Removed model '{model_id}' from configuration")
                        with open(self.config_path, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=4)
                await asyncio.to_thread(_update)
        except Exception as e:
            logger.error(f"Failed to remove model '{model_id}' from config: {e}", exc_info=True)

    def _set_default_for_type(self, model_id: str, model_type: Union[str, List[str]]):
        """Set model as default for its type."""
        types = [model_type] if isinstance(model_type, str) else model_type
        for t in types:
            self._defaults[t] = model_id
    # =========================================================================
    # Model Access
    # =========================================================================

    def get(self, model_id: str) -> Optional[LoadedModel]:
        """
        Get a loaded model by ID, updates last_used timestamp.
        Args:
            model_id: ID of the model to get
        Returns:
            LoadedModel or None if not found
        """
        loaded = self._models.get(model_id)
        if loaded:
            loaded.touch()
        return loaded

    def get_model(self, model_id: str) -> Optional[Any]:
        """Get just the model instance by ID."""
        loaded = self.get(model_id)
        return loaded.model if loaded else None

    def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models with their metadata."""
        return [m.to_dict() for m in self._models.values()]

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded."""
        return model_id in self._models
    async def invoke(self, model_id: str, method: str, **kwargs) -> Any:
        """
        Asynchronously invoke a specific method on a loaded model.
        Handles locking and thread offloading if necessary.
        Args:
            model_id: ID of the model to invoke method on
            method: Name of the method to call
            **kwargs: Arguments to pass to the method
        Returns:
            Result of the method call
        """
        loaded = self.get(model_id)
        if not loaded:
            raise ValueError(f"Model '{model_id}' not included in registry")
        if not hasattr(loaded.model, method):
            raise AttributeError(f"Model '{model_id}' (backend: {loaded.backend}) has no method '{method}'")
        loaded.touch()
        func = getattr(loaded.model, method)
        try:
            # Handle locking and async nature
            if asyncio.iscoroutinefunction(func):
                if loaded.use_lock:
                    async with loaded.lock:
                        return await func(**kwargs)
                else:
                    return await func(**kwargs)
            else:
                # Wrap synchronous call in thread
                if loaded.use_lock:
                    async with loaded.lock:
                        return await asyncio.to_thread(functools.partial(func, **kwargs))
                else:
                    return await asyncio.to_thread(functools.partial(func, **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, f"invoke_{method}", e)
    # =========================================================================
    # Backward Compatible Convenience Methods
    # =========================================================================
    async def _handle_method_error(self, loaded: LoadedModel, method: str, error: Exception):
        """Helper to handle and record errors during method execution."""
        error_msg = str(error)
        logger.error(f"Error in model '{loaded.model_id}' method '{method}': {error_msg}", exc_info=True)
        # Record error in model state
        loaded.last_error = { # pyrefly: ignore
            "type": "method_error",
            "method": method,
            "message": error_msg,
            "timestamp": datetime.now().isoformat()
        }
        # Persist to config
        await self.save_config(model_id=loaded.model_id, error=f"Method '{method}' failed: {error_msg}")
        # Re-raise as RuntimeError to prevent crash but signal failure
        raise RuntimeError(f"Model '{loaded.model_id}' failed during '{method}': {error_msg}") from error
    async def generate_image(self, prompt: str, model_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate an image using specified or default image generator model.
        """
        loaded = self._get_or_default(model_id, "image_generator")
        loaded.touch()
        if not hasattr(loaded.model, "generate_image"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'generate_image'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.generate_image, prompt, **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.generate_image, prompt, **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "generate_image", e)
    async def generate_video(self, prompt: str, model_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate a video using specified or default video generator model.
        """
        loaded = self._get_or_default(model_id, "video_generator")
        loaded.touch()
        if not hasattr(loaded.model, "generate_video"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'generate_video'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.generate_video, prompt, **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.generate_video, prompt, **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "generate_video", e)
    async def transcription(self, audio: Union[str, bytes], model_id: Optional[str] = None, **kwargs) -> str:
        """
        Transcribe audio using specified or default STT model.
        """
        loaded = self._get_or_default(model_id, "stt")
        loaded.touch()
        if not hasattr(loaded.model, "transcription"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'transcription'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.transcription, audio, **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.transcription, audio, **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "transcription", e)
    async def process_audio_chunk(self, audio_bytes: bytes, model_id: Optional[str] = None, stream_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Process audio chunk for streaming recognition.
        """
        loaded = self._get_or_default(model_id, "stt")
        loaded.touch()
        if not hasattr(loaded.model, "process_chunk"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'process_chunk'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.process_chunk, audio_bytes, stream_id if stream_id else "chunk", **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.process_chunk, audio_bytes, stream_id if stream_id else "chunk", **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "process_chunk", e)
    async def finalize_audio(self, model_id: Optional[str] = None, stream_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Finalize streaming recognition.
        """
        loaded = self._get_or_default(model_id, "stt")
        loaded.touch()
        if not hasattr(loaded.model, "finalize"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'finalize'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.finalize, stream_id if stream_id else "chunk", **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.finalize, stream_id if stream_id else "chunk", **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "finalize", e)
    async def get_audio_buffer(self, model_id: Optional[str] = None, stream_id: Optional[str] = None, **kwargs) -> bytes:
        """
        Get audio buffer for streaming recognition.
        """
        loaded = self._get_or_default(model_id, "stt")
        loaded.touch()
        if not hasattr(loaded.model, "get_audio_buffer"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'get_audio_buffer'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.get_audio_buffer, stream_id if stream_id else "chunk", **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.get_audio_buffer, stream_id if stream_id else "chunk", **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "get_audio_buffer", e)
    async def get_full_transcript(self, model_id: Optional[str] = None, stream_id: Optional[str] = None, **kwargs) -> str:
        """Get complete transcript of all finalized results"""
        loaded = self._get_or_default(model_id, "stt")
        loaded.touch()
        if not hasattr(loaded.model, "get_full_transcript"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'get_full_transcript'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.get_full_transcript, stream_id if stream_id else "chunk", **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.get_full_transcript, stream_id if stream_id else "chunk", **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "get_full_transcript", e)
    async def clear_audio_buffer(self, model_id: Optional[str] = None, stream_id: Optional[str] = None, **kwargs):
        """Clear audio buffer"""
        loaded = self._get_or_default(model_id, "stt")
        loaded.touch()
        if not hasattr(loaded.model, "clear_buffer"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'clear_buffer'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.clear_buffer, stream_id if stream_id else "chunk", **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.clear_buffer, stream_id if stream_id else "chunk", **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "clear_buffer", e)
    async def reset_audio(self, model_id: Optional[str] = None, stream_id: Optional[str] = None, **kwargs):
        """Reset model"""
        loaded = self._get_or_default(model_id, "stt")
        loaded.touch()
        if not hasattr(loaded.model, "reset"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'reset'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.reset, stream_id if stream_id else "chunk", **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.reset, stream_id if stream_id else "chunk", **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "reset", e)
    async def reset_all_audio(self, model_id: Optional[str] = None, **kwargs):
        """Reset model"""
        loaded = self._get_or_default(model_id, "stt")
        loaded.touch()
        if not hasattr(loaded.model, "reset"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'reset'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(functools.partial(loaded.model.reset_all, **kwargs))
            return await asyncio.to_thread(functools.partial(loaded.model.reset_all, **kwargs))
        except Exception as e:
            await self._handle_method_error(loaded, "reset_all", e)
    async def cleanup_stale_streams(self, model_id: Optional[str] = None, model_type: str = "stt", **kwargs) -> list:
        """
        Cleanup all stale streams for a model.
        Args:
            model_id: Optional specific model ID
            model_type: Type of model (default: "stt" for audio models)
        Returns:
            List of cleaned stream IDs
        """
        loaded = self._get_or_default(model_id, model_type)
        loaded.touch()
        if not hasattr(loaded.model, "cleanup_stale_streams"):
            logger.warning(f"Model '{loaded.model_id}' does not support 'cleanup_stale_streams'.")
            return []
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(loaded.model.cleanup_stale_streams, **kwargs)
            return await asyncio.to_thread(loaded.model.cleanup_stale_streams, **kwargs)
        except Exception as e:
            await self._handle_method_error(loaded, "cleanup_stale_streams", e)
    async def speech(self, text: str, stream: bool = False, model_id: Optional[str] = None, **kwargs) -> Union[np.ndarray, Generator]:
        """
        Generate speech using specified or default TTS model.
        """
        loaded = self._get_or_default(model_id, "tts")
        loaded.touch()
        if not hasattr(loaded.model, "speech"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'speech'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(loaded.model.speech, text, stream=stream, **kwargs)
            return await asyncio.to_thread(loaded.model.speech, text, stream=stream, **kwargs)
        except Exception as e:
            await self._handle_method_error(loaded, "speech", e)
    async def embed(
        self,
        texts: Union[str, List[str]],
        model_id: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Embed texts using specified or default embedding model.
        """
        loaded = self._get_or_default(model_id, "embedding")
        loaded.touch()
        if not hasattr(loaded.model, "embed"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'embed'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(loaded.model.embed, texts, normalize=normalize, batch_size=batch_size, **kwargs)
            return await asyncio.to_thread(loaded.model.embed, texts, normalize=normalize, batch_size=batch_size, **kwargs)
        except Exception as e:
            await self._handle_method_error(loaded, "embed", e)
    async def create_embedding(
        self,
        texts: Union[str, List[str]],
        task: Optional[str] = None,
        model_id: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Create an embedding for the given text using the specified or default embedding model.
        """
        loaded = self._get_or_default(model_id, "embedding")
        loaded.touch()
        if not hasattr(loaded.model, "create_embedding"):
            if hasattr(loaded.model, "embed"):
                # Fallback if create_embedding not explicitly there but embed is (generic)
                try:
                    if loaded.use_lock:
                        async with loaded.lock:
                            return await asyncio.to_thread(loaded.model.embed, texts, normalize=normalize, batch_size=batch_size, **kwargs)
                    return await asyncio.to_thread(loaded.model.embed, texts, normalize=normalize, batch_size=batch_size, **kwargs)
                except Exception as e:
                    await self._handle_method_error(loaded, "create_embedding_fallback", e)
            raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'create_embedding'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(loaded.model.create_embedding, texts, task=task, normalize=normalize, batch_size=batch_size, **kwargs)
            return await asyncio.to_thread(loaded.model.create_embedding, texts, task=task, normalize=normalize, batch_size=batch_size, **kwargs)
        except Exception as e:
            await self._handle_method_error(loaded, "create_embedding", e)
    async def embed_query(
        self, 
        query: str,
        model_id: Optional[str] = None, 
        normalize: bool = True
    ) -> np.ndarray:
        """Embed a query for retrieval."""
        loaded = self._get_or_default(model_id, "embedding")
        loaded.touch()
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(loaded.model.embed_query, query, normalize=normalize)
            return await asyncio.to_thread(loaded.model.embed_query, query, normalize=normalize)
        except Exception as e:
            await self._handle_method_error(loaded, "embed_query", e)
    async def embed_documents(
        self, 
        documents: List[str],
        model_id: Optional[str] = None,
        titles: Optional[List[str]] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """Embed documents for retrieval."""
        loaded = self._get_or_default(model_id, "embedding")
        loaded.touch()
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(loaded.model.embed_documents, documents, titles=titles, normalize=normalize)
            return await asyncio.to_thread(loaded.model.embed_documents, documents, titles=titles, normalize=normalize)
        except Exception as e:
            await self._handle_method_error(loaded, "embed_documents", e)
    async def rerank(
        self,
        query: str,
        documents: List[str],
        model_id: Optional[str] = None,
        top_k: Optional[int] = None,
        query_task: Optional[str] = None,
        document_task: Optional[str] = None,
    ) -> List[Tuple[int, float, str]]:
        """Rerank documents using specified or default reranker/embedding model."""
        loaded = self._get_or_default(model_id, "reranker")        
        loaded.touch()
        if not hasattr(loaded.model, "rerank"):
             raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'rerank'.")
        try:
            if loaded.use_lock:
                async with loaded.lock:
                    return await asyncio.to_thread(loaded.model.rerank, query, documents, top_k=top_k, query_task=query_task, document_task=document_task)
            return await asyncio.to_thread(loaded.model.rerank, query, documents, top_k=top_k, query_task=query_task, document_task=document_task)
        except Exception as e:
            await self._handle_method_error(loaded, "rerank", e)
    async def generate(
        self, 
        prompt: str, 
        model_id: Optional[str] = None,
        **kwargs
    ) -> Union[Dict, AsyncGenerator, str]:
        """Generate text using the specified or default LLM."""
        loaded = self._get_or_default(model_id, "llm")
        loaded.touch()
        stream = kwargs.get("stream", False)
        try:
            if not loaded.use_lock:
                return await asyncio.to_thread(loaded.model.generate, prompt=prompt, **kwargs)
            if stream:
                return self._stream_wrapper(loaded.model.generate, loaded.lock, prompt=prompt, **kwargs)
            else:
                if hasattr(loaded.model, "generate"):
                    async with loaded.lock:
                        return await asyncio.to_thread(loaded.model.generate, prompt=prompt, **kwargs)
                else:
                     raise NotImplementedError(f"Model '{loaded.model_id}' does not support 'generate'.")
        except Exception as e:
            await self._handle_method_error(loaded, "generate", e)
    async def _chat_internal(
        self, 
        messages: List[Dict[str, Any]], 
        model_type: str,
        model_id: Optional[str] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[Dict, AsyncGenerator, str]:
        """Internal helper for chat completion."""
        loaded = self._get_or_default(model_id, model_type)
        loaded.touch()
        try:
            if not loaded.use_lock:
                if stream:
                    return self._stream_wrapper(loaded.model.chat, messages=messages, stream=True, **kwargs)
                else:
                    return await asyncio.to_thread(loaded.model.chat, messages=messages, stream=False, **kwargs)
            if stream:
                return self._stream_wrapper(loaded.model.chat, loaded.lock, messages=messages, stream=True, **kwargs)
            else:
                async with loaded.lock:
                    return await asyncio.to_thread(loaded.model.chat, messages=messages, stream=False, **kwargs)
        except Exception as e:
            await self._handle_method_error(loaded, f"chat_{model_type}", e)

    def _has_image(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if any message contains an image."""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and (item.get("type") == "image_url" or "image" in item):
                        return True
            elif isinstance(content, dict):
                 if content.get("type") == "image_url" or "image" in content:
                     return True
        return False
    async def chat(
        self, 
        messages: List[Dict[str, Any]], 
        model_id: Optional[str] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[Dict, AsyncGenerator, str]:
        """Route chat to text_chat or vision_chat based on content."""
        if self._has_image(messages):
            return await self.vision_chat(messages, model_id=model_id, stream=stream, **kwargs)
        return await self.text_chat(messages, model_id=model_id, stream=stream, **kwargs)
    async def text_chat(
        self, 
        messages: List[Dict[str, Any]], 
        model_id: Optional[str] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[Dict, AsyncGenerator, str]:
        """Chat completion using the specified or default LLM."""
        return await self._chat_internal(messages, "llm", model_id=model_id, stream=stream, **kwargs)
    async def vision_chat(
        self, 
        messages: List[Dict[str, Any]], 
        model_id: Optional[str] = None,
        stream: bool = False, 
        **kwargs
    ) -> Union[Dict, AsyncGenerator, str]:
        """Chat completion using the specified or default VISION model."""
        return await self._chat_internal(messages, "vision", model_id=model_id, stream=stream, **kwargs)

    def get_default_id(self, model_type: str) -> Optional[str]:
        """Safely retrieve the default model ID for a given model type."""
        return self._defaults.get(model_type)

    def _get_or_default(
        self, 
        model_id: Optional[str], 
        model_type: str
    ) -> LoadedModel:
        """Get model by ID or fall back to default."""
        if model_id:
            loaded = self.get(model_id)
            if not loaded:
                raise ValueError(f"Model '{model_id}' not loaded")
            return loaded
        # Fallback to the provided default_id or lookup in defaults dict
        m_id = self.get_default_id(model_type)
        if m_id:
            loaded = self.get(m_id)
            if loaded:
                return loaded
        raise ValueError(f"No {model_type} model loaded")
    async def _stream_wrapper(self, func, lock: Optional[asyncio.Lock] = None, *args, **kwargs):
        """Wraps a sync generator in an async generator, optionally holding a lock."""
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue()
        sentinel = object()
        lock_released = False
        cancel_event = threading.Event()
        producer_done = threading.Event()
    
        def release_lock():
            nonlocal lock_released
            if lock and not lock_released:
                try:
                
                    def safe_release():
                        if lock.locked():
                            lock.release()
                    loop.call_soon_threadsafe(safe_release)
                    lock_released = True
                except Exception as e:
                    logger.error(f"Error releasing lock: {e}")
    
        def producer():
            try:
                gen = func(*args, **kwargs)
                for item in gen:
                    if cancel_event.is_set():
                        logger.info("Stream cancelled by consumer, stopping producer")
                        if hasattr(gen, 'close'):
                            try:
                                gen.close()
                            except Exception:
                                pass
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)
            except Exception as e:
                logger.error(f"Stream error: {e}")
                asyncio.run_coroutine_threadsafe(queue.put(e), loop)
                asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)
            finally:
                release_lock()
                producer_done.set()
        if lock:
            await lock.acquire()
        t = threading.Thread(target=producer, daemon=True)
        try:
            t.start()
        except Exception:
            if lock:
                lock.release()
            raise
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancel_event.set()
            producer_done.wait(timeout=2.0)
            if not lock_released and lock:
                try:
                    if lock.locked():
                        lock.release()
                    lock_released = True
                    logger.warning("Force released lock after consumer exit")
                except RuntimeError:
                    pass
