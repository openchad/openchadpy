"""
Model Provider Manager Module
Handles discovery, loading, and execution of model provider plugins.
"""
import os
import sys
import json
import logging
import importlib
import importlib.util
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, TYPE_CHECKING, Callable
from .base_provider import BaseModelProvider, ProviderMetadata
if TYPE_CHECKING:
    from .settings import Settings
logger = logging.getLogger(__name__)

class ModelProviderManager:
    """
    Manages model provider plugins.
    Discovers providers in the ModelProvider directory and handles scanning.
    """

    def __init__(self, settings_manager: Optional["Settings"] = None, providers_dir: Optional[str] = None, config_lock: Optional[asyncio.Lock] = None, on_change: Optional[Callable[[Optional[str]], Any]] = None):
        """
        Initialize the manager.
        Args:
            providers_dir: Path to the ModelProvider/ directory.
                          Defaults to OPENCHAD_MODEL_PROVIDERS_DIR or /home/h/fractalist/ModelProvider.
            config_lock: Shared lock for config.json access.
        """
        if providers_dir is None:
            providers_dir = os.environ.get("OPENCHAD_MODEL_PROVIDERS_DIR")
        self.on_change = on_change
        if providers_dir:
            self._providers_dir = Path(providers_dir).resolve()
        else:
            # Fallback for package structure: find ModelProvider relative to this script
            self._providers_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent / "ModelProvider"
            self._providers_dir = self._providers_dir.resolve()
        self._providers: Dict[str, BaseModelProvider] = {}
        self._metadata: Dict[str, ProviderMetadata] = {}
        self._provider_models: Dict[str, List[Dict[str, Any]]] = {}
        self._discovered = False
        self.config_lock = config_lock or asyncio.Lock()
        self.settings_manager = settings_manager

    def get_credential_sensitive_providers(self) -> List[str]:
        """Return provider IDs that declared rescan_on_credentials = True."""
        return [
            pid for pid, provider in self._providers.items()
            if getattr(provider, "rescan_on_credentials", False)
        ]

    async def discover_and_load(self) -> Dict[str, ProviderMetadata]:
        """
        Scan providers directory and load all available providers.
        Directory structure: ModelProvider/{publisher}/{plugin}/main.py
        """
        if not self._providers_dir.exists():
            logger.warning(f"ModelProvider directory not found: {self._providers_dir}")
            self._discovered = True
            return {}
        logger.info(f"Discovering model providers in: {self._providers_dir}")
        count = 0
        for publisher_dir in self._providers_dir.iterdir():
            if not publisher_dir.is_dir() or publisher_dir.name.startswith(('_', '.')):
                continue
            for plugin_dir in publisher_dir.iterdir():
                if not plugin_dir.is_dir() or plugin_dir.name.startswith(('_', '.')):
                    continue
                main_py = plugin_dir / "main.py"
                if not main_py.exists():
                    continue
                publisher_name = publisher_dir.name
                plugin_name = plugin_dir.name
                provider_id = f"{publisher_name}/{plugin_name}".lower()
                try:
                    await self._load_provider(provider_id, plugin_dir)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to load provider {provider_id}: {e}")
        self._discovered = True
        logger.info(f"Discovery complete. Loaded {count} model providers.")
        return self._metadata

    async def _load_provider(self, provider_id: str, plugin_dir: Path):
        """Load a single provider plugin."""
        main_py = plugin_dir / "main.py"
        manifest_path = plugin_dir / "manifest.json"
        # Ensure plugin paths are in sys.path
        plugin_path = str(plugin_dir.resolve())
        if plugin_path not in sys.path:
            logger.debug(f"Adding plugin path to sys.path: {plugin_path}")
            sys.path.insert(0, plugin_path)
        # Add project python directory
        python_path = str(Path(os.environ.get("OPENCHAD_UV_PROJECT_DIR")).resolve()) #pyrefly: ignore
        if python_path not in sys.path:
            logger.debug(f"Adding python path to sys.path: {python_path}")
            sys.path.insert(0, python_path)
        module_name = f"provider_{provider_id.replace('/', '_')}"
        # If module already exists, remove it to allow clean reload
        if module_name in sys.modules:
            del sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, main_py)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module from {main_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # Load metadata from manifest if available
        metadata = self._load_metadata_from_manifest(manifest_path, plugin_dir.name, plugin_dir.parent.name)
        spec.loader.exec_module(module)
        # Find the provider class
        provider_class = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type) 
                and issubclass(attr, BaseModelProvider)
                and attr is not BaseModelProvider
                and getattr(attr, 'provider_id', '') != ''
            ):
                provider_class = attr
                break
        if provider_class is None:
            raise ValueError(f"No valid BaseModelProvider class found in {main_py}")
        # Instantiate and register
        instance = provider_class()
        self._providers[provider_id] = instance
        self._metadata[provider_id] = metadata
        instance.settings_manager = self.settings_manager
        # Connect change notification if supported
        if hasattr(instance, 'on_change'):
            if self.on_change:
            
                def make_on_change(pid=provider_id):
                    if asyncio.iscoroutinefunction(self.on_change):
                    
                        async def async_wrapper():
                            return await self.on_change(pid)
                        return async_wrapper
                    else:
                    
                        def sync_wrapper():
                            return self.on_change(pid)
                        return sync_wrapper
                instance.on_change = make_on_change() #pyrefly: ignore
            else:
                instance.on_change = None #pyrefly: ignore
        logger.info(f"Loaded model provider: {provider_id}")

    async def reload_provider(self, provider_id: str, plugin_dir: Path):
        """Reload an existing provider."""
        logger.info(f"Reloading model provider: {provider_id}")
        if provider_id in self._providers:
            try:
                await self._providers[provider_id].close()
            except Exception as e:
                logger.error(f"Error closing provider {provider_id} during reload: {e}")
        await self._load_provider(provider_id, plugin_dir)

    async def unload_provider(self, provider_id: str):
        """Unload a provider."""
        if provider_id in self._providers:
            logger.info(f"Unloading model provider: {provider_id}")
            try:
                await self._providers[provider_id].close()
            except Exception as e:
                logger.error(f"Error closing provider {provider_id} during unload: {e}")
            del self._providers[provider_id]
            if provider_id in self._metadata:
                del self._metadata[provider_id]

    def _load_metadata_from_manifest(self, manifest_path: Path, plugin_name: str, publisher_name: str) -> ProviderMetadata:
        """Helper to read manifest."""
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    data = json.load(f)
                return ProviderMetadata(
                    name=data.get('name', plugin_name),
                    version=data.get('version', '1.0.0'),
                    description=data.get('description', ''),
                    author=data.get('author', publisher_name),
                    requirements=data.get('requirements', [])
                )
            except Exception as e:
                logger.warning(f"Failed to read manifest for {plugin_name}: {e}")
        return ProviderMetadata(
            name=plugin_name,
            version="1.0.0",
            description=f"Model provider: {publisher_name}/{plugin_name}",
            author=publisher_name
        )

    async def scan_all(self, provider_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Trigger scanning across registered providers. If provider_id is specified, only that provider is scanned and others use cache."""
        if not self._providers:
            return []
    
        async def _scan_provider(pid: str, provider: BaseModelProvider):
            try:
                logger.info(f"Scanning models with provider: {pid}")
                models = await provider.scan()
                self._provider_models[pid] = models
            except Exception as e:
                logger.error(f"Error during scan with provider {pid}: {e}")
                self._provider_models[pid] = []
        tasks = []
        for pid, prov in self._providers.items():
            # Scan if provider matches OR if we haven't cached models yet
            if provider_id is None or pid == provider_id or pid not in self._provider_models:
                tasks.append(_scan_provider(pid, prov))
        if tasks:
            await asyncio.gather(*tasks)
        all_models = []
        for pid_models in self._provider_models.values():
            all_models.extend(pid_models)
        return all_models

    async def update_config(self, config_path: str, provider_id: Optional[str] = None):
        """Scan all models and update config.json with auto-loading and auto-defaults."""
        if not self._providers:
            logger.info("No model providers loaded. Skipping config update.")
            return
        # Perform Scan
        scan_results = await self.scan_all(provider_id=provider_id)
        available_models = {m['id']: m for m in scan_results if 'id' in m}
        try:
            async with self.config_lock:
            
                def _do_update():
                    config = {"models": {}, "defaults": {}, "available_models": {}}
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    config = json.loads(content)
                        except (json.JSONDecodeError, Exception) as e:
                            logger.warning(f"Failed to read config, initializing: {e}")
                    # Update available models
                    config['available_models'] = available_models
                    # Ensure section keys exist
                    if 'models' not in config: config['models'] = {}
                    if 'defaults' not in config: config['defaults'] = {}
                    # A. Sync existing models and auto-load models tagged with auto_load: True
                    for mid, model in available_models.items():
                        if mid in config['models']:
                            # Update existing entry with latest metadata (sync backend etc)
                            # but preserve stateful fields
                            for k, v in model.items():
                                if k not in ['auto_load', 'last_error', 'mmproj_path']:
                                    config['models'][mid][k] = v
                        elif model.get('auto_load'):
                            config['models'][mid] = model
                            logger.info(f"Auto-loaded model: {mid}")
                    # B. Clean up stale models/defaults
                    # 1. Clean up models: must be in available_models
                    config['models'] = {
                        k: v for k, v in config['models'].items()
                        if k in available_models
                    }
                    # 2. Clean up defaults: must be in config['models'] (loaded models)
                    config['defaults'] = {
                        k: v for k, v in config['defaults'].items()
                        if v in config['models']
                    }
                    # C. Auto-set defaults by type (if not set)
                    types_to_check = ['llm', 'stt', 'voice_activity', 'command_recognizer']
                    for mtype in types_to_check:
                        current_default = config['defaults'].get(mtype)
                        if not current_default or current_default not in config['models']:
                            # Only find an active model of this type from the 'models' section
                            potential_models = [
                                mid for mid, m in config['models'].items()
                                if mtype == m.get('model_type') or (isinstance(m.get('model_type'), list) and mtype in m.get('model_type'))
                            ]
                            if potential_models:
                                config['defaults'][mtype] = potential_models[0]
                                logger.info(f"Auto-set default for {mtype}: {potential_models[0]}")
                            else:
                                # If no active model of this type, clear the default
                                config['defaults'].pop(mtype, None)
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=4)
                await asyncio.to_thread(_do_update)
                logger.info(f"Updated {config_path} with {len(available_models)} available models.")
        except Exception as e:
            logger.error(f"Failed to update config file {config_path}: {e}")
