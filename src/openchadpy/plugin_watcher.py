"""
Plugin Watcher - Dynamic hot-reload system for plugin-based components.
Watches the following directories for changes:
- Backend/{publisher}/{plugin}/main.py
- Tools/{publisher}/{plugin}/main.py
- Pipeline/{publisher}/{plugin}/main.py
- ModelProvider/{publisher}/{plugin}/main.py
Features:
- Auto-reload on file modification
- Auto-load on new plugin creation
- Auto-unload on plugin deletion
"""
from .tool_base import ToolBase
import asyncio
import logging
import os
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set
from watchfiles import Change, awatch

if TYPE_CHECKING:
    from .backend_registry import BackendRegistry
    from .pipeline_manager import PipelineManager
    from .tool_manager import ToolManager
    from .model_manager import ModelManager
    from .mcp_manager import MCPManager
    from .event_emitter import EventEmitter
    from .model_provider import ModelProviderManager
    from .settings import Settings
    from mcp.server.fastmcp import FastMCP
    
logger = logging.getLogger(__name__)

# Script directory for resolving paths
_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent

class PluginWatcher:
    """
    Watches plugin directories for changes and dispatches reload/load/unload events.
    Usage:
        watcher = PluginWatcher(
            backend_registry=backend_registry,
            tool_manager=tool_manager,
            # ... other managers
        )
        await watcher.start()
    """
    # Plugin directory names to watch
    PLUGIN_DIRS = ["Backend", "Tools", "Pipeline", "ModelProvider"]

    def __init__(
        self,
        mcp_instance: Optional["FastMCP"] = None,
        backend_registry: Optional["BackendRegistry"] = None,
        pipeline_manager: Optional["PipelineManager"] = None,
        tool_manager: Optional["ToolManager"] = None,
        model_manager: Optional["ModelManager"] = None,
        event_emitter: Optional["EventEmitter"] = None,
        mcp_manager: Optional["MCPManager"] = None,
        model_provider_manager: Optional["ModelProviderManager"] = None,
        config_path: Optional[str] = None,
        project_root: Optional[str] = None,
        backends_dir: Optional[str] = None,        
        pipelines_dir: Optional[str] = None,
        tools_dir: Optional[str] = None,
        model_providers_dir: Optional[str] = None,
        settings_manager: Optional["Settings"] = None,
    ):
        """
        Initialize the plugin watcher.
        Args:
            backend_registry: BackendRegistry instance for Backend plugins
            pipeline_manager: PipelineManager instance for Pipeline plugins
            tool_manager: ToolManager instance for Tools plugins
            model_manager: ModelManager instance for dependency injection
            event_emitter: EventEmitter instance for dependency injection
            model_provider_manager: ModelProviderManager instance for ModelProvider plugins
            config_path: Path to config.json for updating available models
            project_root: Root directory of the project (default: auto-detected)
            backends_dir: Path to Backend plugins directory
            pipelines_dir: Path to Pipeline plugins directory
            tools_dir: Path to Tools plugins directory
            model_providers_dir: Path to ModelProvider directory
        """
        self.mcp_instance = mcp_instance
        self.backend_registry = backend_registry
        self.pipeline_manager = pipeline_manager
        self.tool_manager = tool_manager
        self.model_manager = model_manager
        self.event_emitter = event_emitter
        self.mcp_manager = mcp_manager
        self.model_provider_manager = model_provider_manager
        self.settings_manager = settings_manager
        self.project_root = Path(project_root) if project_root else _PROJECT_ROOT
        self.config_path = config_path
        # Initialize plugin paths mapping
        self.plugin_paths = {
            "Backend": backends_dir or str(self.project_root / "Backend"),
            "Pipeline": pipelines_dir or str(self.project_root / "Pipeline"),
            "Tools": tools_dir or str(self.project_root / "Tools"),
            "ModelProvider": model_providers_dir or str(self.project_root / "ModelProvider"),
        }
        # Track loaded plugins per category: {"Backend": {"publisher:plugin", ...}}
        self._loaded_plugins: Dict[str, Set[str]] = {d: set() for d in self.PLUGIN_DIRS}
        # Watcher task reference
        self._watcher_task: Optional[asyncio.Task] = None
        self._running = False

    def _parse_plugin_path(self, file_path: str) -> Optional[tuple]:
        """
        Parse a file path to extract category, publisher, and plugin name.
        Args:
            file_path: Absolute or relative path to a file
        Returns:
            Tuple of (category, publisher, plugin_name) or None if not a plugin file
        """
        try:
            path = Path(file_path).resolve()
            # Check if this is a main.py or manifest.json in a plugin directory
            if path.name not in ["main.py", "manifest.json", "settings.toml"]:
                # Check if it's any Python file in a plugin directory
                if path.suffix != ".py":
                    return None
            # If it's settings.toml, we can just return a Settings category
            if path.name == "settings.toml":
                # Check if it's inside any known plugin path
                for base_path in self.plugin_paths.values():
                    if path.is_relative_to(Path(base_path).resolve()):
                        return ("Settings", "global", "settings.toml")
                return None
            # Find which plugin category this path belongs to
            for category, base_path in self.plugin_paths.items():
                try:
                    base_path_obj = Path(base_path).resolve()
                    if path.is_relative_to(base_path_obj):
                        # Extract relative parts
                        rel_parts = path.relative_to(base_path_obj).parts
                        # Standard handling for other plugins: {publisher}/{plugin}/file.py
                        if len(rel_parts) >= 2:
                            publisher = rel_parts[0]
                            plugin = rel_parts[1]
                            # Skip hidden/underscore directories
                            if publisher.startswith(('.', '_')) or plugin.startswith(('.', '_')):
                                return None
                            return (category, publisher, plugin)
                except (ValueError, Exception):
                    continue
            return None
        except Exception as e:
            logger.debug(f"Failed to parse plugin path {file_path}: {e}")
            return None

    def _get_plugin_key(self, publisher: str, plugin: str, category: Optional[str] = None) -> str:
        """Generate plugin key. For standard plugins, publisher/plugin."""
        return f"{publisher}/{plugin}".lower()

    def _inject_dependencies(self, tool_instance: Optional[ToolBase]):
        """Inject dependencies into a tool instance."""
        if self.tool_manager and tool_instance:
            self.tool_manager.inject_managers(tool_instance)
        logger.info(f"Injected dependencies into {getattr(tool_instance, 'name', 'unknown tool')}")

    async def _handle_change(self, change_type: Change, file_path: str):
        """
        Handle a file change event.
        Args:
            change_type: Type of change (added, modified, deleted)
            file_path: Path to the changed file
        """
        parsed = self._parse_plugin_path(file_path)
        if not parsed:
            return
        category, publisher, plugin = parsed
        plugin_key = self._get_plugin_key(publisher, plugin, category)
        logger.info(f"Plugin change detected: {change_type.name} {category}/{publisher}/{plugin}")
        try:
            if change_type == Change.modified:
                await self._reload_plugin(category, plugin_key)
            elif change_type == Change.added:
                await self._load_plugin(category, plugin_key)
            elif change_type == Change.deleted:
                await self._unload_plugin(category, plugin_key)
        except Exception as e:
            logger.error(f"Failed to handle plugin change {category}/{plugin_key}: {e}", exc_info=True)

    async def _reload_plugin(self, category: str, plugin_key: str):
        """Reload an existing plugin."""
        logger.info(f"Reloading {category} plugin: {plugin_key}")
        if category == "Settings" and self.settings_manager:
            await self.settings_manager.initialize()
            return
        if category == "Backend" and self.backend_registry:
            await self.backend_registry.reload_backend(plugin_key)
        elif category == "Tools" and self.tool_manager:
            if await self.tool_manager.reload_tool(plugin_key):
                # plugin_key is "publisher/plugin" (slash); loaded_tools uses
                # "publisher_plugin" (underscore)  normalise before lookup.
                storage_key = re.sub(r"[:/\\]", "_", plugin_key)
                tool_instance = self.tool_manager.loaded_tools.get(storage_key)
                if tool_instance:
                    self._inject_dependencies(tool_instance)
            if self.mcp_instance:
                self.tool_manager.export_all_tools(self.mcp_instance)
        elif category == "Pipeline" and self.pipeline_manager:
            await self.pipeline_manager.reload_pipeline(plugin_key)
        elif category == "ModelProvider" and self.model_provider_manager:
            publisher, plugin = plugin_key.split("/", 1)
            plugin_dir = self.project_root / category / publisher / plugin
            await self.model_provider_manager.reload_provider(plugin_key, plugin_dir)
            if self.config_path:
                await self.model_provider_manager.update_config(self.config_path)
                if self.model_manager:
                    await self.model_manager.load_config()

    async def _load_plugin(self, category: str, plugin_key: str):
        """Load a new plugin."""
        logger.info(f"Loading new {category} plugin: {plugin_key}")
        if category == "Settings" and self.settings_manager:
            await self.settings_manager.initialize()
            return
        # Check if plugin directory has main.py
        publisher, plugin = plugin_key.split(":", 1)
        plugin_dir = self.project_root / category / publisher / plugin
        main_py = plugin_dir / "main.py"
        if not main_py.exists():
            logger.debug(f"No main.py found for {category}/{plugin_key}, skipping load")
            return
        if category == "Backend" and self.backend_registry:
            try:
                await self.backend_registry.discover()
                self._loaded_plugins[category].add(plugin_key)
            except Exception as e:
                logger.error(f"Failed to load Backend {plugin_key}: {e}")
        elif category == "Tools" and self.tool_manager:
            try:
                if await self.tool_manager.load_tool(plugin_key):
                    self._loaded_plugins[category].add(plugin_key)
                    tool_instance = self.tool_manager.loaded_tools.get(plugin_key)
                    if tool_instance:
                        self._inject_dependencies(tool_instance)
            except Exception as e:
                logger.error(f"Failed to load Tool {plugin_key}: {e}")
            if self.mcp_instance:
                self.tool_manager.export_all_tools(self.mcp_instance)        
        elif category == "Pipeline" and self.pipeline_manager:
            try:
                await self.pipeline_manager.discover()
                self._loaded_plugins[category].add(plugin_key)
            except Exception as e:
                logger.error(f"Failed to load Pipeline {plugin_key}: {e}")
        elif category == "ModelProvider" and self.model_provider_manager:
            try:
                publisher, plugin = plugin_key.split(":", 1)
                plugin_dir = self.project_root / category / publisher / plugin
                await self.model_provider_manager.reload_provider(plugin_key, plugin_dir)
                self._loaded_plugins[category].add(plugin_key)
                if self.config_path:
                    await self.model_provider_manager.update_config(self.config_path)
                    if self.model_manager:
                        await self.model_manager.load_config()
            except Exception as e:
                logger.error(f"Failed to load ModelProvider {plugin_key}: {e}")

    async def _unload_plugin(self, category: str, plugin_key: str):
        """Unload a deleted plugin."""
        logger.info(f"Unloading {category} plugin: {plugin_key}")
        if category == "Settings" and self.settings_manager:
            await self.settings_manager.initialize()
            return
        if category == "Backend" and self.backend_registry:
            try:
                self.backend_registry.unload_backend(plugin_key)
                self._loaded_plugins[category].discard(plugin_key)
            except Exception as e:
                logger.error(f"Failed to unload Backend {plugin_key}: {e}")
        elif category == "Tools" and self.tool_manager:
            try:
                self.tool_manager.unload_tool(plugin_key)
                self._loaded_plugins[category].discard(plugin_key)
            except Exception as e:
                logger.error(f"Failed to unload Tool {plugin_key}: {e}")
            if self.mcp_instance:
                self.tool_manager.export_all_tools(self.mcp_instance)            
        elif category == "Pipeline" and self.pipeline_manager:
            try:
                self.pipeline_manager.unload_pipeline(plugin_key)
                self._loaded_plugins[category].discard(plugin_key)
            except Exception as e:
                logger.error(f"Failed to unload Pipeline {plugin_key}: {e}")
        elif category == "ModelProvider" and self.model_provider_manager:
            try:
                await self.model_provider_manager.unload_provider(plugin_key)
                self._loaded_plugins[category].discard(plugin_key)
                if self.config_path:
                    await self.model_provider_manager.update_config(self.config_path)
                    if self.model_manager:
                        await self.model_manager.load_config()
            except Exception as e:
                logger.error(f"Failed to unload ModelProvider {plugin_key}: {e}")

    async def _watch_plugins(self):
        """Main watcher loop."""
        # Build list of directories to watch
        watch_dirs = []
        for category, dir_path_str in self.plugin_paths.items():
            dir_path = Path(dir_path_str)
            if dir_path.exists():
                watch_dirs.append(str(dir_path))
                logger.info(f"Watching {category} plugin directory: {dir_path}")
            else:
                logger.warning(f"{category} plugin directory not found: {dir_path}")
        if not watch_dirs:
            logger.error("No plugin directories to watch!")
            return
        try:
            # Watch all directories with 500ms debounce
            async for changes in awatch(*watch_dirs, debounce=500):
                for change_type, file_path in changes:
                    await self._handle_change(change_type, file_path)
        except asyncio.CancelledError:
            logger.info("Plugin watcher cancelled")
        except Exception as e:
            logger.error(f"Plugin watcher error: {e}", exc_info=True)

    async def start(self):
        """Start watching plugin directories."""
        if self._running:
            logger.warning("Plugin watcher already running")
            return
        self._running = True
        logger.info("Starting plugin watcher...")
        self._watcher_task = asyncio.create_task(self._watch_plugins())

    def stop(self):
        """Stop watching plugin directories."""
        if self._watcher_task:
            self._watcher_task.cancel()
            self._watcher_task = None
        self._running = False
        logger.info("Plugin watcher stopped")
# Convenience function to create and start watcher
async def create_plugin_watcher(
    mcp_instance: Optional["FastMCP"] = None,
    backend_registry: Optional["BackendRegistry"] = None,
    pipeline_manager: Optional["PipelineManager"] = None,
    tool_manager: Optional["ToolManager"] = None,
    model_manager: Optional["ModelManager"] = None,
    mcp_manager: Optional["MCPManager"] = None,
    settings_manager: Optional["Settings"] = None,
    event_emitter: Optional["EventEmitter"] = None,
    model_provider_manager: Optional["ModelProviderManager"] = None,
    config_path: Optional[str] = None,
    project_root: Optional[str] = None,
    backends_dir: Optional[str] = None,
    pipelines_dir: Optional[str] = None,
    tools_dir: Optional[str] = None,
    model_providers_dir: Optional[str] = None,
) -> PluginWatcher:
    """
    Create and start a plugin watcher.
    Returns:
        Running PluginWatcher instance
    """
    watcher = PluginWatcher(
        mcp_instance=mcp_instance,
        backend_registry=backend_registry,
        pipeline_manager=pipeline_manager,
        tool_manager=tool_manager,
        model_manager=model_manager,
        mcp_manager=mcp_manager,
        settings_manager=settings_manager,
        event_emitter=event_emitter,
        model_provider_manager=model_provider_manager,
        config_path=config_path,
        project_root=project_root,
        backends_dir=backends_dir,
        pipelines_dir=pipelines_dir,
        tools_dir=tools_dir,
        model_providers_dir=model_providers_dir,
    )
    await watcher.start()
    return watcher