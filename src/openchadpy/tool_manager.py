"""
Tool Manager - Discovers, loads, and manages tools from the Tools/ directory.
"""
import os
import sys
import logging
import importlib
import importlib.util
import re
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import asyncio
from .tool_base import ToolBase
from .context import workspace_ctx, tab_id_ctx
if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP
    
logger = logging.getLogger(__name__)

@dataclass
class ToolMetadata:
    """Metadata describing a tool."""
    name: str
    version: str
    description: str
    author: str = ""
    requirements: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "requirements": self.requirements,
        }
    
class ToolManager:
    """Manages the lifecycle of tools."""

    _active_workspace : Optional[str]
    _active_tab_id : Optional[str]

    def __init__(self, tools_directory: str):
        """Initialize the tool manager."""
        self.tools_directory = Path(tools_directory).resolve()
        # Flat registry: tool_instance.name → ToolBase
        self.loaded_tools: Dict[str, ToolBase] = {}
        self._metadata: Dict[str, ToolMetadata] = {}
        self._tool_modules: Dict[str, Any] = {}   # module_name → module
        # Maps tool_instance.name → module_name (for cleanup)
        self._tool_module_name: Dict[str, str] = {}
        self.managers: Dict[str, Any] = {}
        self._active_workspace = None
        self._active_tab_id = None
        tools_parent = str(self.tools_directory.parent)
        if tools_parent not in sys.path:
            sys.path.insert(0, tools_parent)
    
    def set_active(self, w:str, tid: str):
        logger.info(f"workspace: {w} tab_id: {tid}")
        self._active_workspace = w
        self._active_tab_id = tid

    @property
    def active_workspace(self) -> str:
        return self._active_workspace or "global"
    
    @property
    def active_tab_id(self) -> str:
        return self._active_tab_id or "global"

    # Internal helpers
    def _load_metadata_from_manifest(
        self, manifest_path: Path, fallback_name: str
    ) -> ToolMetadata:
        """Read manifest.json without loading the tool class."""
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            return ToolMetadata(
                name=data.get("name", fallback_name),
                version=data.get("version", "1.0.0"),
                description=data.get("description", ""),
                author=data.get("author", ""),
                requirements=data.get("requirements", []),
            )
        except Exception as e:
            logger.warning(f"Failed to read manifest at {manifest_path}: {e}")
            return ToolMetadata(
                name=fallback_name,
                version="1.0.0",
                description=f"Tool: {fallback_name}",
                author="",
            )

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a directory name to be a valid Python identifier segment."""
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
            logger.warning(f"Invalid name: {name}, generating hash")
            return hashlib.sha256(name.encode()).hexdigest()[:9]
        return name
    
    # Discovery
    def _discover_local_items(self, directory: Path) -> List[Tuple[str, Path]]:
        """
        Scan directory for tools.
        Returns a list of (storage_key, tool_path) tuples where storage_key is
        ``publisher_plugin``  used only as the Python module name. The actual
        registry key is set later from ``tool_instance.name``.
        """
        items: List[Tuple[str, Path]] = []
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return items
        for publisher_dir in directory.iterdir():
            if not publisher_dir.is_dir() or publisher_dir.name.startswith((".", "_")):
                continue
            publisher_name = self._sanitize_name(publisher_dir.name)
            for plugin_dir in publisher_dir.iterdir():
                if not plugin_dir.is_dir() or plugin_dir.name.startswith((".", "_")):
                    continue
                main_py = plugin_dir / "main.py"
                if not main_py.exists():
                    continue
                plugin_name = self._sanitize_name(plugin_dir.name)
                storage_key = f"{publisher_name}_{plugin_name}"
                items.append((storage_key, main_py))
                manifest_path = plugin_dir / "manifest.json"
                if manifest_path.exists():
                    self._metadata[storage_key] = self._load_metadata_from_manifest(
                        manifest_path, plugin_name
                    )
                logger.debug(f"Discovered: {storage_key}")
        return items

    def discover_tools(self) -> List[Tuple[str, Path]]:
        """Scan the Tools/ directory. Returns (storage_key, main_py_path) pairs."""
        return self._discover_local_items(self.tools_directory)
    
    # Manager injection
    def set_managers(self, **kwargs):
        """Set manager references for injection into tools."""
        self.managers.update(kwargs)

    def inject_managers(self, tool_instance: ToolBase):
        """Inject manager references into a tool instance."""
        for name, manager in self.managers.items():
            if hasattr(tool_instance, name):
                setattr(tool_instance, name, manager)
    
    # Load / Unload / Reload
    async def load_tool(self, storage_key: str, _rebuild: bool = True) -> bool:
        """
        Load a tool from disk.
        Args:
            storage_key: ``publisher_plugin`` string  used only to locate the
                         file.  The registry key is set from ``tool_instance.name``.
            _rebuild:    Unused; kept for call-site compatibility with the
                         concurrent loader (which passes ``_rebuild=False``).
        """
        parts = storage_key.split("_")
        if len(parts) < 2:
            logger.error(f"Invalid tool key: {storage_key} (expected publisher_plugin)")
            return False
        publisher_name = parts[0]
        plugin_name = "_".join(parts[1:])
        tool_path = self.tools_directory / publisher_name / plugin_name / "main.py"
        module_name = f"tool_{storage_key}"
        if not tool_path.exists():
            logger.error(f"Tool not found: {storage_key} (no main.py at {tool_path})")
            return False
        # Ensure the plugin directory is importable
        plugin_path = str(tool_path.parent.resolve())
        if plugin_path not in sys.path:
            sys.path.insert(0, plugin_path)
        # Ensure project root is importable
        project_dir = os.environ.get("OPENCHAD_UV_PROJECT_DIR")
        if project_dir:
            python_path = str(Path(project_dir).resolve())
            if python_path not in sys.path:
                sys.path.insert(0, python_path)
        try:
            spec = importlib.util.spec_from_file_location(module_name, tool_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create spec for {storage_key}")
                return False
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            await asyncio.to_thread(spec.loader.exec_module, module)
            if not hasattr(module, "Tool"):
                logger.error(f"Tool {storage_key} has no 'Tool' class export")
                return False
            tool_class = module.Tool
            if not issubclass(tool_class, ToolBase):
                logger.error(f"Tool {storage_key} does not inherit from ToolBase")
                return False
            tool_instance: ToolBase = tool_class()
            self.inject_managers(tool_instance)
            tool_instance.on_register()
            # Key the registry by the instance's declared name
            tool_name = tool_instance.name
            self.loaded_tools[tool_name] = tool_instance
            self._tool_modules[module_name] = module
            self._tool_module_name[tool_name] = module_name
            logger.info(f"Loaded tool: '{tool_name}' (module: {module_name})")
            return True
        except Exception as e:
            logger.error(f"Failed to load {storage_key}: {e}", exc_info=True)
            return False

    def unload_tool(self, tool_name: str) -> bool:
        """
        Unregister and unload a tool by its declared name.
        Args:
            tool_name: Value of ``tool_instance.name`` as stored in the registry.
        """
        tool = self.loaded_tools.get(tool_name)
        if tool is None:
            logger.warning(f"Cannot unload: tool '{tool_name}' not found")
            return False
        try:
            tool.on_unregister()
        except Exception as e:
            logger.warning(f"on_unregister error for '{tool_name}': {e}")
        del self.loaded_tools[tool_name]
        module_name = self._tool_module_name.pop(tool_name, None)
        if module_name:
            sys.modules.pop(module_name, None)
            self._tool_modules.pop(module_name, None)
        logger.info(f"Unloaded tool: '{tool_name}'")
        return True
    
    async def reload_tool(self, tool_name: str) -> bool:
        """
        Hot-reload a tool by its declared name.
        The storage_key (``publisher_plugin``) is reconstructed from the module
        name that was recorded at load time so we can re-locate the file without
        re-scanning the directory.
        """
        # Recover storage_key from the module name recorded at load time
        module_name = self._tool_module_name.get(tool_name)
        if module_name and module_name.startswith("tool_"):
            storage_key = module_name[len("tool_"):]
        else:
            # Fallback: derive storage_key from tool_name
            storage_key = re.sub(r"[:/\\]", "_", tool_name)
        # Locate the tool directory to purge submodules
        parts = storage_key.split("_")
        if len(parts) >= 2:
            tool_dir = self.tools_directory / parts[0] / "_".join(parts[1:])
            tool_dir_str = str(tool_dir.resolve()) if tool_dir.exists() else None
        else:
            tool_dir_str = None
        full_module_name = f"tool_{storage_key}"
        # Purge stale modules (top-level + any helpers the tool imported)
        stale = [
            k for k, v in list(sys.modules.items())
            if k == full_module_name
            or k.startswith(f"{full_module_name}.")
            or (
                tool_dir_str
                and (spec   := getattr(v, "__spec__", None))
                and (origin := getattr(spec, "origin", None))
                and str(Path(origin).resolve()).startswith(tool_dir_str)
            )
        ]
        for mod in stale:
            sys.modules.pop(mod, None)
        if stale:
            logger.debug(f"Purged {len(stale)} stale module(s) for '{tool_name}'")
        # Unload live instance
        self.unload_tool(tool_name)
        # Load fresh copy
        return await self.load_tool(storage_key)
    
    # Bulk operations
    async def discover_and_load_all(self) -> Dict[str, bool]:
        """Discover and load all tools concurrently for fast startup."""
        items = await asyncio.to_thread(self.discover_tools)
        load_tasks = [self.load_tool(storage_key, _rebuild=False) for storage_key, _ in items]
        storage_keys = [storage_key for storage_key, _ in items]
        results: Dict[str, bool] = {}
        if load_tasks:
            outcomes = await asyncio.gather(*load_tasks, return_exceptions=True)
            for storage_key, outcome in zip(storage_keys, outcomes):
                if isinstance(outcome, Exception):
                    logger.error(f"Failed to load {storage_key}: {outcome}")
                    results[storage_key] = False
                else:
                    results[storage_key] = bool(outcome)
        logger.info(
            f"Loaded {sum(1 for v in results.values() if v)}/{len(results)} tools"
        )
        return results
    
    # Execution
    async def execute_tool(
        self, tool_name: str, caller: str = "direct", **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by its declared name.
        Args:
            tool_name: Value of ``tool_instance.name``.
            caller:    Caller context checked against ``tool.allowed_callers``.
            **kwargs:  Arguments forwarded to ``tool.execute``.
        """

        tool = self.loaded_tools.get(tool_name)
        if tool is None:
            _mcp = self.managers["mcp_manager"]
            if _mcp: 
                # check whether it's a mcp tool
                if _mcp.has_tool(tool_name):
                    return await _mcp.execute_tool(tool_name, caller, **kwargs)
            return {"error": f"Tool '{tool_name}' not found."}
        if hasattr(tool, "allowed_callers") and caller not in tool.allowed_callers:
            return {
                "error": f"Caller '{caller}' is not permitted to invoke '{tool_name}'."
            }
        try:
            return await tool.execute(caller=caller, **kwargs)
        except Exception as e:
            logger.error(f"Tool execution error ('{tool_name}'): {e}", exc_info=True)
            return {"error": str(e)}
    
    # MCP export
    def export_all_tools(self, mcp_instance: "FastMCP") -> Dict[str, bool]:
        """
        Register all MCP-eligible tools with a FastMCP server instance.
        Only tools with ``"mcp_client"`` in ``allowed_callers`` are exported.
        """
        mcp_instance._tool_manager._tools.clear()
        results: Dict[str, bool] = {}
        _type_map: Dict[str, str] = {
            "string":  "str",
            "integer": "int",
            "number":  "float",
            "boolean": "bool",
            "array":   "list",
            "object":  "dict",
        }
        for tool_name, tool in self.loaded_tools.items():
            if "mcp_client" not in (tool.allowed_callers or []):
                results[tool_name] = False
                logger.debug(f"export_all_tools: skipped '{tool_name}' (not mcp_client)")
                continue
            try:
                wrapper = self._synthesize_mcp_wrapper(tool, _type_map)
            except Exception as exc:
                results[tool_name] = False
                logger.error(
                    f"export_all_tools: failed to synthesize wrapper for '{tool_name}': {exc}",
                    exc_info=True,
                )
                continue
            try:
                mcp_instance.add_tool(wrapper)
                results[tool_name] = True
                logger.info(f"export_all_tools: registered '{tool_name}'")
            except Exception as exc:
                results[tool_name] = False
                logger.error(
                    f"export_all_tools: failed to register '{tool_name}': {exc}",
                    exc_info=True,
                )
        exported = sum(1 for v in results.values() if v)
        logger.info(
            f"export_all_tools: {exported}/{len(self.loaded_tools)} tools exported to MCP"
        )
        return results

    def _synthesize_mcp_wrapper(
        self, tool: ToolBase, type_map: Dict[str, str]
    ) -> Any:
        """
        Build a genuine async function whose parameters mirror the tool's
        ``input_schema`` so that FastMCP can introspect a real signature.
        """
        properties: Dict[str, Any] = tool.input_schema.get("properties") or {}
        required: List[str]        = tool.input_schema.get("required")    or []
        required_params = [(n, m) for n, m in properties.items() if n     in required]
        optional_params = [(n, m) for n, m in properties.items() if n not in required]
        param_parts: List[str] = []
        for param_name, meta in required_params:
            py_type = type_map.get(meta.get("type", "string"), "Any")
            param_parts.append(f"{param_name}: {py_type}")
        for param_name, meta in optional_params:
            py_type = type_map.get(meta.get("type", "string"), "Any")
            param_parts.append(f"{param_name}: Optional[{py_type}] = None")
        params_str = ", ".join(param_parts)
        fn_name    = tool.name
        source = (
            f"async def {fn_name}({params_str}):\n"
            f"    return await _execute(**{{k: v for k, v in locals().items() if v is not None}})\n"
        )
        async def _execute(**kwargs) -> Dict[str, Any]:
            return await tool.execute(caller="mcp_client", **kwargs)
        namespace: Dict[str, Any] = {
            "_execute": _execute,
            "Optional": Optional,
            "Any":      Any,
        }
        exec(compile(source, f"<mcp_tool:{fn_name}>", "exec"), namespace)  # noqa: S102
        fn           = namespace[fn_name]
        fn.__doc__   = tool.description
        fn.__module__ = __name__
        return fn
    
    # Introspection helpers
    def get_tool(self, tool_name: str) -> Optional[ToolBase]:
        """Return a tool instance by its declared name, or None."""
        return self.loaded_tools.get(tool_name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return metadata for all loaded tools."""
        return [
            {"id": name, **tool.get_schema()}
            for name, tool in self.loaded_tools.items()
        ]

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function tool schemas for direct-callable tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.loaded_tools.values()
            if "direct" in (tool.allowed_callers or [])
        ]

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return Claude API compatible tool schemas."""
        return [tool.get_schema() for tool in self.loaded_tools.values() if "direct" in (tool.allowed_callers or [])]