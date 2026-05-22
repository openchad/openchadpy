import asyncio
import json
import logging
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from mcp import ClientSession
if TYPE_CHECKING:
    from mcp import StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.websocket import websocket_client
    from mcp.client.streamable_http import streamable_http_client
    from .settings import Settings
    from .event_emitter import EventEmitter
from .tool_base import ToolBase
logger = logging.getLogger(__name__)

# Supported transport types
TRANSPORT_STDIO = "stdio"
TRANSPORT_SSE = "sse"
TRANSPORT_WEBSOCKET = "websocket"
TRANSPORT_STREAMABLE_HTTP = "http"
TRANSPORT_WEBRTC = "webrtc"

class MCPTool(ToolBase):
    """
    Proxy tool that forwards execution to an MCP server.
    """

    def __init__(self, session: 'Any', mcp_tool_name: str, description: str, input_schema: Dict[str, Any]):
        self.session = session
        self.mcp_tool_name = mcp_tool_name
        self.name = mcp_tool_name
        self.description = description
        self.input_schema = input_schema
        self.allowed_callers = ["direct", "code_execution"]


    async def execute(self, **kwargs) -> Dict[str, Any]:
        try:
            # MCP expects the tool name without our prefix
            result = await self.session.call_tool(self.mcp_tool_name, arguments=kwargs)
            # MCP results can be a list of content blocks
            # We want to return a simplified dict for our system
            output = []
            if hasattr(result, "content"):
                for block in result.content:
                    if hasattr(block, "text"):
                        output.append(block.text)
                    elif hasattr(block, "data"):
                        output.append(block.data)
            return {"result": "\n".join(output) if output else str(result)}
        except Exception as e:
            logger.error(f"Error executing MCP tool {self.mcp_tool_name}: {e}")
            return {"error": str(e)}

class MCPManager:
    """
    Manages MCP server subprocesses and sessions.
    Supported transport types (set via server_config["type"]):
        - "stdio"            subprocess over stdin/stdout (default)
        - "sse"              HTTP Server-Sent Events
        - "websocket"        WebSocket
        - "http"             Streamable HTTP (HTTP/2 or chunked HTTP/1.1)
        - "webrtc"           WebRTC (not implemented yet)
    """
    mcp_tools: List[MCPTool]

    def __init__(self, settings_manager: 'Settings', emitter: 'EventEmitter'):
        self.settings_manager = settings_manager
        self.emitter = emitter
        self.mcp_tools = []
        self.mcp_statuses: Dict[str, str] = {}
        self.sessions: Dict[str, 'Any'] = {}
        self._clients: Dict[str, 'Any'] = {}
        self._runner_tasks: Dict[str, asyncio.Task] = {}
        self._stop_events: Dict[str, asyncio.Event] = {}
        self._loaded_configs: Dict[str, Dict[str, Any]] = {}
        self._update_lock = asyncio.Lock()
        self.heartbeat_task = None  # Not started yet

    async def start(self):
        """Call this after instantiation to begin background tasks."""
        self.settings_manager.subscribe(self._on_settings_changed)
        asyncio.create_task(self.load_mcp_tools())

    async def _on_settings_changed(self, key: str):
        if key == 'Others/app_settings/mcp.servers':
            asyncio.create_task(self._throttled_update())

    async def _throttled_update(self):
        async with self._update_lock:
            servers = await self.settings_manager.get('Others/app_settings/mcp.servers') or {}
            # 1. Unload servers that are no longer in settings or disabled
            currently_tracked = set(self._loaded_configs.keys()) | set(self.sessions.keys())
            for server_name in list(currently_tracked):
                if server_name not in servers:
                    logger.info(f"MCP server '{server_name}' removed from settings. Unloading...")
                    await self.mcp_unload(server_name)
                else:
                    server_config = servers[server_name]
                    if not server_config.get("enabled", False):
                        logger.info(f"MCP server '{server_name}' disabled in settings. Unloading...")
                        await self.mcp_unload(server_name)
            # 2. Load or update enabled servers
            for server_name, server_config in servers.items():
                if not server_config.get("enabled", False):
                    continue
                old_config = self._loaded_configs.get(server_name)
                is_loaded = server_name in self._loaded_configs
                if not is_loaded:
                    logger.info(f"MCP server '{server_name}' enabled/added. Loading...")
                    await self.mcp_load(server_name, server_config)
                elif old_config != server_config:
                    logger.info(f"MCP server '{server_name}' configuration changed. Reloading...")
                    await self.mcp_unload(server_name)
                    await self.mcp_load(server_name, server_config)
            # 3. Emit the changes 
            await self.emitter.emit("mcp_statuses", self.mcp_statuses)

    async def reload_mcp_server(self, server_name: str):
        servers = await self.settings_manager.get('Others/app_settings/mcp.servers')
        if not servers:
            return
        if server_name not in servers:
            logger.error(f"MCP server '{server_name}' not found.")
            return
        server_config = servers[server_name]
        await self.mcp_unload(server_name)
        await self.mcp_load(server_name, server_config)

    def get_openai_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-compatible function tool schemas."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            }
            for tool in self.mcp_tools
    ]

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return Claude API compatible tool schemas."""
        return [tool.get_schema() for tool in self.mcp_tools]
    
    # Transport factories
    async def _connect_stdio(self, server_config: Dict[str, Any]):
        """
        Start a stdio subprocess and return (read_stream, write_stream, transport_ctx).
        Required config keys:
            command (str)        executable to run
            args    (list[str])  command-line arguments  [optional, default []]
            env     (dict)       environment variables   [optional, default os.environ]
        """
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        command = server_config["command"]
        args = server_config.get("args", [])
        env = server_config.get("env", os.environ.copy())
        server_params = StdioServerParameters(command=command, args=args, env=env)
        transport_ctx = stdio_client(server_params)
        read_stream, write_stream = await transport_ctx.__aenter__()
        return read_stream, write_stream, transport_ctx

    async def _connect_sse(self, server_config: Dict[str, Any]):
        """
        Connect to an MCP server via Server-Sent Events (SSE) and return
        (read_stream, write_stream, transport_ctx).
        Required config keys:
            url     (str)   full SSE endpoint URL, e.g. "http://host:port/sse"
        Optional config keys:
            headers (dict)  extra HTTP headers (e.g. Authorization)
            timeout (float) connection timeout in seconds [default: 30]
        """
        from mcp.client.sse import sse_client
        url = server_config["url"]
        headers = server_config.get("headers", {})
        timeout = server_config.get("timeout", 30)
        transport_ctx = sse_client(url=url, headers=headers, timeout=timeout)
        read_stream, write_stream = await transport_ctx.__aenter__()
        return read_stream, write_stream, transport_ctx

    async def _connect_websocket(self, server_config: Dict[str, Any]):
        """
        Connect to an MCP server via WebSocket and return
        (read_stream, write_stream, transport_ctx).
        Required config keys:
            url     (str)   WebSocket URL, e.g. "ws://host:port/ws"
                             (use "wss://" for TLS)
        Optional config keys:
            headers (dict)  extra HTTP headers sent during the WS handshake
        """
        from .websocket_client import websocket_client_with_headers
        url = server_config["url"]
        headers = server_config.get("headers", {})
        transport_ctx = websocket_client_with_headers(url=url, headers=headers)
        read_stream, write_stream = await transport_ctx.__aenter__()
        return read_stream, write_stream, transport_ctx

    async def _connect_webrtc(self, server_config: Dict[str, Any]):
        from .webrtc_client import webrtc_client
        signaling_url = server_config["signaling_url"]
        channel_label = server_config.get("channel_label", "mcp")
        ice_servers = server_config.get("ice_servers", [])
        transport_ctx = webrtc_client(signaling_url, channel_label=channel_label, ice_servers=ice_servers)
        read_stream, write_stream = await transport_ctx.__aenter__()
        return read_stream, write_stream, transport_ctx

    async def _connect_streamable_http(self, server_config: Dict[str, Any]):
        import httpx
        from mcp.client.streamable_http import streamable_http_client
        url = server_config["url"]
        headers = server_config.get("headers", {})
        timeout = server_config.get("timeout", 30.0)
        sse_read_timeout = server_config.get("sse_read_timeout", 300.0)
        self._http_client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout, read=sse_read_timeout),
        )
        transport_ctx = streamable_http_client(
            url=url,
            http_client=self._http_client,
        )
        read_stream, write_stream, _ = await transport_ctx.__aenter__()
        return read_stream, write_stream, transport_ctx
    
    # Main loader
    async def _run_server_wrapper(self, server_name: str, server_config: Dict[str, Any], stop_event: asyncio.Event):
        try:
            await self._run_server(server_name, server_config, stop_event)
        except Exception as e:
            logger.error(f"Failed to run MCP server '{server_name}': {e}")
            self.mcp_statuses[server_name] = "error"
            await self.emitter.emit("mcp_statuses", self.mcp_statuses)

    async def _run_server(self, server_name: str, server_config: Dict[str, Any], stop_event: asyncio.Event):
        _TRANSPORT_FACTORY = {
            TRANSPORT_STDIO:           self._connect_stdio,
            TRANSPORT_SSE:             self._connect_sse,
            TRANSPORT_WEBSOCKET:       self._connect_websocket,
            TRANSPORT_STREAMABLE_HTTP: self._connect_streamable_http,
            TRANSPORT_WEBRTC:          self._connect_webrtc,
        }
        mcp_type = server_config.get("type", TRANSPORT_STDIO)
        factory = _TRANSPORT_FACTORY.get(mcp_type)
        if factory is None:
            raise ValueError(
                f"Unknown transport type '{mcp_type}' for server '{server_name}'. "
                f"Supported types: {list(_TRANSPORT_FACTORY)}"
            )
        read_stream, write_stream, transport_ctx = await factory(server_config)
        self._clients[server_name] = transport_ctx
        loaded_tools = []
        try:
            async with ClientSession(read_stream, write_stream) as session:
                self.sessions[server_name] = session
                await session.initialize()
                # Discover tools
                result = await session.list_tools()
                for tool in result.tools:
                    mcp_tool = MCPTool(
                        session=session,
                        mcp_tool_name=tool.name,
                        description=tool.description or "No description provided",
                        input_schema=tool.inputSchema,
                    )
                    loaded_tools.append(mcp_tool)
                    self.mcp_tools.append(mcp_tool)
                    logger.info(f"Loaded MCP tool: {mcp_tool.name} from {server_name} ({mcp_type})")
                self.mcp_statuses[server_name] = "connected"
                await self.emitter.emit("mcp_statuses", self.mcp_statuses)
                # Wait for stop event
                await stop_event.wait()
        finally:
            if loaded_tools:
                self.mcp_tools = [t for t in self.mcp_tools if t not in loaded_tools]
            self.sessions.pop(server_name, None)
            self._clients.pop(server_name, None)
            try:
                await transport_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error exiting transport context for server '{server_name}': {e}")
            if hasattr(transport_ctx, "_http_client") and transport_ctx._http_client:
                try:
                    await transport_ctx._http_client.aclose()
                except Exception:
                    pass

    async def mcp_load(self, server_name: str, server_config: Dict[str, Any]) -> None:
        try:
            self._loaded_configs[server_name] = server_config
            self.mcp_statuses[server_name] = "connecting"
            await self.emitter.emit("mcp_statuses", self.mcp_statuses)
            stop_event = asyncio.Event()
            self._stop_events[server_name] = stop_event
            # Start runner task
            task = asyncio.create_task(
                self._run_server_wrapper(server_name, server_config, stop_event)
            )
            self._runner_tasks[server_name] = task
        except Exception as e:
            logger.error(f"Failed to start MCP server '{server_name}': {e}")
            self.mcp_statuses[server_name] = "error"
            await self.emitter.emit("mcp_statuses", self.mcp_statuses)

    async def mcp_unload(self, server_name: str):
        """
        Unload an MCP server and remove its tools.
        """
        if server_name in self.mcp_statuses:
            self.mcp_statuses[server_name] = "disconnecting"
            await self.emitter.emit("mcp_statuses", self.mcp_statuses)
        self._loaded_configs.pop(server_name, None)
        stop_event = self._stop_events.pop(server_name, None)
        if stop_event:
            stop_event.set()
        task = self._runner_tasks.pop(server_name, None)
        if task:
            try:
                await task
            except Exception as e:
                logger.error(f"Error during runner task completion for '{server_name}': {e}")
        self.mcp_statuses[server_name] = "disconnected"
        await self.emitter.emit("mcp_statuses", self.mcp_statuses)
        logger.info(f"Unloaded MCP server: {server_name}")

    async def load_mcp_tools(self) -> None:
        """
        Load MCP servers from config and discover their tools.
        Each entry in mcp.servers must have a "type" key that selects the
        transport.  Additional keys are transport-specific  see the
        _connect_* helpers above for the full list.
        Example config:
            mcp:
              servers:
                my_stdio_server:
                  type: stdio
                  command: /usr/local/bin/my-mcp-server
                  args: ["--flag"]
                my_sse_server:
                  type: sse
                  url: http://localhost:8080/sse
                  headers:
                    Authorization: Bearer <token>
                my_ws_server:
                  type: websocket
                  url: ws://localhost:9090/ws
                my_http_server:
                  type: streamable_http
                  url: http://localhost:7070/mcp
                  timeout: 60
        """
        servers = await self.settings_manager.get('Others/app_settings/mcp.servers')
        if not servers:
            return
        for server_name, server_config in servers.items():
            if server_config.get("enabled"):
                await self.mcp_load(server_name, server_config)

    def has_tool(self, tool_name: str) -> bool:
        """Return True if an MCP tool with the given name is currently loaded."""
        return any(t.name == tool_name for t in self.mcp_tools)

    async def execute_tool(self, tool_name: str, caller: str = "direct",
                        **kwargs) -> Dict[str, Any]:
        """
        Find and execute an MCP tool by name, enforcing caller permissions.
        Args:
            tool_name: The name of the MCP tool to execute.
            caller:    The caller context (e.g. "direct", "code_execution").
            **kwargs:  Arguments forwarded to the tool.
        Returns:
            A dict with a "result" key on success, or an "error" key on failure.
        """
        # Locate the tool in the registered list
        tool = next((t for t in self.mcp_tools if t.name == tool_name), None)
        if tool is None:
            logger.warning(f"execute_tool: unknown tool '{tool_name}'")
            return {"error": f"Tool '{tool_name}' not found."}
        # Enforce caller whitelist defined on the tool
        if caller not in tool.allowed_callers:
            logger.warning(
                f"execute_tool: caller '{caller}' not allowed for tool '{tool_name}' "
                f"(allowed: {tool.allowed_callers})"
            )
            return {"error": f"Caller '{caller}' is not permitted to invoke '{tool_name}'."}
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            # Outer safety net  tool.execute already catches internally,
            # but guard against anything unexpected (e.g. a TypeError on bad kwargs).
            logger.error(f"execute_tool: unexpected error executing '{tool_name}': {e}")
            return {"error": str(e)}
        
    # Shutdown
    async def shutdown(self):
        """
        Close all sessions and transports.
        """
        self.settings_manager.unsubscribe(self._on_settings_changed)
        for stop_event in list(self._stop_events.values()):
            stop_event.set()
        for task in list(self._runner_tasks.values()):
            try:
                await task
            except Exception:
                pass
        self._stop_events.clear()
        self._runner_tasks.clear()
        self.sessions.clear()
        self._clients.clear()