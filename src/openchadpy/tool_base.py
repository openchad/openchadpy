import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Callable, Optional, Awaitable, Union, TYPE_CHECKING
from .context import workspace_ctx, tab_id_ctx, model_id_ctx
from .database import Database
if TYPE_CHECKING:
    from .model_manager import ModelManager
    from .tool_manager import ToolManager
    from .settings import Settings
    from .event_emitter import EventEmitter
    from .mcp_manager import MCPManager
CallerType = Literal["direct", "code_execution", "mcp_client"]

logger = logging.getLogger(__name__) 

class ToolRegistry(): 
    call: Callable[..., Awaitable[Dict[str, Any]]]
    schema: Dict[str, Any]

    def __init__(self, call: Callable[..., Awaitable[Dict[str, Any]]], schema: Dict[str, Any]):
        """Initialize instance-specific state and dependencies."""
        self.call = call
        self.schema = schema

    async def execute(self, **kwargs) -> Dict[str, Any]:
        return await self.call(**kwargs)
    
class ToolBase(ABC):
    """
    Base class for Claude-compatible programmatic tools.
    Each tool instance maintains its own state and can be executed
    independently without affecting other tool instances.
    """
    # Class-level metadata (shared across all instances)
    name: str = ""
    description: str = ""
    input_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": []
    }
    allowed_callers: List[CallerType] = ["direct"]
    app_name: Optional[str] = None
    tool_manager: Optional["ToolManager"]
    model_manager: Optional["ModelManager"]
    settings_manager: Optional["Settings"]
    event_emitter: Optional["EventEmitter"]
    mcp_manager: Optional["MCPManager"]

    async def llm_tool(
        self,
        query: str,
        tool_registry: Optional[Dict[str, ToolRegistry]] = None,
    ) -> Dict[str, Any]:
        chat_kwargs: Dict[str, Any] = {}
        mid_ctx = model_id_ctx.get()
        tools: list = []
        
        if tool_registry:
            for reg in tool_registry.values():
                tools.append(reg.schema)
        else: 
            if self.tool_manager:
                tools.extend(self.tool_manager.get_openai_schemas())
            if self.mcp_manager:
                tools.extend(self.mcp_manager.get_openai_schemas())
            
        if tools:
            chat_kwargs["tools"] = tools
                
        if self.model_manager:
            model_id = mid_ctx if mid_ctx else self.model_manager.get_default_id("llm")
            if model_id:
                response = await self.model_manager.text_chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a tool-calling assistant. You MUST call a tool for every single response"
                                "RULES:\n"
                                "1. Every response must be a tool call.\n"
                                "2. Read each tool's description fully and follow it exactly.\n"
                                "3. Match the user's intent to the correct tool. Do not guess or improvise.\n"
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    model_id=model_id,
                    stream=False,
                    **chat_kwargs,
                )
                
                assert isinstance(response, dict), f"Expected dict, got {type(response)}"
                logger.info(f"[llm_tool] query {query}")
                logger.info(f"[llm_tool] response {type(response)}, {json.dumps(response)}")
                logger.info(f"[llm_tool] tools available {type(tools)}, {json.dumps(tools)}")
                try:
                    tool_calls = response["choices"][0]["message"].get("tool_calls")
                except (KeyError, IndexError, TypeError):
                    logger.error(f"[llm_tool] error when try to get tool calls")
                    tool_calls = None
                    return {}
                if not tool_calls:
                    logger.error("[llm_tool] no tool_calls")
                if tool_calls and (self.tool_manager or tool_registry):
                    results = {}
                    for call in tool_calls:
                        fn_name: str = call["function"]["name"]
                        raw_args: str = call["function"].get("arguments", "{}")
                        call_id: str = call.get("id", fn_name)
                        try:
                            kwargs = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            logger.warning(f"[llm_tool] Failed to parse arguments for {fn_name}: {raw_args}")
                            kwargs = {}
                            return {}
                        logger.info(f"[llm_tool] Executing tool '{fn_name}' with kwargs={kwargs}")
                        if tool_registry and fn_name in tool_registry:
                            result = await tool_registry[fn_name].execute(**kwargs)
                        else:
                            if self.tool_manager:
                                result = await self.tool_manager.execute_tool(
                                    fn_name,
                                    caller="direct",
                                    workspace=self.workspace,
                                    tab_id=self.tab_id,
                                    **kwargs,
                                )
                            else:
                                return {}
                        results[call_id] = result
                    return results[next(iter(results))] if len(results) == 1 else results
                return {}
            else:
                logger.error("Model ID not found")
                return {}
        logger.error("Model manager not found")
        return {}

    def html(self, tag: str, **kwargs) -> str:
        children = kwargs.pop('children', None)
        attrs_list = [f'{k}="{v}"' for k, v in kwargs.items()]
        attrs = (" " + " ".join(attrs_list)) if attrs_list else ""
        if tag in ['input', 'br', 'img', 'hr', 'meta', 'link']:
            return f'<{tag}{attrs}/>'
        if children is not None:
            if isinstance(children, list):
                children_str = "\n".join(str(c) for c in children)
            else:
                children_str = str(children)
            indented = "\n".join("  " + line for line in children_str.splitlines())
            return f'<{tag}{attrs}>\n{indented}\n</{tag}>'
        else:
            return f'<{tag}{attrs}></{tag}>'

    def __init__(self):
        """Initialize instance-specific state and dependencies."""
        # Validate required class attributes
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} must define a non-empty 'name' attribute")
        if not self.description:
            raise ValueError(f"{self.__class__.__name__} must define a non-empty 'description' attribute")        
        self.tool_manager = None
        self.model_manager = None
        self.settings_manager = None

    @property
    def tab_db(self) -> Database:
        """Getter for Tab Database"""
        logger.info("Tab Database: Workspace and Tab_Id = {}, {}".format(self.workspace, self.tab_id))
        return Database(workspace=self.workspace, tab_id=self.tab_id)
    
    @property
    def db(self) -> Database:
        """Getter for Database"""
        return Database(workspace="global", tab_id="global")
    
    @property
    def workspace(self) -> str:
        """Getter for workspace name, preferring tool_manager over global context"""
        context_value = workspace_ctx.get()
        
        # If context has "global", prefer tool_manager
        if context_value == "global" and self.tool_manager:
            return self.tool_manager.active_workspace or "global"
        
        # Otherwise use context if available
        return context_value or (self.tool_manager.active_workspace if self.tool_manager else None) or "global"

    @property
    def tab_id(self) -> str:
        """Getter for tab_id, preferring tool_manager over global context"""
        context_value = tab_id_ctx.get()
        
        # If context has "global", prefer tool_manager
        if context_value == "global" and self.tool_manager:
            return self.tool_manager.active_tab_id or "global"
        
        # Otherwise use context if available
        return context_value or (self.tool_manager.active_tab_id if self.tool_manager else None) or "global"

    def get_schema(self) -> Dict[str, Any]:
        """
        Return LLM API compatible tool schema.
        This can be passed directly to the LLM API tools array.
        Returns:
            Schema dictionary containing tool metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "allowed_callers": self.allowed_callers
        }
    @abstractmethod

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.
        Must be async to support parallel calling from code execution.
        Subclasses must implement this method to define tool behavior.
        Args:
            **kwargs: Tool-specific parameters matching input_schema
        Returns:
            Dictionary containing the tool result (will be serialized to JSON)
        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        pass

    def on_register(self) -> None:
        """
        Called when tool is loaded/registered.
        Override this method to perform setup operations like:
        - Initializing connections
        - Loading resources
        - Validating configuration
        """
        pass

    def on_unregister(self) -> None:
        """
        Called when tool is unloaded/unregistered.
        Override this method to perform cleanup operations like:
        - Closing connections
        - Releasing resources
        - Saving state
        """
        pass

    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"<Tool: {self.name}>"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.name}: {self.description[:50]}..." if len(self.description) > 50 else f"{self.name}: {self.description}"