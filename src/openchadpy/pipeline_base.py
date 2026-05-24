import json
import json
import logging
from openchadpy.tool_base import ToolRegistry
from typing import Any, Optional, List, Dict, Callable, Awaitable, TYPE_CHECKING
import asyncio
from pathlib import Path
from .context import workspace_ctx, tab_id_ctx, model_id_ctx
from .database import Database
if TYPE_CHECKING:
    from .model_manager import ModelManager
    from .tool_manager import ToolManager
    from .settings import Settings
    from .event_emitter import EventEmitter
    from .mcp_manager import MCPManager
logger = logging.getLogger(__name__) 
class PipelineBase:
    stream: bool
    attempt: int
    prompt: str
    _stream_start_time: float
    _stream_end_time: float
    _completion_tokens: int
    _prompt_tokens: int
    _costs: List[Dict[str, Any]]
    # Managers
    tool_manager: Optional["ToolManager"]
    model_manager: Optional["ModelManager"]
    settings_manager: Optional["Settings"]
    event_emitter: Optional["EventEmitter"]
    mcp_manager: Optional["MCPManager"]
    workspace: Optional[str]
    history: List[Dict[str, str]]    
    model_responses: List[Dict[str, str]]
    context: str
    query: Optional[str]
    files: Optional[List[str]]
    tab_id: Optional[str]
    branch_id: Optional[str]
    response_branch: Optional[int]
    index: Optional[int]
    tb: Optional[str]
    tab_db: Optional["Database"]
    db: Optional["Database"]
    send_event: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]]
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    tool_choice: Dict[str, Any]
    tool_calls: Optional[List[Dict[str, Any]]]
    args: Dict[str, Any]
    pricing: Optional[Dict[str, Any]]
    model_name: Optional[str]
    model_id: Optional[str]
    set_continue: Optional[Callable[[bool], None]]
    cancel_event: asyncio.Event
    last_response: str
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
    
    def __init__(self, **kwargs):
        self.attempt = 0
        # Initialize all attributes
        self.settings_manager = kwargs.get('settings_manager', None)
        self.event_emitter = kwargs.get('event_emitter', None)
        self.mcp_manager = kwargs.get('mcp_manager', None)
        self.tool_manager = kwargs.get('tool_manager', None)
        self.model_manager = kwargs.get('model_manager', None)
        self.messages = kwargs.get('messages', [])
        self.args = kwargs.get('args', {})
        self.workspace = kwargs.get('workspace', None)        
        self.model_responses = []
        self.last_response = ""
        self.tools = []
        self.tool_choice = {}
        self.tool_calls = []
        self.history = []
        self.context = ""
        self.query = kwargs.get('query', None)
        self.files = kwargs.get('files', [])
        self.tab_id = kwargs.get('tab_id', None)
        self.branch_id = kwargs.get('branch_id', None)
        self.response_branch = kwargs.get('response_branch', None)
        self.index = kwargs.get('index', None)
        self.stream = kwargs.get('stream', False)
        self.tb = kwargs.get('tb', None)
        self.tab_db = kwargs.get('database', None)
        self.db = Database(workspace="global", tab_id="global")
        self.pricing = kwargs.get('pricing', None)
        self.model_name = kwargs.get('model_name', None)
        self.model_id = kwargs.get('model_id', None)
        self.set_continue = kwargs.get('set_continue', None)
        self.cancel_event = kwargs.get('cancel_event', asyncio.Event())
        workspace_ctx.set(self.workspace)
        tab_id_ctx.set(self.tab_id)

    async def get_history(self) -> List[Dict[str, str]]:
        """
        Get history
        """
        return []
    
    async def setup(self) -> None:
        """
        Async initialization hook called after __init__.
        Override this to perform async setup like reading from the database.
        """
        pass

    async def process_chunk(self, chunk, **kwargs) -> Any:
        pass
    
    async def finalize(self, **kwargs) -> Any:
        pass
    
    async def response(self, res, **kwargs) -> Any:
        pass
    
    async def start(self, **kwargs) -> None:
        pass
    
    async def end(self, **kwargs) -> None:
        pass
    
    async def stop(self, **kwargs) -> None:
        pass
