from contextvars import ContextVar
from typing import Any, Dict, Optional, Callable, Awaitable, List, TYPE_CHECKING
if TYPE_CHECKING:
    from .pipeline_base import PipelineBase
# Context variable
workspace_ctx: ContextVar[Optional[str]] = ContextVar("workspace", default="global")
tab_id_ctx: ContextVar[Optional[str]] = ContextVar("tab_id", default="global")
model_id_ctx: ContextVar[Optional[str]] = ContextVar("model_id", default=None)
