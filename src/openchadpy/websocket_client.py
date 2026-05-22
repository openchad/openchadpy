import websockets
from mcp.client.session import ClientSession
from mcp.types import JSONRPCMessage
from mcp.shared.message import SessionMessage
from anyio.streams.memory import MemoryObjectSendStream, MemoryObjectReceiveStream
import anyio
from contextlib import asynccontextmanager

@asynccontextmanager
async def websocket_client_with_headers(url: str, headers: dict):
    async with websockets.connect(url, additional_headers=headers) as ws:
        read_send, read_recv = anyio.create_memory_object_stream(max_buffer_size=0)
        write_send, write_recv = anyio.create_memory_object_stream(max_buffer_size=0)

        async def ws_reader():
            async for msg in ws:
                json_rpc_msg = JSONRPCMessage.model_validate_json(msg)
                await read_send.send(SessionMessage(message=json_rpc_msg))
       
        async def ws_writer():
            async for session_msg in write_recv:
                payload = session_msg.message.model_dump_json(by_alias=True, exclude_none=True)
                await ws.send(payload)
       
        async with anyio.create_task_group() as tg:
            tg.start_soon(ws_reader)
            tg.start_soon(ws_writer)
            yield read_recv, write_send
            tg.cancel_scope.cancel()