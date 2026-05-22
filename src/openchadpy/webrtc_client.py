"""
!!!!!!!! WORK IN PROGRESSS !!!!!!!!!!!!!!
"""
from __future__ import annotations
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import ValidationError
from aiortc import RTCDataChannel, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from websockets.asyncio.client import connect as ws_connect
from mcp import types
from mcp.shared.session import SessionMessage
logger = logging.getLogger(__name__)
@asynccontextmanager
async def webrtc_client(
    signaling_url: str,
    channel_label: str = "mcp",
    ordered: bool = True,
    ice_timeout: float = 10.0,
    open_timeout: float = 15.0,
    ice_servers = [
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    ]
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ],
    None,
]:
    """WebRTC DataChannel transport for MCP  client side.
    Mirrors ``mcp.client.websocket.websocket_client`` exactly: it is an
    async context manager that yields ``(read_stream, write_stream)``.
    Parameters
    ----------
    signaling_url:
        WebSocket URL of the signaling server, e.g. ``"ws://localhost:8765"``.
    channel_label:
        RTCDataChannel label (default ``"mcp"``).
    ordered:
        Whether the data channel guarantees ordered delivery.
    ice_timeout:
        Seconds to wait for ICE gathering and for the answer from the server.
    open_timeout:
        Seconds to wait for the RTCDataChannel to reach *open* state after
        the WebRTC handshake completes.
    Yields
    ------
    ``(read_stream, write_stream)``
        Same anyio stream pair as every other MCP transport.
        * ``read_stream``   receive ``SessionMessage | Exception`` objects
        * ``write_stream``  send ``SessionMessage`` objects
    """
    #
    #   read_stream_writer  →  [ buffer ]  →  read_stream   (MCP session reads)
    #   write_stream        →  [ buffer ]  →  write_stream_reader  (we drain)
    #
    config = RTCConfiguration(iceServers=ice_servers)
    read_stream_writer: MemoryObjectSendStream[SessionMessage | Exception]
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception]
    write_stream: MemoryObjectSendStream[SessionMessage]
    write_stream_reader: MemoryObjectReceiveStream[SessionMessage]
    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)
    # asyncio.Queue bridges aiortc's *synchronous* on_message callback into
    # the async anyio world without blocking the event loop.
    # EOFError is used as a sentinel to signal channel closure.
    _incoming: asyncio.Queue[SessionMessage | Exception | EOFError] = asyncio.Queue()
    pc = RTCPeerConnection(configuration=config)
    # The *offerer* (client) creates the data channel.
    # The *answerer* (server) receives it via the @pc.on("datachannel") event.
    channel: RTCDataChannel = pc.createDataChannel(channel_label, ordered=ordered)
    _channel_open: asyncio.Event = asyncio.Event()
    _ice_complete: asyncio.Event = asyncio.Event()
    @channel.on("open")
    def _on_open() -> None:
        logger.debug("data channel '%s' open", channel_label)
        _channel_open.set()
    @channel.on("close")
    def _on_close() -> None:
        logger.debug("data channel '%s' closed", channel_label)
        _incoming.put_nowait(EOFError("WebRTC data channel closed"))
    @channel.on("message")
    def _on_message(raw: str | bytes) -> None:
        """Parse incoming text → SessionMessage and push to queue."""
        text = raw if isinstance(raw, str) else raw.decode()
        try:
            msg = types.JSONRPCMessage.model_validate_json(text)
            _incoming.put_nowait(SessionMessage(msg))
        except ValidationError as exc:
            _incoming.put_nowait(exc)
    @pc.on("icegatheringstatechange")
    def _on_ice_state() -> None:
        logger.debug("ICE gathering state → %s", pc.iceGatheringState)
        if pc.iceGatheringState == "complete":
            _ice_complete.set()
    try:
        async with ws_connect(signaling_url) as ws:
            # Step 1  create offer and start ICE gathering
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            # Step 2  wait for all ICE candidates so SDP is self-contained
            try:
                await asyncio.wait_for(_ice_complete.wait(), timeout=ice_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "ICE gathering timed out after %.1fs  sending partial SDP",
                    ice_timeout,
                )
            # Step 3  send complete offer (candidates embedded in SDP)
            await ws.send(
                json.dumps(
                    {
                        "type": "offer",
                        "sdp": pc.localDescription.sdp,
                    }
                )
            )
            logger.debug("offer → signaling server %s", signaling_url)
            # Step 4  receive answer
            try:
                raw_answer = await asyncio.wait_for(ws.recv(), timeout=ice_timeout)
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Timed out waiting for answer from signaling server at {signaling_url}"
                ) from None
            answer_msg = json.loads(raw_answer)
            if answer_msg.get("type") != "answer":
                raise RuntimeError(
                    f"Expected 'answer' from signaling server, "
                    f"got type={answer_msg.get('type')!r}: {raw_answer[:200]}"
                )
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer_msg["sdp"], type="answer")
            )
            logger.debug("remote description set; awaiting data channel open")
        # Step 5  wait for the data channel to become ready
        try:
            await asyncio.wait_for(_channel_open.wait(), timeout=open_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Data channel '{channel_label}' did not open within {open_timeout}s. "
                "Check that the MCP server also has a WebRTC transport running."
            ) from None
        logger.info(
            "WebRTC MCP transport ready  channel=%r  signaling=%s",
            channel_label,
            signaling_url,
        )
        async def _incoming_to_stream() -> None:
            """Relay incoming queue → read_stream until EOF sentinel."""
            async with read_stream_writer:
                while True:
                    item = await _incoming.get()
                    if isinstance(item, EOFError):
                        # Channel closed  close the stream so the MCP session
                        # exits cleanly rather than hanging.
                        logger.debug("EOF sentinel received; closing read_stream")
                        break
                    await read_stream_writer.send(item)
        async def _stream_to_dc() -> None:
            """Relay write_stream → RTCDataChannel."""
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    if channel.readyState != "open":
                        logger.warning(
                            "dropping outgoing message  channel state is '%s'",
                            channel.readyState,
                        )
                        break
                    payload = session_message.message.model_dump(
                        by_alias=True, mode="json", exclude_none=True
                    )
                    channel.send(json.dumps(payload))
        async with anyio.create_task_group() as tg:
            tg.start_soon(_incoming_to_stream)
            tg.start_soon(_stream_to_dc)
            # Hand control back to the MCP ClientSession
            yield (read_stream, write_stream)
            # Caller exited the `async with webrtc_client(...)` block
            tg.cancel_scope.cancel()
    finally:
        await pc.close()
        logger.debug("RTCPeerConnection closed")