from fastapi import Request, HTTPException
from fastapi.responses import StreamingResponse, Response
import os
import mimetypes
from typing import Optional

def range_requests_response(
    request: Request, file_path: str, chunk_size: int = 1024 * 1024  # 1MB chunks
):
    """
    Handle HTTP Range requests for video streaming with seek support
    This allows:
    - Video seeking/scrubbing
    - Progressive loading
    - Large file support without loading everything into memory
    Args:
        request: FastAPI Request object (to get Range header)
        file_path: Path to the file
        chunk_size: Size of chunks to stream (default 1MB)
    """
    file_size = os.path.getsize(file_path)
    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "application/octet-stream"
    # Get range header
    range_header = request.headers.get("range")
    # No range request - stream entire file
    if not range_header:
        def iterfile():
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk
        return StreamingResponse(
            iterfile(),
            media_type=mime_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(file_size),
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    # Parse range header (format: "bytes=start-end")
    try:
        range_str = range_header.replace("bytes=", "")
        range_start, range_end = range_str.split("-")
        start = int(range_start) if range_start else 0
        end = int(range_end) if range_end else file_size - 1
        # Validate range
        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(
                status_code=416, detail="Requested range not satisfiable"
            )
        content_length = end - start + 1
        def iterfile_range():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    read_size = min(chunk_size, remaining)
                    chunk = f.read(read_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        return StreamingResponse(
            iterfile_range(),
            status_code=206,  # Partial Content
            media_type=mime_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=416, detail=f"Invalid range request: {str(e)}")

def stream_processed_video(
    request: Request, temp_file_path: str, mime_type: str = "video/mp4"
):
    """
    Stream a processed video from a temporary file with range support
    Automatically deletes the temp file after streaming
    """
    try:
        # Get file size
        file_size = os.path.getsize(temp_file_path)
        range_header = request.headers.get("range")
        if not range_header:
            # Stream entire file
            def iterfile():
                try:
                    with open(temp_file_path, "rb") as f:
                        while chunk := f.read(1024 * 1024):
                            yield chunk
                finally:
                    # Clean up temp file after streaming
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
            return StreamingResponse(
                iterfile(),
                media_type=mime_type,
                headers={
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(file_size),
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                },
            )
        # Range request
        range_str = range_header.replace("bytes=", "")
        range_start, range_end = range_str.split("-")
        start = int(range_start) if range_start else 0
        end = int(range_end) if range_end else file_size - 1
        content_length = end - start + 1
        def iterfile_range():
            try:
                with open(temp_file_path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = f.read(min(1024 * 1024, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        return StreamingResponse(
            iterfile_range(),
            status_code=206,
            media_type=mime_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Cache-Control": "no-cache, no-store, must-revalidate",
            },
        )
    except Exception as e:
        # Make sure to clean up on error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise
