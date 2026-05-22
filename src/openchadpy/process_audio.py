from fastapi.responses import FileResponse, Response
import ffmpeg
import os
import tempfile
import mimetypes

def process_audio(file_path: str, bitrate: str = None, format: str = None):
    """
    Process audio files (convert format, change bitrate)
    Args:
        file_path: Absolute path to the audio file
        bitrate: Target bitrate (e.g., "128k", "320k", "192k")
        format: Output format (mp3, wav, ogg, m4a, flac, etc.)
    Requires: ffmpeg-python (pip install ffmpeg-python)
    Note: Also requires FFmpeg to be installed on your system
    """
    try:
        # If no processing needed, return original
        if not bitrate and not format:
            return FileResponse(
                file_path,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        # Determine output format
        _, input_ext = os.path.splitext(file_path)
        output_format = format if format else input_ext[1:]  # Remove dot from extension
        # Create temporary output file
        with tempfile.NamedTemporaryFile(
            suffix=f".{output_format}", delete=False
        ) as tmp_file:
            output_path = tmp_file.name
        try:
            # Build ffmpeg command
            stream = ffmpeg.input(file_path)
            # Set output options
            output_options = {}
            if bitrate:
                output_options["audio_bitrate"] = bitrate
            # Process the audio
            stream = ffmpeg.output(stream, output_path, **output_options)
            ffmpeg.run(
                stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
            )
            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(output_path)
            if not mime_type:
                mime_types = {
                    "mp3": "audio/mpeg",
                    "wav": "audio/wav",
                    "ogg": "audio/ogg",
                    "m4a": "audio/mp4",
                    "flac": "audio/flac",
                    "aac": "audio/aac",
                }
                mime_type = mime_types.get(output_format, "audio/mpeg")
            # Read processed file
            with open(output_path, "rb") as f:
                audio_data = f.read()
            return Response(
                content=audio_data,
                media_type=mime_type,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                os.unlink(output_path)
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise Exception(f"FFmpeg error processing audio: {error_message}")
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")
# Alternative: Simpler version without FFmpeg (returns original file)

def process_audio_simple(file_path: str, bitrate: str = None, format: str = None):
    """
    Simple audio handler - just serves the file
    Use this if you don't want to install FFmpeg
    """
    return FileResponse(
        file_path,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
