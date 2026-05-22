from fastapi import Request
import ffmpeg
import os
import tempfile
import mimetypes

def process_video(
    file_path: str, resolution: str = None, fps: int = None, format: str = None #pyrefly: ignore
):
    """
    Process video files (resize, change fps, convert format)
    Returns the path to the processed temp file for streaming
    Supports all FFmpeg-compatible formats including:
    - Container formats: MP4, MKV, AVI, MOV, WebM, FLV, WMV, MPEG, 3GP, etc.
    - Codecs: H.264, H.265/HEVC, VP8, VP9, AV1, etc.
    Args:
        file_path: Absolute path to the video file
        resolution: Target resolution (e.g., "720p", "1080p", "480p" or "1280x720")
        fps: Target frames per second
        format: Output format (mp4, webm, avi, mkv, mov, flv, wmv, etc.)
    Returns:
        Tuple of (temp_file_path, mime_type) for streaming
    Requires: ffmpeg-python (pip install ffmpeg-python)
    Note: Also requires FFmpeg to be installed on your system
    """
    try:
        # Determine output format
        _, input_ext = os.path.splitext(file_path)
        output_format = format if format else input_ext[1:]  # Remove dot from extension
        # Create temporary output file (don't auto-delete, we'll stream it)
        temp_fd, output_path = tempfile.mkstemp(suffix=f".{output_format}")
        os.close(temp_fd)  # Close file descriptor, we'll use the path
        # Build ffmpeg command
        stream = ffmpeg.input(file_path) #pyrefly: ignore
        # Parse resolution
        if resolution:
            # Convert common resolution strings to dimensions
            resolution_map = {
                "480p": "854x480",
                "720p": "1280x720",
                "1080p": "1920x1080",
                "1440p": "2560x1440",
                "4k": "3840x2160",
                "2160p": "3840x2160",
            }
            target_resolution = resolution_map.get(resolution.lower(), resolution)
            # Apply scale filter
            stream = ffmpeg.filter(stream, "scale", target_resolution) #pyrefly: ignore
        # Set output options
        output_options = {}
        # Set FPS if provided
        if fps:
            output_options["r"] = fps
        # Video codec options based on output format
        codec_map = {
            "mp4": {"vcodec": "libx264", "acodec": "aac"},
            "webm": {"vcodec": "libvpx-vp9", "acodec": "libopus"},
            "mkv": {"vcodec": "libx264", "acodec": "aac"},  # MKV supports many codecs
            "avi": {"vcodec": "mpeg4", "acodec": "mp3"},
            "mov": {"vcodec": "libx264", "acodec": "aac"},
            "flv": {"vcodec": "flv", "acodec": "mp3"},
        }
        # Apply codec settings if format is specified
        if output_format in codec_map:
            output_options.update(codec_map[output_format]) #pyrefly: ignore
        else:
            # Default to H.264 + AAC for unknown formats
            output_options["vcodec"] = "libx264" #pyrefly: ignore
            output_options["acodec"] = "aac" #pyrefly: ignore
        # Process the video
        stream = ffmpeg.output(stream, output_path, **output_options) #pyrefly: ignore
        ffmpeg.run( #pyrefly: ignore
            stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
        )
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(output_path)
        if not mime_type:
            mime_types = {
                "mp4": "video/mp4",
                "webm": "video/webm",
                "avi": "video/x-msvideo",
                "mkv": "video/x-matroska",
                "mov": "video/quicktime",
                "flv": "video/x-flv",
                "wmv": "video/x-ms-wmv",
                "mpg": "video/mpeg",
                "mpeg": "video/mpeg",
                "m4v": "video/x-m4v",
                "3gp": "video/3gpp",
                "ogv": "video/ogg",
                "ts": "video/mp2t",
            }
            mime_type = mime_types.get(output_format, "video/mp4")
        return output_path, mime_type
    except ffmpeg.Error as e: #pyrefly: ignore
        # Clean up temp file on error
        if "output_path" in locals() and os.path.exists(output_path): #pyrefly: ignore
            os.unlink(output_path)
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise Exception(f"FFmpeg error processing video: {error_message}")
    except Exception as e:
        # Clean up temp file on error
        if "output_path" in locals() and os.path.exists(output_path): #pyrefly: ignore
            os.unlink(output_path)
        raise Exception(f"Error processing video: {str(e)}")
# Generate thumbnail from video

def generate_video_thumbnail(file_path: str, time: str = "00:00:01"):
    """
    Generate a thumbnail image from a video at a specific time
    Args:
        file_path: Absolute path to the video file
        time: Time position for thumbnail (format: HH:MM:SS or seconds)
    Returns: Tuple of (temp_file_path, mime_type) for the thumbnail
    """
    try:
        temp_fd, output_path = tempfile.mkstemp(suffix=".jpg")
        os.close(temp_fd)
        # Extract single frame at specified time
        stream = ffmpeg.input(file_path, ss=time) #pyrefly: ignore
        stream = ffmpeg.output(stream, output_path, vframes=1) #pyrefly: ignore
        ffmpeg.run( #pyrefly: ignore
            stream, overwrite_output=True, capture_stdout=True, capture_stderr=True
        )
        return output_path, "image/jpeg"
    except Exception as e:
        if "output_path" in locals() and os.path.exists(output_path): #pyrefly: ignore
            os.unlink(output_path)
        raise Exception(f"Error generating video thumbnail: {str(e)}")
