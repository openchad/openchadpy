import os
import json
from typing import Dict, Any, Optional, List, Set
import aiofiles
import filetype
import logging
logger = logging.getLogger(__name__)

def get_file_path(filename: str, base_dir: str = ".") -> str:
    """Get the full path for a file relative to base directory"""
    if os.path.isabs(filename):
        return filename
    return os.path.normpath(os.path.join(base_dir, filename))

def get_folder_path(path: str, base_dir: str = ".") -> str:
    """Get the full path for a folder, handling absolute vs relative paths"""
    if os.path.isabs(path) or (len(path) >= 2 and path[1] == ":"):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(base_dir, path))

def get_file_mtime(filepath: str) -> float:
    """Get the modification time of a file"""
    try:
        if os.path.exists(filepath):
            return os.path.getmtime(filepath)
    except Exception as e:
        logger.error(f"Error getting mtime for {filepath}: {e}")
    return 0.0

async def ensure_file_exists_async(filepath: str, initial_content: str = "") -> bool:
    """Ensure a file exists (async), creating it with initial content if not"""
    try:
        dir_path = os.path.dirname(filepath)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if not os.path.exists(filepath):
            async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
                await f.write(initial_content)
            return True  # File was created
        return False  # File already existed
    except Exception as e:
        logger.error(f"Error ensuring file exists {filepath}: {e}")
        return False

def detect_media_type(filepath: str) -> str:
    """Detect media type using filetype library"""
    try:
        kind = filetype.guess(filepath)
        if kind:
            return kind.mime
        # Fallback to extensions if filetype fails
        ext = os.path.splitext(filepath)[1].lower()
        extension_map = {
            ".pcm": "audio/pcm",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".txt": "text/plain",
            ".json": "application/json",
            ".md": "text/markdown"
        }
        return extension_map.get(ext, "application/octet-stream")
    except Exception as e:
        logger.error(f"Error detecting media type for {filepath}: {e}")
        return "application/octet-stream"

async def should_read_as_text_async(filepath: str) -> bool:
    """Determine if a file should be read as text (async)"""
    try:
        if not os.path.exists(filepath):
            return False
        async with aiofiles.open(filepath, mode="rb") as f:
            chunk = await f.read(8192)
        if not chunk:
            return True
        if b"\x00" in chunk:
            return False
        try:
            chunk.decode("utf-8")
            return True
        except UnicodeDecodeError:
            for encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    chunk.decode(encoding)
                    return True
                except UnicodeDecodeError:
                    continue
            return False
    except Exception as e:
        logger.error(f"Error checking if text file {filepath}: {e}")
        return False

def list_folder_contents(folder_path: str, recursive: bool = True) -> List[str]:
    """List all files and folders in a directory"""
    try:
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return []
        results: List[str] = []
        if recursive:
            for root, dirs, files in os.walk(folder_path):
                rel_root = os.path.relpath(root, folder_path)
                if rel_root == ".":
                    rel_root = ""
                else:
                    rel_root = rel_root.replace("\\", "/") + "/"
                for f in files:
                    results.append(rel_root + f)
                for d in dirs:
                    results.append(rel_root + d + "/")
        else:
            for item in os.listdir(folder_path):
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    results.append(item + "/")
                else:
                    results.append(item)
        return sorted(results)
    except Exception as e:
        logger.error(f"Error listing folder {folder_path}: {e}")
        return []

async def file_handler(data: dict) -> dict:
    """Handle file operations and return response (async)"""
    try:
        command = data.get("command")
        filename = data.get("filename")
        project_dir = os.environ.get("OPENCHAD_PROJECT_DIR")
        base_dir = data.get("base_dir") or "."
        if project_dir:
            base_dir = os.path.join(project_dir, base_dir)

        if not filename:
            return {"error": "Filename required"}
        filepath = get_file_path(filename, base_dir)
        if command == "read":
            await ensure_file_exists_async(filepath, data.get("initial_content", ""))
            if await should_read_as_text_async(filepath):
                async with aiofiles.open(filepath, mode="r", encoding="utf-8") as f:
                    content = await f.read()
                return {
                    "data": {
                        "content": content,
                        "mtime": get_file_mtime(filepath),
                        "exists": True,
                        "mime_type": "text/plain",
                    }
                }
            else:
                mime_type = detect_media_type(filepath)
                return {
                    "data": {
                        "content": os.path.abspath(filepath),
                        "mtime": get_file_mtime(filepath),
                        "exists": True,
                        "mime_type": mime_type,
                    }
                }
        elif command == "write":
            content = data.get("content", "")
            dir_path = os.path.dirname(filepath)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
            async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
                await f.write(content)
            return {"data": {"status": "written", "mtime": get_file_mtime(filepath)}}
        elif command == "exists":
            exists = os.path.exists(filepath)
            return {
                "data": {
                    "exists": exists,
                    "mtime": get_file_mtime(filepath) if exists else 0,
                }
            }
        elif command == "delete":
            if os.path.exists(filepath):
                os.remove(filepath)
                return {"data": {"status": "deleted"}}
            return {"data": {"status": "not_found"}}
        elif command == "mtime":
            return {
                "data": {
                    "mtime": get_file_mtime(filepath),
                    "exists": os.path.exists(filepath),
                }
            }
        else:
            return {"error": f"Unknown command: {command}"}
    except Exception as e:
        logger.error(f"Error in file_handler: {e}", exc_info=True)
        return {"error": str(e)}

async def folder_handler(data: dict) -> dict:
    """Handle folder operations and return response (async)"""
    try:
        command = data.get("command")
        path = data.get("path")
        project_dir = os.environ.get("OPENCHAD_PROJECT_DIR")
        base_dir = data.get("base_dir") or "."
        if project_dir:
            base_dir = os.path.join(project_dir, base_dir)

        recursive = data.get("recursive", True)
        if not path:
            return {"error": "Path required"}
        folder_path = get_folder_path(path, base_dir)
        logger.info(f"Folder path: {os.path.abspath(folder_path)}")
        if command == "list":
            contents = list_folder_contents(folder_path, recursive)
            exists = os.path.exists(folder_path) and os.path.isdir(folder_path)
            return {
                "data": {
                    "contents": contents,
                    "exists": exists,
                    "path": folder_path,
                }
            }
        elif command == "exists":
            exists = os.path.exists(folder_path) and os.path.isdir(folder_path)
            return {
                "data": {
                    "exists": exists,
                    "path": folder_path,
                }
            }
        elif command == "create":
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
                return {"data": {"status": "created", "path": folder_path}}
            return {"data": {"status": "already_exists", "path": folder_path}}
        else:
            return {"error": f"Unknown command: {command}"}
    except Exception as e:
        logger.error(f"Error in folder_handler: {e}", exc_info=True)
        return {"error": str(e)}
