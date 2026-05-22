import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Set, List
from watchfiles import awatch
from .file import get_file_path, get_folder_path, get_file_mtime, list_folder_contents
from .event_emitter import event_emitter
logger = logging.getLogger(__name__)
# File subscription tracking: {conn_id: set of file paths}
file_subscriptions: Dict[str, Set[str]] = {}
# Reverse mapping: {file_path: set of conn_ids}
file_watchers: Dict[str, Set[str]] = {}
# File watcher tasks: {file_path: asyncio.Task}
file_watcher_tasks: Dict[str, asyncio.Task] = {}
# Folder subscription tracking: {conn_id: set of folder paths}
folder_subscriptions: Dict[str, Set[str]] = {}
# Reverse mapping: {folder_path: set of conn_ids}
folder_watchers: Dict[str, Set[str]] = {}
# Folder watcher tasks: {folder_path: asyncio.Task}
folder_watcher_tasks: Dict[str, asyncio.Task] = {}

# File Subscription Functions
async def watch_file_task(file_path: str, file_key: str):
    """Watch a single file for changes using watchfiles"""
    try:
        # Get the directory containing the file
        dir_path = os.path.dirname(file_path) or "."
        filename = os.path.basename(file_path)
        # Debounce set to 500ms to batch rapid changes
        async for changes in awatch(dir_path, recursive=False, debounce=500):
            for change_type, changed_path in changes:
                # Check if this change is for our file
                if (
                    os.path.basename(changed_path) == filename
                    or changed_path == file_path
                ):
                    timestamp = (
                        await asyncio.to_thread(get_file_mtime, file_path) if os.path.exists(file_path) else 0.0
                    )
                    await notify_file_change(file_key, timestamp)
    except asyncio.CancelledError:
        logger.info(f"File watcher cancelled for {file_key}")
    except Exception as e:
        logger.error(f"Error in file watcher for {file_key}: {e}")

async def subscribe_file(conn_id: str, filename: str, base_dir: str = "."):
    """Subscribe a websocket to file changes using watchfiles"""
    file_path = get_file_path(filename, base_dir)
    file_key = file_path  # Use absolute path as key
    # Add to subscriptions
    if conn_id not in file_subscriptions:
        file_subscriptions[conn_id] = set()
    file_subscriptions[conn_id].add(file_key)
    # Add to watchers and start watcher task if first subscriber
    if file_key not in file_watchers:
        file_watchers[file_key] = set()
        # Start the watchfiles task
        task = asyncio.create_task(watch_file_task(file_path, file_key))
        file_watcher_tasks[file_key] = task
        logger.info(f"Started watchfiles for {file_key}")
    file_watchers[file_key].add(conn_id)
    logger.info(f"Subscribed {conn_id} to file {file_key}")
    return {"status": "subscribed", "filename": filename, "path": file_path}

async def unsubscribe_file(conn_id: str, filename: str, base_dir: str = "."):
    """Unsubscribe a websocket from file changes"""
    file_path = get_file_path(filename, base_dir)
    file_key = file_path
    if conn_id in file_subscriptions:
        file_subscriptions[conn_id].discard(file_key)
        if not file_subscriptions[conn_id]:
            del file_subscriptions[conn_id]
    if file_key in file_watchers:
        file_watchers[file_key].discard(conn_id)
        if not file_watchers[file_key]:
            del file_watchers[file_key]
            # Cancel the watcher task
            if file_key in file_watcher_tasks:
                file_watcher_tasks[file_key].cancel()
                del file_watcher_tasks[file_key]
                logger.info(f"Stopped watchfiles for {file_key}")
    logger.info(f"Unsubscribed {conn_id} from file {file_key}")
    return {"status": "unsubscribed", "filename": filename, "path": file_path}

async def notify_file_change(file_key: str, timestamp: float):
    """Notify all subscribers of a file change"""
    if file_key not in file_watchers:
        return
    event_name = f"file_changed:{file_key}"
    data = {
        "timestamp": int(timestamp * 1000),  # Convert to milliseconds
        "exists": os.path.exists(file_key),
    }
    conn_ids = list(file_watchers[file_key])
    # event_emitter handles both Tauri (Emitter.emit_str) and WebSocket modes
    disconnected = await event_emitter.emit(event_name, data, conn_ids=conn_ids)
    cleanup_disconnected_files(disconnected)

def cleanup_disconnected_files(disconnected_ids: list[str]):
    """Remove disconnected websockets from file subscriptions"""
    for conn_id in disconnected_ids:
        if conn_id in file_subscriptions:
            for file_path in file_subscriptions[conn_id]:
                if file_path in file_watchers:
                    file_watchers[file_path].discard(conn_id)
                    if not file_watchers[file_path]:
                        del file_watchers[file_path]
                        # Cancel the watcher task
                        if file_path in file_watcher_tasks:
                            file_watcher_tasks[file_path].cancel()
                            del file_watcher_tasks[file_path]
            del file_subscriptions[conn_id]
# Folder Subscription Functions

async def watch_folder_task(folder_path: str):
    """Watch a folder for changes using watchfiles"""
    try:
        while True:
            if os.path.exists(folder_path):
                # Notify immediately that it exists now (and send current contents)
                contents = await asyncio.to_thread(list_folder_contents, folder_path)
                timestamp = datetime.now().timestamp()
                await notify_folder_change(folder_path, timestamp, contents)
                # Watch the folder itself
                # Debounce set to 500ms to batch rapid changes
                async for changes in awatch(folder_path, debounce=500):
                    # Get updated folder contents
                    contents = await asyncio.to_thread(list_folder_contents, folder_path)
                    timestamp = datetime.now().timestamp()
                    await notify_folder_change(folder_path, timestamp, contents)
                    # If folder is deleted, break to handle re-creation
                    if not os.path.exists(folder_path):
                        break
            else:
                # Folder doesn't exist.
                # Notify that it doesn't exist
                timestamp = datetime.now().timestamp()
                await notify_folder_change(folder_path, timestamp, [])
                # Watch parent directory until folder is created
                parent_dir = os.path.dirname(folder_path) or "."
                if not os.path.exists(parent_dir):
                    # If parent doesn't exist, just poll
                    await asyncio.sleep(1)
                    continue
                # Watch parent for creation of our folder
                async for changes in awatch(parent_dir, recursive=False, debounce=500):
                    if os.path.exists(folder_path):
                        break
                    if not os.path.exists(parent_dir):
                        break
    except asyncio.CancelledError:
        logger.info(f"Folder watcher cancelled for {folder_path}")
    except Exception as e:
        logger.error(f"Error in folder watcher for {folder_path}: {e}")
        await asyncio.sleep(1)

async def subscribe_folder(conn_id: str, path: str, base_dir: str = "."):
    """Subscribe a websocket to folder changes using watchfiles"""
    folder_path = get_folder_path(path, base_dir)
    # Add to subscriptions
    if conn_id not in folder_subscriptions:
        folder_subscriptions[conn_id] = set()
    folder_subscriptions[conn_id].add(folder_path)
    # Add to watchers and start watcher task if first subscriber
    if folder_path not in folder_watchers:
        folder_watchers[folder_path] = set()
        # Start the watchfiles task
        task = asyncio.create_task(watch_folder_task(folder_path))
        folder_watcher_tasks[folder_path] = task
        logger.info(f"Started watchfiles for folder {folder_path}")
    folder_watchers[folder_path].add(conn_id)
    # Get initial folder contents
    contents = await asyncio.to_thread(list_folder_contents, folder_path)
    exists = await asyncio.to_thread(os.path.exists, folder_path) and await asyncio.to_thread(os.path.isdir, folder_path)
    logger.info(f"Subscribed {conn_id} to folder {folder_path}")
    return {
        "status": "subscribed",
        "path": folder_path,
        "contents": contents,
        "exists": exists,
    }

async def unsubscribe_folder(conn_id: str, path: str, base_dir: str = "."):
    """Unsubscribe a websocket from folder changes"""
    folder_path = get_folder_path(path, base_dir)
    if conn_id in folder_subscriptions:
        folder_subscriptions[conn_id].discard(folder_path)
        if not folder_subscriptions[conn_id]:
            del folder_subscriptions[conn_id]
    if folder_path in folder_watchers:
        folder_watchers[folder_path].discard(conn_id)
        if not folder_watchers[folder_path]:
            del folder_watchers[folder_path]
            # Cancel the watcher task
            if folder_path in folder_watcher_tasks:
                folder_watcher_tasks[folder_path].cancel()
                del folder_watcher_tasks[folder_path]
                logger.info(f"Stopped watchfiles for folder {folder_path}")
    logger.info(f"Unsubscribed {conn_id} from folder {folder_path}")
    return {"status": "unsubscribed", "path": folder_path}

async def notify_folder_change(folder_path: str, timestamp: float, contents: List[str]):
    """Notify all subscribers of a folder change"""
    if folder_path not in folder_watchers:
        return
    event_name = f"folder_changed:{folder_path}"
    data = {
        "timestamp": int(timestamp * 1000),  # Convert to milliseconds
        "contents": contents,
        "exists": os.path.exists(folder_path) and os.path.isdir(folder_path),
    }
    conn_ids = list(folder_watchers[folder_path])
    # event_emitter handles both Tauri (Emitter.emit_str) and WebSocket modes
    disconnected = await event_emitter.emit(event_name, data, conn_ids=conn_ids)
    cleanup_disconnected_folders(disconnected)

def cleanup_disconnected_folders(disconnected_ids: list[str]):
    """Remove disconnected websockets from folder subscriptions"""
    for conn_id in disconnected_ids:
        if conn_id in folder_subscriptions:
            for folder_path in folder_subscriptions[conn_id]:
                if folder_path in folder_watchers:
                    folder_watchers[folder_path].discard(conn_id)
                    if not folder_watchers[folder_path]:
                        del folder_watchers[folder_path]
                        # Cancel the watcher task
                        if folder_path in folder_watcher_tasks:
                            folder_watcher_tasks[folder_path].cancel()
                            del folder_watcher_tasks[folder_path]
            del folder_subscriptions[conn_id]

def remove_id(conn_id: str):
    """Clean up subscriptions for a disconnected websocket ID"""
    cleanup_disconnected_files([conn_id])
    cleanup_disconnected_folders([conn_id])