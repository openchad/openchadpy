import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
@dataclass
class StartupStatus:
    is_ready: bool = False
    current_task: str = "Starting..."
    progress: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    error: Optional[str] = None
    tasks_completed: List[str] = field(default_factory=list)


class StartupHandler(logging.Handler):
    """Logging handler that captures logs for the startup tracker."""

    def __init__(self, tracker: 'StartupTracker'):
        super().__init__()
        self.tracker = tracker
        self.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tracker.add_log(msg)
        except Exception:
            self.handleError(record)

class StartupTracker:
    """Tracks the startup progress and logs of the application."""

    def __init__(self):
        self._status = StartupStatus()
        self._logs: List[str] = []
        self._handler = StartupHandler(self)

    def get_handler(self) -> logging.Handler:
        return self._handler

    def add_log(self, message: str):
        self._logs.append(message)
        # Keep only the last 1000 logs to avoid memory issues
        if len(self._logs) > 1000:
            self._logs.pop(0)

    def update_status(self, task: Optional[str] = None, progress: Optional[float] = None, is_ready: bool = False, error: Optional[str] = None):
        if task:
            self._status.current_task = task
            if task not in self._status.tasks_completed:
                self._status.tasks_completed.append(task)
        if progress is not None:
            self._status.progress = progress
        if is_ready:
            self._status.is_ready = True
            self._status.end_time = time.time()
            self._status.progress = 100.0
        if error:
            self._status.error = error

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": asdict(self._status),
            "logs": self._logs
        }
# Global instance
startup_tracker = StartupTracker()
