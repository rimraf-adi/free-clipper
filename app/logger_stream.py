"""Terminal Log Streamer — captures all stdout, stderr, and logging records into a thread-safe buffer for real-time GUI display.
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from typing import List, Dict, Any

# Strip ANSI color codes for clean GUI presentation
_ANSI_CLEANER = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_LOG_BUFFER: List[Dict[str, Any]] = []
_BUFFER_LOCK = threading.Lock()
MAX_LOG_BUFFER_SIZE = 1000


def add_log_line(text: str, level: str = "INFO") -> None:
    """Adds a log line to the global terminal log buffer."""
    clean_text = _ANSI_CLEANER.sub("", text).strip()
    if not clean_text:
        return

    ts = time.strftime("%H:%M:%S")
    entry = {"time": ts, "level": level, "text": clean_text}

    with _BUFFER_LOCK:
        _LOG_BUFFER.append(entry)
        if len(_LOG_BUFFER) > MAX_LOG_BUFFER_SIZE:
            _LOG_BUFFER.pop(0)


class StreamLogger(object):
    """Wraps sys.stdout or sys.stderr to capture all terminal writes."""

    def __init__(self, original_stream, level: str = "INFO"):
        self.original_stream = original_stream
        self.level = level

    def write(self, buf):
        self.original_stream.write(buf)
        if buf and buf.strip():
            add_log_line(buf, self.level)

    def flush(self):
        self.original_stream.flush()


class GuiLoggingHandler(logging.Handler):
    """Python logging Handler that forwards all log records to the GUI buffer."""

    def emit(self, record):
        try:
            msg = self.format(record)
            add_log_line(msg, record.levelname)
        except Exception:
            self.handleError(record)


# Install global hooks on startup
_installed = False

def install_terminal_log_streamer():
    global _installed
    if _installed:
        return
    _installed = True

    # Redirect sys.stdout and sys.stderr
    sys.stdout = StreamLogger(sys.stdout, "INFO")
    sys.stderr = StreamLogger(sys.stderr, "ERROR")

    # Add logging handler to root logger
    root_logger = logging.getLogger()
    handler = GuiLoggingHandler()
    formatter = logging.Formatter("%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_terminal_logs(limit: int = 300) -> List[Dict[str, Any]]:
    """Returns the most recent terminal log entries."""
    with _BUFFER_LOCK:
        return list(_LOG_BUFFER[-limit:])
