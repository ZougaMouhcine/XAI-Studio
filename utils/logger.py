"""
XAI Studio — Centralized Logging
=================================
Provides a configured logger that can be shared across all modules.
Also supports a callback mechanism so the UI status bar can display log messages.
"""

import logging
from config.settings import LOG_FORMAT, LOG_DATE_FORMAT


# ---------------------------------------------------------------------------
# Callback handler — forwards log records to a registered callback (e.g. UI)
# ---------------------------------------------------------------------------
class CallbackHandler(logging.Handler):
    """Custom logging handler that forwards messages to a registered callback."""

    def __init__(self):
        super().__init__()
        self._callback = None

    def set_callback(self, callback):
        """Register a callback function: callback(level: str, message: str)."""
        self._callback = callback

    def emit(self, record):
        if self._callback is not None:
            try:
                msg = self.format(record)
                self._callback(record.levelname, msg)
            except Exception:
                self.handleError(record)


# Singleton handler shared across the application
_callback_handler = CallbackHandler()
_callback_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger pre-configured with console + callback handlers."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Console output
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(console)

        # Callback output (for UI status bar)
        logger.addHandler(_callback_handler)

    return logger


def set_ui_callback(callback):
    """
    Register a UI callback to receive log messages in real time.

    Parameters
    ----------
    callback : callable(level: str, message: str)
    """
    _callback_handler.set_callback(callback)
