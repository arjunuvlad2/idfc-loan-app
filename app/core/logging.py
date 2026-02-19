"""
LOGGING
Structured console logging used by every module via get_logger(__name__).
"""
import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger. Call once at startup in main.py."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Always pass __name__ from the calling module."""
    return logging.getLogger(name)
