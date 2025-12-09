"""Centralized logging configuration for the smellai project.

This module provides a consistent logging setup for both application code and tests.
Import and call setup_logging() at the start of your application or in test fixtures.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_format: str | None = None,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure logging for the application.

    Args:
        level: Logging level (default: logging.INFO)
        log_format: Custom log format string (optional)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance

    Example:
        >>> from logging_config import setup_logging
        >>> logger = setup_logging()
        >>> logger.info("Application started")
    """
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure handlers
    handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Return a logger for the caller
    return logging.getLogger(__name__)


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Logger instance

    Example:
        >>> from logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)
