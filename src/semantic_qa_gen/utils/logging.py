"""Logging utilities for SemanticQAGen."""

import logging
import sys
from typing import Optional, Dict, Any


def setup_logger(name: str = None, level: str = "INFO",
                log_file: Optional[str] = None,
                log_format: str = None) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional path to log file.
        log_format: Optional log format string.
        
    Returns:
        Configured logger instance.
    """
    # Get logger
    logger = logging.getLogger(name or "semantic_qa_gen")
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Default format includes timestamp, level, and message
    if not log_format:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
