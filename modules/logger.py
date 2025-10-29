"""Logging configuration for the processing pipeline with color support."""

import logging
import sys
from pathlib import Path
from typing import Optional
from tqdm import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    """Custom handler that uses tqdm.write() to avoid interrupting progress bars."""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, fmt: str, datefmt: str = None, use_colors: bool = True):
        """
        Initialize colored formatter.
        
        Args:
            fmt: Log format string
            datefmt: Date format string
            use_colors: Enable color output
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with colors.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log string
        """
        if self.use_colors and record.levelname in self.COLORS:
            # Add color to level name
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
            )
        
        return super().format(record)


def setup_logger(config: Optional[dict] = None) -> logging.Logger:
    """
    Setup and configure logger with file and console handlers.
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        Configured logger instance
    """
    # Default configuration
    if config is None:
        log_level = "INFO"
        log_file = None
        log_format = "[%(asctime)s] [%(levelname)s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        colored_output = True
    else:
        logging_config = config.get('logging', {})
        log_level = logging_config.get('log_level', 'INFO')
        log_file = logging_config.get('log_file')
        log_format = logging_config.get('log_format', "[%(asctime)s] [%(levelname)s] %(message)s")
        date_format = logging_config.get('date_format', "%Y-%m-%d %H:%M:%S")
        colored_output = logging_config.get('colored_output', True)
    
    # Create logger
    logger = logging.getLogger('gesture_processing')
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    logger.propagate = False
    
    # Console handler with tqdm support
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = ColoredFormatter(log_format, datefmt=date_format, use_colors=colored_output)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Use colored formatter for console
    console_formatter = ColoredFormatter(
        log_format, 
        datefmt=date_format, 
        use_colors=colored_output
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified) - no colors for file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Regular formatter for file (no colors)
        file_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
