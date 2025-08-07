import logging
import sys
from typing import Optional, Union, TextIO


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on their level."""

    COLORS = {
        logging.DEBUG: Colors.BLUE,
        logging.INFO: Colors.CYAN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.RED + Colors.BOLD,
        25: Colors.GREEN,  # SUCCESS level (between WARNING and INFO)
    }

    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record):
        # Save original format
        orig_format = self._style._fmt

        # Apply color formatting based on log level
        color = self.COLORS.get(record.levelno, Colors.WHITE)
        self._style._fmt = f"{color}{orig_format}{Colors.RESET}"

        # Call the original formatter
        result = super().format(record)

        # Restore original format
        self._style._fmt = orig_format

        return result


class Logger:
    """
    Centralized logger class that provides colored logging functionality.

    This logger is built on top of the standard Python logging module and
    provides convenience methods for different log levels with colored output.
    """

    # Define SUCCESS level between WARNING and INFO
    SUCCESS_LEVEL = 25

    def __init__(
        self,
        name: str = "exo_finder",
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        format_str: str = "%(asctime)s - %(levelname)s - %(message)s",
        date_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        """
        Initialize the logger.

        Args:
            name: Logger name
            level: Minimum log level to display
            log_file: Optional file path to write logs to
            format_str: Log message format string
            date_format: Date format string for log timestamps
        """
        # Register SUCCESS level
        logging.addLevelName(self.SUCCESS_LEVEL, "SUCCESS")

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear any existing handlers
        self.logger.propagate = False  # Prevent propagation to root logger

        # Create console handler with colored formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        colored_formatter = ColoredFormatter(fmt=format_str, datefmt=date_format)
        console_handler.setFormatter(colored_formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if log_file is provided
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(message, *args, **kwargs)

    def success(self, message: str, *args, **kwargs):
        """Log a success message (custom level)."""
        self.logger.log(self.SUCCESS_LEVEL, message, *args, **kwargs)

    def set_level(self, level: Union[int, str]):
        """
        Set the logging level.

        Args:
            level: Either a string (DEBUG, INFO, etc.) or an integer level
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def add_file_handler(self, log_file: str, level: Optional[int] = None):
        """
        Add a file handler to the logger.

        Args:
            log_file: Path to the log file
            level: Optional log level for this handler
        """
        if level is None:
            level = self.logger.level

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)


# Create a default logger instance for easy import
default_logger = Logger(name="exo_finder")


# Convenience functions to use the default logger
def debug(message: str, *args, **kwargs):
    default_logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    default_logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    default_logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    default_logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    default_logger.critical(message, *args, **kwargs)


def success(message: str, *args, **kwargs):
    default_logger.success(message, *args, **kwargs)


def set_level(level: Union[int, str]):
    default_logger.set_level(level)


def add_file_handler(log_file: str, level: Optional[int] = None):
    default_logger.add_file_handler(log_file, level)


def get_logger(
    name: str = "exo_finder",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    date_format: str = "%Y-%m-%d %H:%M:%S",
) -> Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        level: Minimum log level to display
        log_file: Optional file path to write logs to
        format_str: Log message format string
        date_format: Date format string for log timestamps

    Returns:
        A configured Logger instance
    """
    return Logger(
        name=name,
        level=level,
        log_file=log_file,
        format_str=format_str,
        date_format=date_format,
    )
