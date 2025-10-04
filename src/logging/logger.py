"""
Advanced logging system for the AI Assistant System.
Provides colored console output and file logging capabilities.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import settings


class Logger:
    """Advanced logger with colored console output and file logging."""
    
    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'BOLD_RED': '\033[1;91m',
        'BOLD_GREEN': '\033[1;92m',
        'BOLD_YELLOW': '\033[1;93m',
        'BOLD_BLUE': '\033[1;94m',
        'BOLD_MAGENTA': '\033[1;95m',
        'BOLD_CYAN': '\033[1;96m',
    }
    
    def __init__(self, name: str = "ai_assistant", log_dir: Optional[Path] = None):
        """Initialize logger with optional custom log directory."""
        self.name = name
        self.log_dir = log_dir or settings.LOG_DIR
        self.log_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.verbose = settings.VERBOSE_LOGGING
        self.debug_mode = settings.DEBUG_MODE
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _write_to_file(self, msg: str):
        """Write message to log file."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {msg}\n")
        except Exception as e:
            print(f"Failed to write to log file: {e}")
    
    def _write_error_to_file(self, msg: str, exc_info: bool = False):
        """Write error message to log file with full traceback."""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] {msg}\n")
                if exc_info:
                    import traceback
                    f.write(f"[{datetime.now().isoformat()}] TRACEBACK:\n")
                    f.write(traceback.format_exc())
                    f.write(f"[{datetime.now().isoformat()}] END TRACEBACK\n")
        except Exception as e:
            print(f"Failed to write error to log file: {e}")
    
    def _format_message(self, level: str, message: str, color: str = None) -> str:
        """Format message with timestamp and optional color."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        if color and sys.stdout.isatty():
            return f"{color}{formatted_msg}{self.COLORS['RESET']}"
        return formatted_msg
    
    def info(self, msg: str):
        """Log info message."""
        log_msg = self._format_message("INFO", msg, self.COLORS['BLUE'])
        print(log_msg)
        self._write_to_file(f"[INFO] {msg}")
    
    def success(self, msg: str):
        """Log success message."""
        log_msg = self._format_message("SUCCESS", msg, self.COLORS['GREEN'])
        print(log_msg)
        self._write_to_file(f"[SUCCESS] {msg}")
    
    def error(self, msg: str, exc_info: bool = False):
        """Log error message."""
        log_msg = self._format_message("ERROR", msg, self.COLORS['RED'])
        print(log_msg)
        self._write_error_to_file(f"[ERROR] {msg}", exc_info)
    
    def warning(self, msg: str):
        """Log warning message."""
        log_msg = self._format_message("WARNING", msg, self.COLORS['YELLOW'])
        print(log_msg)
        self._write_to_file(f"[WARNING] {msg}")
    
    def debug(self, msg: str):
        """Log debug message (only if debug mode is enabled)."""
        if self.debug_mode or self.verbose:
            log_msg = self._format_message("DEBUG", msg, self.COLORS['MAGENTA'])
            print(log_msg)
        self._write_to_file(f"[DEBUG] {msg}")
    
    def step(self, step_num: int, total: int, desc: str):
        """Log step information."""
        log_msg = f"[STEP {step_num}/{total}]: {desc}"
        formatted_msg = self._format_message("STEP", log_msg, self.COLORS['BOLD_YELLOW'])
        print(f"\n{formatted_msg}")
        self._write_to_file(f"[STEP {step_num}/{total}] {desc}")
    
    def critical(self, msg: str):
        """Log critical error message."""
        log_msg = self._format_message("CRITICAL", msg, self.COLORS['BOLD_RED'])
        print(log_msg)
        self._write_to_file(f"[CRITICAL] {msg}")
    
    def exception(self, msg: str, exc_info: bool = True):
        """Log exception with traceback."""
        log_msg = self._format_message("EXCEPTION", msg, self.COLORS['RED'])
        print(log_msg)
        if exc_info:
            import traceback
            traceback.print_exc()
        self._write_error_to_file(f"[EXCEPTION] {msg}", exc_info)
    
    def separator(self, char: str = "=", length: int = 50):
        """Print a visual separator."""
        separator_line = char * length
        if sys.stdout.isatty():
            print(f"{self.COLORS['DIM']}{separator_line}{self.COLORS['RESET']}")
        else:
            print(separator_line)
        self._write_to_file(separator_line)
    
    def banner(self, title: str):
        """Print a banner with title."""
        self.separator("=", len(title) + 4)
        if sys.stdout.isatty():
            print(f"{self.COLORS['BOLD_CYAN']}  {title}  {self.COLORS['RESET']}")
        else:
            print(f"  {title}  ")
        self.separator("=", len(title) + 4)
        self._write_to_file(f"BANNER: {title}")


# Global logger instance
logger = Logger()
