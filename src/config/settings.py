"""
Configuration management for the AI Assistant System.
Handles loading settings from environment variables and .env files.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class Settings:
    """Centralized configuration management."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize settings with optional custom env file."""
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load .env from project root
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                load_dotenv()  # Load from system environment
        
        self._load_settings()
    
    def _load_settings(self):
        """Load all configuration settings."""
        # API Configuration
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
        self.GEMINI_API_KEY_SECONDARY = os.getenv("GEMINI_API_KEY_SECONDARY", "")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{self.GEMINI_MODEL}:generateContent"
        
        # System Configuration
        self.MAX_HISTORY = int(os.getenv("MAX_HISTORY", "100"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Database Configuration
        self.DB_PATH = os.getenv("DB_PATH", "db/agent_memory.db")
        
        # Directory Configuration
        self.BASE_DIR = Path(os.getenv("BASE_DIR", ".")).resolve()
        self.TOOLS_DIR = self.BASE_DIR / os.getenv("TOOLS_DIR", "tools")
        self.TEMP_DIR = self.BASE_DIR / os.getenv("TEMP_DIR", "temp")
        self.TEST_DIR = self.BASE_DIR / os.getenv("TEST_DIR", "test")
        self.OUTPUT_DIR = self.BASE_DIR / os.getenv("OUTPUT_DIR", "output")
        self.LOG_DIR = self.BASE_DIR / os.getenv("LOG_DIR", "log")
        
        # Voice Configuration
        self.VOICE_ENABLED = os.getenv("VOICE_ENABLED", "false").lower() == "true"
        self.TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "en")
        self.TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
        
        # Google Search Configuration (uses Gemini API key)
        # No separate API key needed - uses GEMINI_API_KEY
        
        # Context Management Configuration
        self.CONTEXT_CACHE_SIZE = int(os.getenv("CONTEXT_CACHE_SIZE", "1000"))
        self.CONTEXT_MAX_LENGTH = int(os.getenv("CONTEXT_MAX_LENGTH", "50000"))
        self.SESSION_PERSISTENCE = os.getenv("SESSION_PERSISTENCE", "true").lower() == "true"
        
        # Gemini API Advanced Configuration
        self.GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
        self.GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
        self.GEMINI_TOP_P = float(os.getenv("GEMINI_TOP_P", "0.95"))
        self.GEMINI_TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))
        self.GEMINI_SAFETY_SETTINGS = os.getenv("GEMINI_SAFETY_SETTINGS", "moderate")
        
        # Development Configuration
        self.DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
        self.VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
        
        # Ensure all necessary directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.BASE_DIR,
            self.TOOLS_DIR,
            self.TEMP_DIR,
            self.TEST_DIR,
            self.OUTPUT_DIR,
            self.LOG_DIR,
            Path(self.DB_PATH).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_api_keys(self) -> bool:
        """Validate that at least one API key is provided."""
        return bool(self.GEMINI_API_KEY or self.GEMINI_API_KEY_SECONDARY)
    
    def get_api_key(self, use_secondary: bool = False) -> str:
        """Get the appropriate API key."""
        if use_secondary and self.GEMINI_API_KEY_SECONDARY:
            return self.GEMINI_API_KEY_SECONDARY
        return self.GEMINI_API_KEY
    
    def __repr__(self):
        """String representation of settings (without sensitive data)."""
        return f"Settings(BASE_DIR={self.BASE_DIR}, VOICE_ENABLED={self.VOICE_ENABLED}, DEBUG_MODE={self.DEBUG_MODE})"


# Global settings instance
settings = Settings()
