"""
Configuration utilities for the NazareAI Framework.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Global settings loaded from environment variables."""
    
    # API Keys
    openrouter_api_key: str = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    serpapi_api_key: str = Field(default_factory=lambda: os.getenv("SERPAPI_API_KEY"))
    replicate_api_token: Optional[str] = Field(default_factory=lambda: os.getenv("REPLICATE_API_TOKEN"))
    
    # Model Configuration
    default_model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "anthropic/claude-3.5-sonnet:beta")
    )
    openrouter_api_base: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
    )
    
    # Application Settings
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    max_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_RETRIES", "3"))
    )
    timeout: float = Field(
        default_factory=lambda: float(os.getenv("TIMEOUT", "30.0"))
    )
    
    # Web Interface
    host: str = Field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0")
    )
    port: int = Field(
        default_factory=lambda: int(os.getenv("PORT", "8000"))
    )
    cors_origins: str = Field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "*")
    )

    def __init__(self, **kwargs):
        """Initialize settings and validate required fields."""
        super().__init__(**kwargs)
        
        if not self.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        if not self.serpapi_api_key:
            raise ValueError("SERPAPI_API_KEY environment variable is required")


def load_settings(env_file: Optional[str] = None) -> Settings:
    """
    Load settings from environment variables and .env file.

    Args:
        env_file: Optional path to .env file

    Returns:
        Settings instance
    """
    # Find .env file
    if env_file:
        env_path = Path(env_file)
    else:
        # Try to find .env file in current or parent directories
        current_dir = Path.cwd()
        env_path = current_dir / ".env"
        
        if not env_path.exists():
            parent_dir = current_dir.parent
            env_path = parent_dir / ".env"

    # Load environment variables from .env file if it exists
    if env_path.exists():
        print(f"Loading environment variables from {env_path}")
        load_dotenv(env_path, override=True)
    else:
        print("No .env file found")

    # Configure logging based on DEBUG setting
    debug = os.getenv("DEBUG", "false").lower() == "true"
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="DEBUG" if debug else "INFO")

    # Print environment variables for debugging
    print("\nEnvironment variables:")
    print(f"OPENROUTER_API_KEY: {'*' * 8}{os.getenv('OPENROUTER_API_KEY')[-4:] if os.getenv('OPENROUTER_API_KEY') else 'Not set'}")
    print(f"SERPAPI_API_KEY: {'*' * 8}{os.getenv('SERPAPI_API_KEY')[-4:] if os.getenv('SERPAPI_API_KEY') else 'Not set'}")
    print(f"DEBUG: {os.getenv('DEBUG', 'Not set')}")

    return Settings() 