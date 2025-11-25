"""Configuration and logging for macro-driven NVDA simulation."""

import logging
from pathlib import Path
from typing import List, Optional

from pydantic import Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback: pydantic-settings not installed, use pydantic BaseSettings directly
    try:
        from pydantic import BaseSettings
        # For pydantic v1, use Config class instead of SettingsConfigDict
        class SettingsConfigDict:
            """Fallback SettingsConfigDict for pydantic v1."""
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
    except ImportError:
        raise ImportError(
            "pydantic-settings is required. Install with: pip install pydantic-settings"
        )


class Settings(BaseSettings):
    """Configuration class using Pydantic BaseSettings.
    
    Supports loading from environment variables and .env files.
    """
    
    model_config = SettingsConfigDict(
        # env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # API Keys
    FRED_API_KEY: Optional[str] = None
    
    # Date ranges
    START_DATE: str = "2010-01-01"
    END_DATE: str = "2025-10-31"
    TRAIN_END: str = "2022-12-31"
    TEST_START: str = "2023-01-01"
    
    # Ticker
    TICKER: str = "NVDA"
    
    # Directories
    CACHE_DIR: str = "data_cache"
    RESULTS_DIR: str = "results"
    
    # Random seed
    RANDOM_STATE: int = 42
    
    # Model parameters
    RF_N_ESTIMATORS: int = 300
    RF_MAX_DEPTH: Optional[int] = None
    RF_MIN_SAMPLES_SPLIT: int = 2
    
    # LSTM parameters (if using)
    LSTM_UNITS: int = 50
    LSTM_DROPOUT: float = 0.2
    LSTM_EPOCHS: int = 100
    LSTM_BATCH_SIZE: int = 32
    LSTM_SEQUENCE_LENGTH: int = 4
    
    # Simulation parameters
    N_SIMULATIONS: int = 10000
    DAYS_AHEAD: int = 252  # 1 year
    CONFIDENCE_LEVEL: float = 0.95
    
    # Feature engineering flags
    INCLUDE_PRICE_FEATURES: bool = True
    INCLUDE_TECHNICAL_FEATURES: bool = True
    INCLUDE_MARKET_FEATURES: bool = True
    INCLUDE_TIME_FEATURES: bool = True
    INCLUDE_INTERACTION_FEATURES: bool = True
    
    # Macro indicators to fetch
    MACRO_INDICATORS: List[str] = Field(default_factory=lambda: ["VIX", "TNX", "SP500"])
    
    def __init__(self, **kwargs):
        """Initialize settings and create directories."""
        super().__init__(**kwargs)
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create cache and results directories if they don't exist."""
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    @property
    def cache_dir(self) -> Path:
        """Get cache directory as Path object."""
        return Path(self.CACHE_DIR)
    
    @property
    def results_dir(self) -> Path:
        """Get results directory as Path object."""
        return Path(self.RESULTS_DIR)
    
    # Backward compatibility aliases
    @property
    def data_cache_dir(self) -> Path:
        """Alias for cache_dir (backward compatibility)."""
        return self.cache_dir
    
    @property
    def ticker(self) -> str:
        """Alias for TICKER (backward compatibility)."""
        return self.TICKER
    
    @property
    def start_date(self) -> str:
        """Alias for START_DATE (backward compatibility)."""
        return self.START_DATE
    
    @property
    def end_date(self) -> Optional[str]:
        """Alias for END_DATE (backward compatibility)."""
        return self.END_DATE
    
    @property
    def random_seed(self) -> int:
        """Alias for RANDOM_STATE (backward compatibility)."""
        return self.RANDOM_STATE
    
    @property
    def macro_indicators(self) -> List[str]:
        """Alias for MACRO_INDICATORS (backward compatibility)."""
        return self.MACRO_INDICATORS


# Global configuration instance
cfg = Settings()


# Logging configuration
_logging_configured = False


def get_logger(name: str = "finmc-tech") -> logging.Logger:
    """Get or create a logger with consistent configuration.
    
    Configures logging.basicConfig once (INFO level) on first call.
    Subsequent calls return loggers with the same configuration.
    
    Args:
        name: Logger name (typically module name)
    
    Returns:
        Configured logger instance
    """
    global _logging_configured
    
    if not _logging_configured:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _logging_configured = True
    
    return logging.getLogger(name)


# Global logger instance
logger = get_logger("finmc-tech")

