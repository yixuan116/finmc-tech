"""
Macro-driven NVDA return simulation package.

This package provides tools for:
- Fetching macro and firm-specific data
- Building features from macro and firm data
- Training models (RF, LSTM) for return prediction
- Running macro scenario Monte Carlo simulations
- Visualizing results
"""

__version__ = "0.0.2"

from finmc_tech.config import Settings, cfg, get_logger, logger

__all__ = ["Settings", "cfg", "get_logger", "logger"]

