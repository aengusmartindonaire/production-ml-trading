"""
lrcm: Production ML System for Systematic Trading

This package contains reusable code for the project:
- Data loading and preprocessing
- Feature engineering
- Model construction and tuning
- Walk-forward backtesting
- Evaluation metrics and plotting
"""

from . import data_loading
from . import preprocessing
from . import features
from . import models
from . import backtest
from . import evaluation
from . import plots

# Convenience re-exports so notebooks can do:
#   from lrcm import run_walk_forward_tv_style
from .backtest import run_walk_forward_tv_style
from .evaluation import (
    compute_backtest_summary,
    summarize_by_year,
)
from .plots import plot_equity_curve, plot_drawdowns
from .data_loading import load_market_data
from .preprocessing import preprocess_data
from .features import create_features
from .models import build_model, tune_hyperparameters

__version__ = "0.1.0"