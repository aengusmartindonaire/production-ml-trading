"""
Feature definitions and simple feature-selection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


# --- Core factor lists from the project spec --- #

# 10 Bloomberg core risk factors
RSK_COLS = [
    "Sz",        # Size
    "Prof",      # Profitability
    "Vol",       # Volatility
    "Trd Act",   # Trading Activity
    "Lev",       # Leverage
    "Mom",       # Momentum
    "Val",       # Value
    "Gr",        # Growth
    "Dvd Yld",   # Dividend Yield
    "Earn Var",  # Earnings Variability
]

# 11 candidate extra features
EXTRA_FEATURES = [
    "P/S",
    "BEst P/S BF12M",
    "P/B",
    "BEst P/B BF12M",
    "P/E",
    "BEst P/E BF12M",
    "ROE LF",
    "Beta:Y-1",
    "Total Return:Y-1",
    "Number of Employees:Y",
    "Market Cap",
]


def classify_extra_feature(feature: str, max_abs_corr: float) -> str:
    """
    Replicates your notebook logic:
    decide whether an extra feature is redundant based on its
    maximum |correlation| with the 10 core risk factors.
    """
    if max_abs_corr > 0.7:
        return "drop: very high redundancy (|r| > 0.7)"
    elif max_abs_corr > 0.5:
        return "drop: high redundancy (|r| > 0.5)"
    elif max_abs_corr > 0.4:
        return "review: moderate correlation (0.4 < |r| ≤ 0.5)"
    else:
        return "keep candidate: low correlation (|r| ≤ 0.4)"


def build_extra_feature_decisions(max_corr_with_core: Dict[str, float]) -> pd.DataFrame:
    """
    Turn your max correlation dict into the DataFrame you built in the notebook.

    Parameters
    ----------
    max_corr_with_core:
        dict {feature_name -> max |corr| vs any core factor}

    Returns
    -------
    DataFrame with columns:
        feature, max_abs_corr_with_core, rule_based_decision
    """
    rows = []
    for feature, max_abs_corr in max_corr_with_core.items():
        rows.append(
            {
                "feature": feature,
                "max_abs_corr_with_core": max_abs_corr,
                "rule_based_decision": classify_extra_feature(feature, max_abs_corr),
            }
        )
    return pd.DataFrame(rows)

