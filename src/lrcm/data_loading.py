"""
Data loading helpers for Bloomberg factor panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def load_bloomberg_panel(path: PathLike) -> pd.DataFrame:
    """
    Load the main Bloomberg panel from a parquet file.

    In the big notebook you used something like:
        df_blg = pd.read_parquet("20251109_Blg_Rsk_Factors.parquet")

    Here we just wrap that into a function.
    """
    path = Path(path)
    df = pd.read_parquet(path)
    return df


def prepare_main_dataframe(df_blg: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic standard cleaning from Section 1.1 / 1.3:
    - ensure Date is sortable (e.g., int or period)
    - keep only necessary columns
    - any global filters you applied early in the notebook.

    This is intentionally light; you can expand it as you
    move more of your notebook logic into library code.
    """
    df = df_blg.copy()

    # Example: ensure Date is int and sorted
    if not pd.api.types.is_integer_dtype(df["Date"]):
        df["Date"] = df["Date"].astype(int)

    df = df.sort_values(["Date", "Ticker"])

    return df

