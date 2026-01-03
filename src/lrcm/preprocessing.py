"""
Preprocessing utilities: clipping, target creation, GICS cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def clip_forward_return(df: pd.DataFrame, col: str = "FwdRet", q: float = 0.008) -> pd.DataFrame:
    """
    Clip forward returns at symmetric quantiles, as in Section 1.3.

    Parameters
    ----------
    df  : DataFrame with at least `col`
    col : column with raw forward returns (e.g., FwdRetOrig)
    q   : lower tail quantile (you empirically chose ~0.008)

    Returns
    -------
    New DataFrame with:
        - original column untouched
        - 'FwdRet' column with clipped values (if col != 'FwdRet')
    """
    df = df.copy()

    # Remember the original forward return
    if col != "FwdRetOrig" and "FwdRetOrig" not in df.columns:
        df["FwdRetOrig"] = df[col]

    lower = df[col].quantile(q)
    upper = df[col].quantile(1 - q)

    df["FwdRet"] = df[col].clip(lower=lower, upper=upper)

    return df


def handle_rare_gics(df: pd.DataFrame, rare_threshold: int = 500) -> pd.DataFrame:
    """
    Reproduce your 'rare GICS levels' logic from 1.5:
    - mark very small categories as NaN
    - backfill using higher-level buckets (Sector -> Industry -> SubInd).

    Assumes these columns exist:
        'GICS_Sector_Name', 'GICS_Industry_Name', 'GICS_SubInd_Name'
    """
    df = df.copy()

    gics_cols = ["GICS_Sector_Name", "GICS_Industry_Name", "GICS_SubInd_Name"]

    # Step 1: mark rare categories as NaN
    for c in gics_cols:
        vc = df[c].value_counts(dropna=True)
        rare_cats = vc[vc < rare_threshold].index
        df.loc[df[c].isin(rare_cats), c] = pd.NA

    # Step 2: fill sector, then use it to fill industry/subind
    df["GICS_Sector_Name"] = df["GICS_Sector_Name"].fillna("Other")

    df.loc[df["GICS_Industry_Name"].isna(), "GICS_Industry_Name"] = df.loc[
        df["GICS_Industry_Name"].isna(), "GICS_Sector_Name"
    ]
    df.loc[df["GICS_SubInd_Name"].isna(), "GICS_SubInd_Name"] = df.loc[
        df["GICS_SubInd_Name"].isna(), "GICS_Sector_Name"
    ]

    return df

