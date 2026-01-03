"""
Walk-forward backtest logic for the TV-style strategy.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone


def run_walk_forward_tv_style(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    model,
    window: int = 12,
    top_bottom_frac: float = 0.10,
    min_names: int = 30,
    window_type: str = "rolling",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Walk-forward backtest for a TV-style cross-sectional model.

    Parameters
    ----------
    df :
        Panel with columns:
            - 'Date'
            - 'FwdRet'      (clipped forward return, for IC)
            - 'FwdRetOrig'  (unclipped forward return, for P&L)
            - feature_cols
    feature_cols :
        List of feature column names to feed into the model.
    model :
        Any sklearn-compatible regressor (e.g., XGBRegressor).
    window :
        Lookback window in number of quarters if window_type='rolling'.
    top_bottom_frac :
        Fraction of names to go long and short (e.g. 0.10 => top/bottom decile).
    min_names :
        Minimum number of names per side (long and short).
    window_type :
        'rolling' (fixed-size window) or 'expanding' (use all history).
    verbose :
        If True, prints progress and summary statistics.

    Returns
    -------
    results_df :
        Per-period performance DataFrame with columns:
            - rank_ic
            - n_universe
            - n_per_side
            - long_return
            - short_return
            - long_short_alpha
            - benchmark_return
    summary_stats :
        Dict with keys:
            mean_ic, ic_hit_rate, cumulative_alpha, sharpe_ratio, max_drawdown
    """
    df = df.copy()
    feature_cols = list(feature_cols)

    # Sorted unique quarters
    dates = sorted(df["Date"].unique())

    if window_type not in {"rolling", "expanding"}:
        raise ValueError("window_type must be 'rolling' or 'expanding'.")

    if window_type == "rolling":
        start_idx = window
    else:  # expanding
        start_idx = 1

    results: List[Dict[str, Any]] = []

    for t_idx in range(start_idx, len(dates)):
        dt = dates[t_idx]

        if window_type == "rolling":
            train_dates = dates[t_idx - window : t_idx]
        else:  # expanding
            train_dates = dates[:t_idx]

        train_mask = df["Date"].isin(train_dates)
        test_mask = df["Date"] == dt

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "FwdRet"]

        X_test = df.loc[test_mask, feature_cols]
        y_test_clipped = df.loc[test_mask, "FwdRet"]
        y_test_actual = df.loc[test_mask, "FwdRetOrig"]

        # Drop rows with missing feature or target values (no leakage)
        train_keep = X_train.notna().all(axis=1) & y_train.notna()
        X_train_clean = X_train[train_keep]
        y_train_clean = y_train[train_keep]

        test_keep = (
            X_test.notna().all(axis=1)
            & y_test_clipped.notna()
            & y_test_actual.notna()
        )
        X_test_clean = X_test[test_keep]
        y_test_clipped_clean = y_test_clipped[test_keep]
        y_test_actual_clean = y_test_actual[test_keep]

        universe_size = len(X_test_clean)
        min_needed = max(min_names * 2, int(2 * top_bottom_frac * universe_size))

        if universe_size < min_needed:
            if verbose:
                print(
                    f"Skipping {dt}: need at least {min_needed} stocks for deciles, "
                    f"but only have {universe_size}."
                )
            continue

        if verbose:
            print(
                f"Train: {train_dates[0]} ... {train_dates[-1]}  |  "
                f"Test: {dt}  |  Universe: {universe_size}"
            )

        # Fit a fresh clone of the model each period (no state carry-over)
        period_model = clone(model)
        period_model.fit(X_train_clean, y_train_clean)

        scores = period_model.predict(X_test_clean)

        # Rank-IC on clipped returns
        if np.std(scores) == 0 or np.std(y_test_clipped_clean) == 0:
            rank_ic = np.nan
        else:
            rank_ic, _ = spearmanr(scores, y_test_clipped_clean)

        # Portfolio formation: long top decile, short bottom decile
        n_per_side = max(min_names, int(top_bottom_frac * universe_size))

        order = np.argsort(scores)  # ascending
        short_idx = order[:n_per_side]
        long_idx = order[-n_per_side:]

        long_ret = float(y_test_actual_clean.iloc[long_idx].mean())
        short_ret = float(y_test_actual_clean.iloc[short_idx].mean())
        long_short_alpha = long_ret - short_ret

        benchmark_ret = float(y_test_actual_clean.mean())

        results.append(
            {
                "Date": dt,
                "rank_ic": rank_ic,
                "n_universe": universe_size,
                "n_per_side": n_per_side,
                "long_return": long_ret,
                "short_return": short_ret,
                "long_short_alpha": long_short_alpha,
                "benchmark_return": benchmark_ret,
            }
        )

    results_df = pd.DataFrame(results).set_index("Date").sort_index()

    # --- Summary stats over the whole backtest --- #
    mean_ic = results_df["rank_ic"].mean()
    ic_hit_rate = (results_df["rank_ic"] > 0).mean()

    cumulative_alpha = results_df["long_short_alpha"].sum()
    alpha_series = results_df["long_short_alpha"]
    alpha_mean = alpha_series.mean()
    alpha_std = alpha_series.std()

    if alpha_std > 0:
        sharpe_ratio = (alpha_mean / alpha_std) * np.sqrt(4.0)  # quarterly -> annual
    else:
        sharpe_ratio = np.nan

    cum_curve = alpha_series.cumsum()
    running_max = cum_curve.cummax()
    drawdowns = cum_curve - running_max
    max_drawdown = float(drawdowns.min())

    summary_stats = {
        "mean_ic": float(mean_ic),
        "ic_hit_rate": float(ic_hit_rate),
        "cumulative_alpha": float(cumulative_alpha),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": max_drawdown,
    }

    if verbose:
        print("\n" + "-" * 60)
        print("SUMMARY STATS (based on long-short alpha):")
        print(f"Mean IC:           {mean_ic:.4f}")
        print(f"IC Hit Rate:       {ic_hit_rate:.2%}")
        print(
            f"Cumulative Alpha:  {cumulative_alpha:+.4f} "
            f"({cumulative_alpha * 100:+.2f}%)"
        )
        print(f"Sharpe Ratio:      {sharpe_ratio:.4f}")
        print(f"Max Drawdown:      {max_drawdown:.4f}")
        print("-" * 60)

    return results_df, summary_stats

