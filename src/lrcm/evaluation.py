"""
Evaluation utilities: IC, hit rate, portfolio stats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def print_performance_metrics(results_df: pd.DataFrame, label: str = "Strategy") -> None:
    """
    Pretty-print the performance summary for a walk-forward run.

    Assumes results_df has columns:
        rank_ic, long_return, short_return, benchmark_return
    """
    print("\n" + "=" * 60)
    print(f"STRATEGY PERFORMANCE METRICS: {label}")
    print("=" * 60)

    # Mean Information Coefficient
    mean_ic = results_df["rank_ic"].mean()
    print(f"Mean Rank IC: {mean_ic:.4f}")

    # IC Hit Rate (% of periods with positive IC)
    ic_hit_rate = (results_df["rank_ic"] > 0).sum() / len(results_df)
    print(f"IC Hit Rate: {ic_hit_rate:.2%}")

    # Long Portfolio Performance
    long_cumulative = results_df["long_return"].sum()
    long_mean = results_df["long_return"].mean()
    long_vol = results_df["long_return"].std()
    long_ir = (long_mean / long_vol) * np.sqrt(4) if long_vol > 0 else np.nan

    print(f"\nLong Portfolio:")
    print(f"  Cumulative Return: {long_cumulative:+.4f} ({long_cumulative*100:+.2f}%)")
    print(f"  Mean Quarterly Return: {long_mean:+.4f}")
    print(f"  Volatility (Quarterly): {long_vol:.4f}")
    print(f"  Information Ratio (Annualized): {long_ir:.4f}")

    # Short Portfolio Performance
    short_cumulative = results_df["short_return"].sum()
    short_mean = results_df["short_return"].mean()
    short_vol = results_df["short_return"].std()
    short_ir = (short_mean / short_vol) * np.sqrt(4) if short_vol > 0 else np.nan

    print(f"\nShort Portfolio:")
    print(f"  Cumulative Return: {short_cumulative:+.4f} ({short_cumulative*100:+.2f}%)")
    print(f"  Mean Quarterly Return: {short_mean:+.4f}")
    print(f"  Volatility (Quarterly): {short_vol:.4f}")
    print(f"  Information Ratio (Annualized): {short_ir:.4f}")

    # Benchmark Comparison
    benchmark_cumulative = results_df["benchmark_return"].sum()
    benchmark_mean = results_df["benchmark_return"].mean()
    print(f"\nBenchmark (Market Average):")
    print(
        f"  Cumulative Return: {benchmark_cumulative:+.4f} "
        f"({benchmark_cumulative*100:+.2f}%)"
    )
    print(f"  Mean Quarterly Return: {benchmark_mean:+.4f}")

    # Excess Returns
    long_excess = long_cumulative - benchmark_cumulative
    short_excess = benchmark_cumulative - short_cumulative
    print(f"\nExcess Returns vs Benchmark:")
    print(f"  Long Portfolio Excess: {long_excess:+.4f} ({long_excess*100:+.2f}%)")
    print(f"  Short Portfolio Excess: {short_excess:+.4f} ({short_excess*100:+.2f}%)")

    print("=" * 60)

