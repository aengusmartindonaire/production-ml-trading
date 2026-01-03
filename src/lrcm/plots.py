"""
Plotting and SHAP-based interpretability helpers.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def shap_segment_summary(
    X_sample: pd.DataFrame,
    shap_values: np.ndarray,
    theme_keyword: str,
    max_display: int = 8,
) -> None:
    """
    Filter X_sample to rows where any column containing theme_keyword is 1,
    then show a SHAP summary plot for that segment.

    This is your notebook logic, but now parameterized.
    """
    theme_cols = X_sample.filter(like=theme_keyword).columns
    print(f"{theme_keyword}-related columns found:", theme_cols.tolist())

    if len(theme_cols) == 0:
        print(f"No columns found containing '{theme_keyword}'.")
        return

    mask = X_sample[theme_cols].sum(axis=1) > 0
    X_seg = X_sample[mask]
    shap_seg = shap_values[mask.values]

    print(f"\n--- SHAP Summary for theme '{theme_keyword}' ONLY ---")
    print(f"Analyzing {mask.sum()} stocks out of {len(X_sample)} total.\n")

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_seg, X_seg, max_display=max_display)
    plt.show()


def explain_single_stock(
    row_id: str,
    X_clean: pd.DataFrame,
    explainer: shap.Explainer,
):
    """
    Your force-plot helper: explain a single Ticker_Date index row.
    """
    stock_to_explain = X_clean.loc[[row_id]]
    stock_expl = explainer(stock_to_explain)

    print(f"--- SHAP Force Plot for {row_id} ---")
    return shap.force_plot(
        explainer.expected_value,
        stock_expl.values[0],
        stock_to_explain.iloc[0],
    )


def plot_stock_waterfall(
    ticker_date_index: str,
    X_clean: pd.DataFrame,
    shap_values: np.ndarray,
    title_suffix: str = "",
    max_display: int = 15,
) -> None:
    """
    Waterfall post-mortem for a given Ticker_Date row,
    using the global shap_values array.
    """
    row_pos = X_clean.index.get_loc(ticker_date_index)
    stock_shap_values = shap_values[row_pos]

    print(f"\nWaterfall for {ticker_date_index} {title_suffix}")
    shap.plots.waterfall(stock_shap_values, max_display=max_display)


def xgb_predict_numeric(model, X):
    """
    Wrapper so SHAP always calls XGBoost with numeric data.
    This is your `xgb_predict_numeric` from the notebook,
    but now takes the model as an argument.
    """
    if isinstance(X, pd.DataFrame):
        X_numeric = X.apply(pd.to_numeric, errors="coerce")
        return model.predict(X_numeric)
    else:
        return model.predict(X)


def plot_pdp(
    model,
    X_background: pd.DataFrame,
    feature_name: str,
) -> None:
    """
    Plot SHAP partial dependence for a single feature.

    Equivalent to your `plot_pdp` helper in the notebook.
    """
    X_sample_num = X_background.apply(pd.to_numeric, errors="coerce")

    def _wrapped_predict(X):
        return xgb_predict_numeric(model, X)

    print(f"--- Displaying PDP for {feature_name} ---")
    shap.partial_dependence_plot(
        feature_name,
        _wrapped_predict,
        X_sample_num,
        model_expected_value=True,
        feature_expected_value=True,
        ice=False,
    )

