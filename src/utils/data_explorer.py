"""
Module for generic data exploration.

Author: Tue Nguyen
"""

from pprint import pprint

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


# --------------------------------------------------
# INTERNAL HELPER FUNCTIONS
# --------------------------------------------------
def _get_array_cols(df: pd.DataFrame) -> list[str]:
    """
    Get columns that contain array-like values in a DataFrame.
    """
    array_dtypes = (list, np.ndarray)
    array_cols = []

    for c in df.columns:
        # Get first non-null value to check type
        if df[c].isna().all():
            first_valid = None
        else:
            first_valid = df[c].dropna().iloc[0]

        # Check if the column is an array
        if isinstance(first_valid, array_dtypes):
            array_cols.append(c)

    return array_cols


# --------------------------------------------------
# SUMMARIZE DATAFRAME
# --------------------------------------------------
def summarize_df(
    df: pd.DataFrame,
    verbose: bool = False,
) -> None:
    """
    Summarize and show key information about a DataFrame for initial exploration.
    Args:
        df: Input DataFrame
        verbose: If True, show more detailed information
    """
    # Concise summary
    if not verbose:
        print(f"Shape: {df.shape}")
        display(df.head(1))
        return None

    # Basic information
    summary = {}
    summary["basic"] = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "memory_usage_mb": (df.memory_usage().sum() / 1024**2).round(2),
        "n_bool_cols": df.select_dtypes(include="bool").shape[1],
        "n_numeric_cols": df.select_dtypes(include="number").shape[1],
        "n_object_cols": df.select_dtypes(include="object").shape[1],
        "n_datetime_cols": df.select_dtypes(include="datetime").shape[1],
    }

    # Data types
    summary["dtypes"] = df.dtypes.to_dict()

    # Examples
    summary["examples"] = df.sample(1, random_state=1).to_dict(orient="records")[0]

    # Duplicates
    array_cols = _get_array_cols(df)
    regular_cols = df.columns.difference(array_cols)

    summary["duplicates"] = {
        "count": df[regular_cols].duplicated().sum(),
        "pct": (df[regular_cols].duplicated().sum() * 100.0 / len(df)).round(2),
    }

    # Null values
    null_info = df.isnull().sum()
    null_cols = null_info[null_info > 0]

    summary["nulls"] = {
        "count": null_cols.to_dict(),
        "pct": (null_cols * 100.0 / len(df)).round(2).to_dict(),
    }

    # Display first row and summary
    display(df.head(1))
    pprint(summary, sort_dicts=False)


# --------------------------------------------------
# SUMMARIZE COLUMNS
# --------------------------------------------------
def summarize_categorical_col(
    df: pd.DataFrame,
    col_name: str,
    max_cat_show: int = 10,
    show_table: bool = False,
    grid: bool = False,
) -> None:
    """
    Summarize a categorical column in a pandas DataFrame
    Args:
        df: Input DataFrame
        col_name: Column name to summarize
        max_cat_show: Maximum number of categories to show
        show_table: If True, show distribution table
        grid: If True, show grid lines in the bar plot
    """
    series = df[col_name].copy()
    n_unique = series.nunique()
    max_cat_show = min(max_cat_show, n_unique)

    # Compute distribution
    dist = series.value_counts().reset_index()
    dist.columns = ["category", "count"]
    dist["category"] = dist["category"].astype("str")

    dist["pct"] = (dist["count"] * 100.0 / len(series)).round(2)
    dist["cum_pct"] = (dist["count"].cumsum() * 100.0 / dist["count"].sum()).round(2)
    dist = dist.head(max_cat_show)

    # Plot distribution
    width = 7
    height = 1 + len(dist) * 0.3
    max_cum_pct = dist["cum_pct"].max()
    title = f"Distribution for column '{col_name}'"
    title += f"\n[top {len(dist)} of {n_unique}] [{max_cum_pct}% cumulative]"

    fig, ax = plt.subplots(figsize=(width, height))
    dist.sort_values("pct").plot.barh(
        x="category", y="pct", ax=ax, legend=False, width=0.8
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=5, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Percentage")
    ax.set_ylabel("Category")

    if grid:
        ax.grid(axis="x", linestyle="-", alpha=0.6)

    plt.margins(x=0.15)
    plt.tight_layout()
    plt.show()

    # Display distribution table
    if show_table:
        display(dist)


def summarize_numeric_col(
    df: pd.DataFrame,
    col_name: str,
) -> None:
    """
    Summarize a numeric column and its log-transformed version in KDE plots.
    """
    series = df[col_name].copy()

    # Plot KDE
    width = 7
    height = 3

    fig, ax = plt.subplots(1, 2, figsize=(width, height))

    # Original
    series.plot.kde(ax=ax[0])
    ax[0].set_title("Original")
    ax[0].set_xlabel("Value")
    ax[0].set_ylabel("Density")

    # Log-transformed
    log_series = series.apply(lambda x: max(1e-6, x)).apply("log")
    log_series.plot.kde(ax=ax[1])
    ax[1].set_title("Log-transformed")
    ax[1].set_xlabel("Log value")
    ax[1].set_ylabel("Density")

    # Add super title
    fig.suptitle(f"KDE plots for '{col_name}'", y=1.01)
    plt.tight_layout()
    plt.show()

    # Summary statistics
    summary = df[[col_name]].describe().T.round(2)
    display(summary)


def summarize_date_col(
    df: pd.DataFrame,
    col_name: str,
    freq: str = "M",
    grid: bool = False,
) -> None:
    """
    Summarize a date column in a pandas DataFrame.
    Args:
        df: Input DataFrame
        col_name: Column name to summarize
        freq: Frequency for time series aggregation
        grid: If True, show grid lines in the time series plot
    """
    series = pd.to_datetime(df[col_name], errors="coerce")

    # Aggregate time series
    ts = series.dt.to_period(freq).value_counts().sort_index().reset_index()
    ts.columns = ["period", "count"]
    ts["period"] = ts["period"].dt.to_timestamp()

    # Plot time series
    width = max(7, ts.shape[0] * 0.09)
    height = 4
    title = f"Time Series for '{col_name}'"
    title += f"\n[frequency: {freq}]"

    fig, ax = plt.subplots(figsize=(width, height))
    # sns.lineplot(data=ts, x="period", y="count", marker="o", ax=ax)
    ts.plot(x="period", y="count", marker="o", ax=ax)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    ax.set_title(title)
    ax.set_xlabel(None)
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0)

    plt.xticks(rotation=45)

    if grid:
        ax.grid(axis="y", linestyle="-", alpha=0.6)

    plt.margins(x=0.01)
    plt.tight_layout()
    plt.show()
