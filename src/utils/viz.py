"""
Module for visualization.

Author: Tue Nguyen
"""

import inspect
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.patches import Rectangle


# --------------------------------------------------
# PLOTTING UTILS
# --------------------------------------------------
def create_plot_grid(
    n_plots: int,
    n_rows: int | None = None,
    n_cols: int | None = None,
    cell_size: tuple[int, int] = (3.5, 3),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create a grid of subplots for plotting.
    Args:
        n_plots: Number of plots
        n_rows: Number of rows
        n_cols: Number of columns
        cell_size: Size of each cell
    """
    # Calculate optimal layout if not provided
    if n_rows is None and n_cols is None:
        n_cols = int(n_plots**0.5)
        n_rows = (n_plots + n_cols - 1) // n_cols
    elif n_rows is None:
        n_rows = (n_plots + n_cols - 1) // n_cols
    elif n_cols is None:
        n_cols = (n_plots + n_rows - 1) // n_rows

    # Create the figure and subplots
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * cell_size[0], n_rows * cell_size[1]),
        squeeze=False,
    )

    axes = axes.flatten()

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].axis("off")

    return fig, axes[:n_plots]


def add_frame_to_ax(
    ax: plt.Axes,
    linewidth: float = 2.0,
    edgecolor: str = "#CCCCCC",
) -> plt.Axes:
    """
    Adds a visible frame (border) to a matplotlib axis.
    Args:
        ax: Matplotlib axis
        linewidth: Border thickness
        edgecolor: Border color
    """
    # Get the axis bounding box
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Create a rectangle patch as frame
    rect = Rectangle(
        (xmin, ymin),  # Bottom-left corner
        xmax - xmin,  # Width
        ymax - ymin,  # Height
        linewidth=linewidth,  # Border thickness
        edgecolor=edgecolor,  # Border color
        facecolor="none",  # No fill
    )

    ax.add_patch(rect)
    return ax


# --------------------------------------------------
# GENERAL PLOTTING
# --------------------------------------------------
def plot_relative_proportion(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    orient: str = "v",
    figsize: tuple[int, int] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes | None:
    """
    Create a 100% stacked bar chart showing the relative proportion by category.
    Args:
        data: Dataframe
        x: x-axis variable
        y: y-axis variable
        hue: category
        orient: Orientation of the plot (v: vertical, h: horizontal)
        figsize: Figure size
        ax: Matplotlib axis
        **kwargs: Additional kwargs
    """
    # Config
    return_ax = False if ax is None else True
    bar_kind = "bar" if orient == "v" else "barh"

    # Prepare plot data
    plot_data = data.pivot(index=x, columns=hue, values=y)
    plot_data = plot_data.apply(lambda x: x / x.sum(), axis=1) * 100

    # Plot parameters
    defaults = {
        "title": f"Relative proportion of '{y}' by '{hue}'",
        "width": 0.75,
    }

    kwargs = {**defaults, **kwargs}

    # Axis
    if not figsize:
        if orient == "v":
            figsize = (1 + len(plot_data) * 0.55, 3.5)
        else:
            figsize = (7, 1 + len(plot_data) * 0.3)

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    # Plot
    plot_data.plot(
        kind=bar_kind,
        stacked=True,
        ax=ax,
        **kwargs,
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    for c in ax.containers:
        pct_thres = 5

        if orient == "v":
            values = [rect.get_height() for rect in c]
        else:
            values = [rect.get_width() for rect in c]

        labels = [f"{v:.1f}" if v > pct_thres else "" for v in values]
        ax.bar_label(c, labels=labels, label_type="center", fontsize=8)

    if return_ax:
        return ax

    plt.tight_layout()
    plt.show()


def plot_bar(
    data: pd.DataFrame,
    x: str,
    y: str,
    orient: str = "v",
    figsize: tuple[int, int] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes | None:
    """
    Create a general bar chart.
    Args:
        data: Dataframe
        x: x-axis variable
        y: y-axis variable
        orient: Orientation of the plot (v: vertical, h: horizontal)
        figsize: Figure size
        ax: Matplotlib axis
        **kwargs: Additional kwargs
    """
    # Config
    return_ax = False if ax is None else True
    bar_kind = "bar" if orient == "v" else "barh"

    # Plot parameters
    defaults = {
        "title": f"Bar chart of '{y}'",
        "width": 0.75,
    }

    kwargs = {**defaults, **kwargs}

    # Axis
    if not figsize:
        if orient == "v":
            figsize = (1 + len(data[x].unique()) * 0.55, 3.5)
        else:
            figsize = (7, 1 + len(data[x].unique()) * 0.3)

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    # Plot
    data.plot(
        x=x,
        y=y,
        kind=bar_kind,
        ax=ax,
        **kwargs,
    )

    ax.get_legend().remove()
    for c in ax.containers:

        if orient == "v":
            labels = [rect.get_height() for rect in c]
        else:
            labels = [rect.get_width() for rect in c]
        ax.bar_label(c, labels=labels, label_type="center", fontsize=9)

    if return_ax:
        return ax

    plt.tight_layout()
    plt.show()


def plot_line(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    figsize: tuple[int, int] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> plt.Axes | None:
    """
    Create a general line chart.
    Args:
        data: Dataframe
        x: x-axis variable
        y: y-axis variable
        hue: category
        figsize: Figure size
        ax: Matplotlib axis
        **kwargs: Additional kwargs
    """
    # Config
    return_ax = False if ax is None else True

    # Plot parameters
    defaults = {
        "title": f"Line chart of '{y}' by '{hue}'",
    }

    kwargs = {**defaults, **kwargs}

    # Axis
    if not figsize:
        figsize = (7, 3.5)

    if not ax:
        _, ax = plt.subplots(figsize=figsize)

    # Plot
    data.pivot(index=x, columns=hue, values=y).plot(
        ax=ax,
        **kwargs,
    )

    if return_ax:
        return ax

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# GRAPH VISUALIZATION
# --------------------------------------------------
def nx_plot_network(
    g: nx.Graph,
    ax: Optional[plt.Axes] = None,
    layout_params: Optional[Dict[str, Any]] = None,
    layout_func: callable = nx.spring_layout,
    **kwargs,
) -> plt.Axes:
    """
    Plot a NetworkX graph using matplotlib.
    Args:
        g: NetworkX graph
        ax: Matplotlib axis
        layout_params: Dictionary of layout parameters
        layout_func: NetworkX layout function to use
        **kwargs: Additional parameters passed to nx.draw()
    """
    # Default parameters for draw
    default_params = {
        "node_size": 50,
        "node_color": "blue",
        "edgecolors": "none",
        "alpha": 0.5,
        "width": 0.5,
        "edge_color": "#B3B3B3",
    }

    # Default layout parameters
    default_layout_params = {"seed": 0, "k": 1, "iterations": 50}

    # Initialize parameters
    draw_params = {**default_params, **kwargs}
    layout_params = {**default_layout_params, **(layout_params or {})}

    # Filter layout parameters
    valid_layout_params = inspect.signature(layout_func).parameters.keys()
    filtered_layout_params = {
        k: v for k, v in layout_params.items() if k in valid_layout_params
    }

    # EPlot
    if ax is None:
        ax = plt.gca()

    pos = layout_func(g, **filtered_layout_params)
    nx.draw(g, pos=pos, ax=ax, **draw_params)

    return ax


def nx_get_node_colors(
    g: nx.Graph,
    attr: str,
    cmap: dict[str, str] | None = None,
    default_color="lightgray",
) -> list[str]:
    """
    Get node colors based on a node attribute.
    Args:
        g: NetworkX graph
        attr: Node attribute to use for coloring
        cmap: Dictionary mapping attribute values to colors
        default_color: Color to use for nodes without the attribute or unknown values
    """
    if cmap is None:
        cmap = {}

    return [cmap.get(g.nodes[node].get(attr), default_color) for node in g.nodes()]
