"""
Module for analyzing sponsor networks from bill sponsorships.

Author: Tue Nguyen
"""

import netbone as nb
import networkx as nx
import pandas as pd

from src.config import cfg
from src.data_prep import data_prep
from src.utils import viz


# --------------------------------------------------
# NETWORK CONSTRUCTION
# --------------------------------------------------
def create_static_network(
    df: pd.DataFrame,
    congresses: list[int] | None = None,
    bill_types: list[str] | None = None,
    policy_areas: list[str] | None = None,
    node_attr_cols: list[str] | None = None,
    backbone_threshold: float = 0.05,
) -> tuple[nx.Graph, nx.Graph, nx.Graph]:
    """
    Create a static network from bill sponsorships.

    Args:
        data: DataFrame with bill sponsorships
        congresses: List of congresses to include
        bill_types: List of bill types to include
        policy_areas: List of policy areas to include
        node_attr_cols: Columns to include as node attributes
        backbone_threshold: Threshold for backbone extraction

    Returns:
       Tuple of NetworkX graphs (bipartite, projected, backbone)
    """
    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------
    # Network data
    df = data_prep.apply_main_filters(df, congresses, bill_types, policy_areas)

    # Network attributes
    if not node_attr_cols:
        node_attr_cols = ["party", "display_name", "sponsor_title"]

    cols = ["sponsor_id"] + list(node_attr_cols)
    sponsor_info = df.drop_duplicates("sponsor_id")[cols].to_dict("records")
    node_attrs = {row.pop("sponsor_id"): row for row in sponsor_info}

    # --------------------------------------------------
    # CREATE BIPARTITE NETWORK
    # --------------------------------------------------
    top_nodes = df["sponsor_id"].unique()
    bottom_nodes = df["bill_id"].unique()

    G_bip = nx.from_pandas_edgelist(
        df,
        source="sponsor_id",
        target="bill_id",
        create_using=nx.Graph,
    )

    G_bip.add_nodes_from(top_nodes, bipartite=0)
    G_bip.add_nodes_from(bottom_nodes, bipartite=1)

    # --------------------------------------------------
    # CREATE PROJECTED NETWORK
    # ------------------------------------------------
    G_proj = nx.bipartite.weighted_projected_graph(G_bip, top_nodes)
    nx.set_node_attributes(G_proj, node_attrs)

    # --------------------------------------------------
    # CREATE BACKBONE NETWORK
    # ------------------------------------------------
    # Backbone extraction
    backbone = nb.marginal_likelihood(nx.to_pandas_edgelist(G_proj))
    G_bone = nb.threshold_filter(backbone, backbone_threshold)

    for n in G_bone.nodes:
        G_bone.nodes[n].update(G_proj.nodes[n])

    return G_bip, G_proj, G_bone


def create_temporal_network(
    df: pd.DataFrame,
    congresses: list[int] | None = None,
    bill_types: list[str] | None = None,
    policy_areas: list[str] | None = None,
    node_attr_cols: list[str] | None = None,
    backbone_threshold: float = 0.05,
    keep_bipartite: bool = False,
) -> list[dict[str, nx.Graph]]:
    """
    Create a temporal network from bill sponsorships.

    Args:
        data: DataFrame with bill sponsorships
        congresses: List of congresses to include
        bill_types: List of bill types to include
        policy_areas: List of policy areas to include
        node_attr_cols: Columns to include as node attributes
        backbone_threshold: Threshold for backbone extraction
        keep_bipartite: Whether to keep the bipartite network

    Returns:
        List of NetworkX graphs ordered by congress
    """
    df = data_prep.apply_main_filters(df, congresses, bill_types, policy_areas)

    networks = []
    periods = sorted(df["congress"].unique())
    for period in periods:
        data = df[df["congress"] == period]
        G_bip, G_proj, G_bone = create_static_network(
            data,
            congresses=[period],
            bill_types=bill_types,
            policy_areas=policy_areas,
            node_attr_cols=node_attr_cols,
            backbone_threshold=backbone_threshold,
        )

        elem = {"congress": period, "proj": G_proj, "bone": G_bone}

        if keep_bipartite:
            elem["bip"] = G_bip

        networks.append(elem)

    return networks


# --------------------------------------------------
# NETWORK VISUALIZATION
# --------------------------------------------------
def draw_list_of_networks(
    G_list: list[nx.Graph],
    captions: list[str] | None = None,
    n_rows: int | None = None,
    n_cols: int | None = None,
    cell_size: tuple[int, int] = (3.5, 3),
    layout_func: callable = nx.spring_layout,
    node_color_using: str = "party",
    color_map: dict[str, str] = cfg.cmap.PARTY,
    **kwargs,
):
    """
    Draw a list of networks.

    Args:
        networks: List of NetworkX graphs
        captions: List of captions for each network
        n_rows: Number of rows
        n_cols: Number of columns
        cell_size: Size of each cell
        layout_func: Network layout function
        **kwargs: Additional parameters passed to nx.draw()
    """
    n_plots = len(G_list)
    fig, axes = viz.create_plot_grid(n_plots, n_rows, n_cols, cell_size)

    for i, g in enumerate(G_list):
        ax = axes[i]
        node_colors = viz.nx_get_node_colors(g, attr=node_color_using, cmap=color_map)

        if captions:
            ax.set_title(captions[i], fontsize=9)

        viz.nx_plot_network(
            g,
            ax=ax,
            layout_func=layout_func,
            node_color=node_colors,
            **kwargs,
        )

        ax = viz.add_frame_to_ax(ax)

    fig.tight_layout()
    return axes
