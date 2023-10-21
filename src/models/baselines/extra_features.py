from typing import Union

import networkx as nx
import pandas as pd
from graph_tool import Graph
from graph_tool.centrality import betweenness


def extra_node_attributes(
    nodes_attribute: pd.DataFrame, teams_composition: dict, graph: Union[nx.Graph, nx.DiGraph], feature_funcs: list
):
    """
    Function used to compute extra nodes attribute in addition to the ones precomputed

    :param nodes_attribute: dataframe containing the nodes attribute (could be None if a synthetic dataset is used)
    :param teams_composition: dictionary mapping from node id to corresponding team ids: {node_id: [...], ...}
    :param graph: graph representing the ecosystem to work on it
    :param feature_funcs: list of functions to compute extra nodes attribute
    """

    if nodes_attribute is None:
        nodes_attribute = [
            (key, inner_val) for key, inner_list in teams_composition.items() for inner_val in inner_list
        ]
        nodes_attribute = pd.DataFrame(nodes_attribute, columns=["node", "team"])
    else:
        nodes_attribute = nodes_attribute.reset_index()

    # Assert feature_funcs is not None, "feature_funcs must be a list of functions".
    assert isinstance(feature_funcs, list), "feature_funcs must be a list of functions"

    for func in feature_funcs:
        mapper = func(graph)
        nodes_attribute[func.__name__] = nodes_attribute["node"].map(mapper)

    nodes_attribute = nodes_attribute.set_index(["node", "team"])

    return nodes_attribute


def betweenness_centrality(graph: Union[nx.Graph, nx.DiGraph]):
    """Compute betweenness centrality for each node in the graph."""

    # Use graph-tool to compute betweenness centrality for performance reasons.
    directed = True if graph.is_directed() else False

    g = Graph(directed=directed)
    g.add_edge_list(graph.edges())

    # Compute betweenness centrality.
    v_betweenness, e_betweenness = betweenness(g)

    # Obtain nodes and values.
    nodes = [int(v) for v in g.vertices()]
    btw_values = v_betweenness.a

    return {node: btw_values[i] for i, node in enumerate(nodes)}


def pagerank_centrality(graph: Union[nx.Graph, nx.DiGraph]):
    """Compute pagerank centrality for each node in the graph."""

    # Use networkx to compute pagerank centrality. Send the graph as undirected as pagerank is not defined for
    # directed graphs.
    page_rank = nx.pagerank(graph.to_undirected())

    return page_rank
