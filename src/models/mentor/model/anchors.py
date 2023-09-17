from typing import Union

import networkx as nx
import numpy as np


def all_pairs_shortest_path_length(graph: Union[nx.Graph, nx.DiGraph], cutoff: int = None) -> dict:
    """Compute the shortest path lengths from a source node to other nodes in the graph."""
    dists_dict = {}
    for node in graph.nodes:
        dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)

    return dists_dict


def get_random_anchorset(num_nodes: int, c: int = 1) -> list:
    """Algorithm for generation of anchor-sets."""
    m = int(np.log2(num_nodes))
    copy = int(c * m)

    anchorset_id = []
    for i in range(m):
        anchor_size = int(num_nodes / np.exp2(i + 1))
        for j in range(copy):
            anchorset_id.append(np.random.choice(num_nodes, size=anchor_size, replace=False))

    return anchorset_id


def get_anchor_distances(dists, anchorset_id: list):
    """Algorithm for computing distances between nodes and anchor-sets."""
    dists_max = np.zeros((dists.shape[0], len(anchorset_id)), dtype=float)
    dists_argmax = np.zeros((dists.shape[0], len(anchorset_id)), dtype=int)
    for i in range(len(anchorset_id)):
        temp_id = anchorset_id[i]
        dist_temp = dists[:, temp_id]
        dist_max_temp, dist_argmax_temp = np.max(dist_temp, axis=-1), np.argmax(dist_temp, axis=-1)
        dists_max[:, i] = dist_max_temp
        dists_argmax[:, i] = temp_id[dist_argmax_temp]

    return dists_max, dists_argmax
