from typing import Union
import pickle
import logging
import os

import pandas as pd
import numpy as np
import networkx as nx


def load_graph_information(dataset_path: str) -> tuple[Union[nx.Graph, nx.DiGraph], dict, dict, pd.DataFrame]:
    """
    Load graph information from the given dataset path.

    :param dataset_path: path to the directory containing dataset files
    :return: a tuple containing the following:
             - graph object (undirected or directed)
             - dictionary mapping node IDs to corresponding teams: {node_id: [team_1, ...]}
             - dictionary mapping team IDs to labels: {team_id: label, ...}
             - DataFrame containing nodes attributes (or None if not available)
    """
    # Load networkx graph.
    with open(f"{dataset_path}/graph.pkl", "rb") as f:
        graph = nx.readwrite.json_graph.node_link_graph(pickle.load(f))

    # Load teams' labels information.
    with open(f"{dataset_path}/teams_label.pkl", "rb") as f:
        teams_label = pickle.load(f)

    # Load nodes attribute (if exist).
    if os.path.exists(f"{dataset_path}/nodes_attribute.pkl"):
        with open(f"{dataset_path}/nodes_attribute.pkl", "rb") as f:
            nodes_attribute = pickle.load(f)
            n_features = len(nodes_attribute.columns)
    else:
        nodes_attribute = None
        n_features = 0

    # Define the teams and the team labels.
    teams = list(teams_label.keys())
    labels = list(teams_label.values())
    # Define the number of teams and the number of unique labels/classes.
    n_teams = len(teams)
    n_classes = len(np.unique(labels))

    # Mapping from node id to corresponding teams.
    teams_composition = nx.get_node_attributes(graph, "Team")
    # Get nodes that not belong to the teams: None value in correspondence of no team belonging.
    nodes_not_belong_to_teams = [node for node, teams in teams_composition.items() if teams is None]
    # Check if there exist some overlap between the members of the teams.
    overlapping_members = True if max(
        [len(teams) for node, teams in teams_composition.items() if teams is not None]) > 1 else False

    # Log useful dataset information.
    logging.info(f"Directed graph: {graph.is_directed()}")
    logging.info(f"Number of nodes: {graph.number_of_nodes()}")
    logging.info(f"Number of edges: {graph.number_of_edges()}")
    logging.info(f"Number of teams: {n_teams}")
    logging.info(f"Number of classes: {n_classes}")
    logging.info(f"Number of features for each node: {n_features}")
    logging.info(f"Number of nodes that not belong to any team: {len(nodes_not_belong_to_teams)}")
    logging.info(f"Overlapping members of the teams: {overlapping_members}")

    return graph, teams_composition, teams_label, nodes_attribute
