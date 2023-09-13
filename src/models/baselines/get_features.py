from typing import Union
import warnings

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings("ignore")


class HandEngineeredFeatures(object):
    def __init__(self, graph: Union[nx.Graph, nx.DiGraph], teams_composition: dict, teams_label: dict):
        """
        Compute features at team level: unique and total followers/followings, number of internal connections,
        assortativity, global clustering coefficient, density.

        :param graph: undirected/directed graph representing the ecosystem to work on it
        :param teams_composition: dictionary mapping from node id to corresponding team ids: {node_id: [...], ...}
        :param teams_label: dictionary containing the class C of each team: {team_id: C, ...}
        """
        self.graph = graph
        self.teams_composition = teams_composition
        self.teams_label = teams_label

        # Remove self loop.
        self.graph.remove_edges_from(nx.selfloop_edges(self.graph))

        # Create dataframe that maps each node to corresponding team.
        teams_info = pd.DataFrame(teams_composition.items(), columns=["node", "team"]).explode("team")
        # Add the information regarding the size of the teams and the corresponding composition.
        size_team = teams_info.groupby("team", as_index=False).agg(size=("node", "nunique"), members=("node", set))
        teams_info = pd.merge(teams_info, size_team, on="team", how="left")
        # Add information about teams label.
        teams_info["label"] = teams_info["team"].map(teams_label)

        # Get the list of the edges of the graph: both directions if undirected.
        edges = list(graph.edges()) if graph.is_directed() else [x[::-1] for x in graph.edges()] + list(graph.edges())
        edges = pd.DataFrame(edges, columns=["source", "destination"])
        # Create a dataframe where for each 'destination', we have the list of all the corresponding followers.
        followers = edges.groupby("destination", as_index=False).agg(followers=("source", lambda x: set(x)))
        followers.rename({"destination": "node"}, axis=1, inplace=True)
        # Create a dataframe where for each 'source', we have the list of all the corresponding followings.
        followings = edges.groupby("source", as_index=False).agg(followings=("destination", lambda x: set(x)))
        followings.rename({"source": "node"}, axis=1, inplace=True)
        # Merge followers and followings information.
        follow = pd.merge(followers, followings, on="node", how="outer")

        # Add the information regarding the followers of the users.
        system = pd.merge(teams_info, follow, on="node", how="left")

        # Fill the empty followers and followings.
        mask = system["followers"].isna()
        system.loc[mask, "followers"] = [set()] * mask.sum()
        mask = system["followings"].isna()
        system.loc[mask, "followings"] = [set()] * mask.sum()

        self.system = system

    def get_features(self, system_team):
        """Compute features at team level."""
        # Number of unique followers and total followers on outside graph considering team aggregation.
        outside_followers = (system_team["followers"] - system_team["members"]).explode()
        unique_outside_followers = outside_followers.nunique()
        total_outside_followers = outside_followers.count()
        # Number of unique followings and total following on outside graph considering team aggregation.
        outside_followings = (system_team["followings"] - system_team["members"]).explode()
        unique_outside_followings = outside_followings.nunique()
        total_outside_followings = outside_followings.count()

        # Number of connections inside the team for each member: intersection between 'followers' and 'members'.
        # Intersection can be expressed in terms of set difference: A intersection B = A - (A - B).
        inside_connections = system_team["members"] - (system_team["members"] - system_team["followers"])
        system_team["inside_connections"] = inside_connections
        total_inside_connections = system_team["inside_connections"].explode().dropna().count()

        # Sub-graph team.
        subgraph = nx.DiGraph() if self.graph.is_directed() else nx.Graph()
        subgraph.add_nodes_from(system_team["node"].unique())
        subgraph.add_edges_from(
            system_team[["node", "inside_connections"]].explode("inside_connections").dropna().values[:, [1, 0]])

        # Assortativity.
        r = nx.degree_assortativity_coefficient(subgraph)
        # Global clustering coefficient.
        cluster = nx.average_clustering(subgraph)
        # Density.
        density = nx.density(subgraph)

        return pd.Series([total_inside_connections, unique_outside_followers, total_outside_followers,
                          unique_outside_followings, total_outside_followings, r, cluster, density],
                         index=["total_inside_connections", "unique_outside_followers", "total_outside_followers",
                                "unique_outside_followings", "total_outside_followings", "assortativity",
                                "average_clustering", "density"])

    def __call__(self):
        """Compute inputs (X) and output (y) features at team level."""
        # Compute team network features.
        tqdm.pandas()
        teams = self.system.groupby("team", as_index=False).progress_apply(self.get_features)

        # Add the information regarding the size of the teams.
        teams_info = self.system[["team", "size", "label"]].groupby("team", as_index=False).first()
        teams = pd.merge(teams, teams_info, on="team", how="left")

        # Adjust some possible NaN values.
        teams[["assortativity", "density"]] = teams[["assortativity", "density"]].fillna(0)
        # There are some teams that could have inf assortativity.
        mask = (teams["assortativity"] == np.inf) | (teams["assortativity"] == -np.inf)
        teams.loc[mask, "assortativity"] = 0

        # Get input features.
        inputs = teams.drop(["team", "label"], axis=1)
        # Get output targets.
        outputs = teams[["label"]]

        return inputs, outputs
