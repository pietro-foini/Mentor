import itertools
from collections import Counter
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import HeteroData
from torch_geometric.utils import from_networkx

from .anchors import all_pairs_shortest_path_length, get_anchor_distances, get_random_anchorset


class DataObject(object):
    """
    Data class containing HeteroData object storing information related to isolated sub-graphs and hypergraph.

    :param graph: a networkx undirected/directed graph representing the ecosystem to work on it
    :param teams_composition: a dictionary containing the mapping from graph node id to corresponding teams,
        e.g. {node_id: [...]}
    :param teams_members: a dictionary containing the mapping from team id to corresponding members,
        e.g. {team_id: [...]}
    :param teams_label: a dictionary containing the class of each team: {team_id: C}
    :param nodes_not_belong_to_teams: list containing the graph node ids that not belong to any team
    :param nodes_attribute: an optional multi-index pandas dataframe (level 0 for node id and level 1 for team id)
        containing node attributes
    :param nodes_name: an optional dictionary containing the mapping from node id to corresponding name,
        e.g. {node_id: "name"}
    :param cutoff: the depth to stop the search during the computation of the shortest paths between nodes of the
        hypergraph. Only paths of length <= cutoff are returned (position channel)
    :param c: a hyperparameter for generation of anchor-sets (position channel)
    :return:
    """

    def __init__(
        self,
        graph: Union[nx.Graph, nx.DiGraph],
        teams_composition: dict,
        teams_members: dict,
        teams_label: dict,
        nodes_not_belong_to_teams: list,
        nodes_attribute: pd.DataFrame = None,
        nodes_name: dict = None,
        cutoff: int = 8,
        c: int = 1,
    ):
        """Initialization."""
        # Define some parameters.
        self.graph = graph
        self.teams_composition = teams_composition
        self.teams_members = teams_members
        self.nodes_not_belong_to_teams = nodes_not_belong_to_teams
        self.nodes_attribute = nodes_attribute
        self.nodes_name = nodes_name
        self.cutoff = cutoff
        self.c = c

        self.hypergraph = None
        self.isolated_graph = None
        self.mask_teams = None

        # Define the number of teams.
        self.n_teams = len(teams_label)

        # Add teams composition (it should be already exist).
        nx.set_node_attributes(self.graph, self.teams_composition, "Team")
        # Add names attribute to node.
        if self.nodes_name:
            nx.set_node_attributes(self.graph, self.nodes_name, "Name")

        # Create torch data object.
        classes = list(teams_label.values())
        n_classes = len(np.unique(classes))

        self.data = HeteroData(n_teams=self.n_teams, y=torch.LongTensor(classes), n_classes=n_classes)

    def get_subgraphs(self):
        """Definition of a new graph formed by isolating sub-graphs (teams) from input graph."""
        # Define list for storing nodes attributes.
        features = []
        # Graph with isolated sub-graphs (teams).
        isolated_graph = nx.DiGraph() if self.graph.is_directed() else nx.Graph()
        # Subgraph isolation procedure: isolation of the teams from the entire graph G relabeling the nodes in order
        # to be unique (no overlapping members between teams) for each team.
        for team_id, members in self.teams_members.items():
            # Keep subgraph.
            subgraph = self.graph.subgraph(members)

            # Store attributes (if exist).
            if self.nodes_attribute is not None:
                nodes_attribute_team = self.nodes_attribute.xs(team_id, level="team", axis=0).loc[
                    list(subgraph.nodes())
                ]
                features.append(nodes_attribute_team.values)
            else:
                features.append(np.ones((len(members), 1)))

            # Relabel nodes.
            subgraph = nx.relabel.convert_node_labels_to_integers(
                subgraph, first_label=isolated_graph.number_of_nodes()
            )

            # Define new team composition without overlapping.
            nx.set_node_attributes(subgraph, {node: team_id for node in subgraph.nodes()}, "Team")

            # Append sub-graph.
            isolated_graph = nx.compose(isolated_graph, subgraph)

        # Remove self-loop.
        isolated_graph.remove_edges_from(nx.selfloop_edges(isolated_graph))

        features = np.concatenate(features)
        composition = list(nx.get_node_attributes(isolated_graph, "Team").values())

        self.isolated_graph = isolated_graph

        return features, composition

    def get_hypergraph(self):
        """Construction of the hypergraph where the edges are weighted based on connections in input graph."""
        # Define the number of hyper nodes.
        n_hypernodes = self.n_teams + len(self.nodes_not_belong_to_teams)
        # Assign an incremental new team for the isolated nodes.
        composition = self.teams_composition.copy()
        for i, node in enumerate(self.nodes_not_belong_to_teams):
            composition[node] = [self.n_teams + i]

        # Keep edges at hypernode level.
        edges_hypernode = []
        for src, dst in self.graph.edges():
            edges_hypernode.extend(list(itertools.product(composition[src], composition[dst])))

        # Get size of each hypernode: default 1.
        size_hypernodes = {i: 1 for i in range(n_hypernodes)}
        for team_id, members in self.teams_members.items():
            size_hypernodes[team_id] = len(members)

        # Create a hypergraph.
        hypergraph = nx.DiGraph() if self.graph.is_directed() else nx.Graph()
        hypergraph.add_nodes_from(range(n_hypernodes))

        # Add weighted edges.
        if self.graph.is_directed():
            new_edges = [(x, y, {"weight": v}) for (x, y), v in Counter(edges_hypernode).items()]
        else:
            new_edges = [
                (x, y, {"weight": v}) for (x, y), v in Counter(tuple(sorted(tup)) for tup in edges_hypernode).items()
            ]

        hypergraph.add_edges_from(new_edges)
        hypergraph.remove_edges_from(nx.selfloop_edges(hypergraph))

        # Set node attribute: size of the team.
        nx.set_node_attributes(hypergraph, size_hypernodes, "Size")

        # Save hypernodes ids that are teams (the first 'n_teams' ones).
        mask_teams = np.full(hypergraph.number_of_nodes(), False)
        mask_teams[range(self.n_teams)] = True

        # Get features of this graph.
        features = np.expand_dims(list(nx.get_node_attributes(hypergraph, "Size").values()), axis=1)

        self.hypergraph = hypergraph
        self.mask_teams = mask_teams

        return features

    def topology_channel(self):
        """Pre-processing phase for the 'topology' channel."""
        # Create sub-graph.
        x, composition = self.get_subgraphs()
        # Convert to torch data object.
        data = from_networkx(self.isolated_graph)

        # Store data on HeteroData object.
        self.data["topology"].x = torch.tensor(x)
        self.data["topology"].edge_index = data.edge_index
        self.data["topology"].composition = torch.LongTensor(composition)
        self.data["topology"].norm = True if self.nodes_attribute is not None else False

    def centrality_channel(self):
        """Pre-processing phase for the 'centrality' channel."""
        # Create hypergraph.
        x = self.get_hypergraph()
        # Convert to torch data object.
        data = from_networkx(self.hypergraph)

        # Edge weights and size nodes.
        edge_weight_centrality = data.weight.unsqueeze(1).detach().clone().numpy()

        self.data["centrality"].x = torch.tensor(x / max(x))
        self.data["centrality"].edge_index = data.edge_index
        self.data["centrality"].edge_weight = torch.tensor(edge_weight_centrality / max(edge_weight_centrality))
        self.data["centrality"].mask_teams = torch.tensor(self.mask_teams)

    def position_channel(self):
        """Pre-processing phase for the 'position' channel."""
        # Create hypergraph.
        if not self.hypergraph:
            self.get_hypergraph()

        # Compute distances between nodes into hypergraph.
        dists = all_pairs_shortest_path_length(self.hypergraph, cutoff=self.cutoff)
        # Get distances into matrix notation.
        rows, cols, data = zip(*[(row, col, 1 / (dists[row][col] + 1)) for row in dists for col in dists[row]])
        dists = coo_matrix(
            (data, (rows, cols)), shape=(self.hypergraph.number_of_nodes(), self.hypergraph.number_of_nodes())
        ).toarray()

        # Generation of anchor-sets.
        anchorsets = get_random_anchorset(self.hypergraph.number_of_nodes(), c=self.c)
        # Compute distances from anchor-sets and nodes.
        dists_max, dists_argmax = get_anchor_distances(dists, anchorsets)

        # Convert to torch data object.
        self.data["position"].x = torch.ones((self.hypergraph.number_of_nodes(), 1))
        self.data["position"].dists_max = torch.from_numpy(dists_max).float()
        self.data["position"].dists_argmax = torch.from_numpy(dists_argmax).long()
        self.data["position"].n_anchorsets = len(anchorsets)
        self.data["position"].mask_teams = torch.tensor(self.mask_teams)

    def __call__(self, topology: bool = True, centrality: bool = True, position: bool = True):
        """Get data object."""
        if topology:
            self.topology_channel()
        if centrality:
            self.centrality_channel()
        if position:
            self.position_channel()

        return self.data
