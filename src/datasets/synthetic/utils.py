import collections
from itertools import permutations

import networkx as nx
import numpy as np
import powerlaw


def directed_er_generator(n: int, m: int = 1, seed: int = 0) -> nx.DiGraph:
    """
    Create a directed Erdos Renyi graph.

    :param n: number of nodes to add into the graph
    :param m: number of possible random edges added to each node
    :param seed: the random seed
    :return: the graph with n nodes and random edges added
    """
    np.random.seed(seed)

    # Create directed graph.
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    # Add random edges to each node ('m' edges for each node).
    # N.B. These edges may be self-loop and/or multiple during the 'm' iterations.
    for i in range(m):
        src = np.arange(n)  # Source.
        dst = np.random.randint(low=0, high=n - 1, size=n)  # Destination.
        graph.add_edges_from(zip(src, dst))

    return graph


def team_maker(
    graph: nx.DiGraph, min_team_size: int = 5, poisson_mean: int = 5, m: int = 1, seed: int = 0
) -> tuple[nx.DiGraph, dict]:
    """
    This function assigns each node to a corresponding team (non-overlapping). No connectivity structure is
    added by this function. Some new nodes can be added to the provided graph if internal algorithm need them.

    :param graph: starting graph
    :param min_team_size: minimum team dimension
    :param poisson_mean: Poisson mean to add to provided minimum team size
    :param m: number of possible random edges added to each new possible node created inside this function
    :param seed: the random seed
    :return: the starting graph where could have been added new nodes and edges, and a dictionary with structure
        {team_id: [node_1, node_2, ...]}
    """
    np.random.seed(seed)

    # In order to obtain non overlapping teams we need to keep track of the nodes already inserted into teams.
    available_nodes = set(graph.nodes())

    # Teams dictionary.
    i = 0
    teams_composition = {}
    while len(available_nodes) > 0:
        # Select the team size.
        team_size = min_team_size
        team_size += np.random.poisson(poisson_mean)

        # Check if the available nodes are enough for creating a team with the current size.
        if len(available_nodes) >= team_size:
            # Choose 'team_size' nodes (without replacement) that will belong to this team.
            members = list(np.random.choice(list(available_nodes), size=team_size, replace=False))
            teams_composition[i] = members
            # Remove chosen nodes from the available ones.
            available_nodes = available_nodes.difference(set(members))
        # If the last remaining nodes are sufficient to build a reasonable team (according to 'min_team_size'),
        # we use these nodes as populating a last new team.
        elif (team_size > len(available_nodes)) & (len(available_nodes) >= min_team_size):
            # Choose the last available nodes that will belong to this team.
            members = list(available_nodes)
            teams_composition[i] = members
            break
        else:
            # Create new nodes in order to reach to build the team.
            max_node = max(graph.nodes()) + 1
            # Get the nodes that we lack to get to build the team.
            new_nodes = np.arange(max_node, max_node + (team_size - len(available_nodes)))
            # Add these nodes to the graph.
            graph.add_nodes_from(new_nodes)

            # Create random edge for this new nodes with other nodes into the network.
            for j in range(m):
                src = new_nodes
                dst = np.random.randint(low=0, high=max(new_nodes), size=len(new_nodes))
                # Add edges.
                graph.add_edges_from(zip(src, dst))

            available_nodes = available_nodes.union(set(new_nodes))
            # Create last team.
            members = list(available_nodes)
            teams_composition[i] = members
            break

        i += 1

    return graph, teams_composition


def clique_adder(graph: nx.DiGraph, teams_composition: dict, seed: int = 0) -> nx.DiGraph:
    """
    Add clique structure to each team inserting edges among nodes.

    :param graph: starting graph
    :param teams_composition: dictionary with structure {team_id: [node_1, node_2, ...]}
    :param seed: the random seed
    :return: starting graph with added clique structure (edges)
    """
    np.random.seed(seed)

    for team_id in teams_composition.keys():
        # Access to the team composition.
        members = teams_composition[team_id]
        # Clique.
        edges = permutations(members, 2)
        graph.add_edges_from(edges)

    return graph


def motif_adder(
    graph: nx.DiGraph,
    teams_composition: dict,
    motif_team_size_ratio: float = 0.8,
    motifs: tuple = (0, 1, 2),
    seed: int = 0,
) -> tuple[nx.DiGraph, dict]:
    """
    Add motif structures to the various teams into the graph. There exist 3 types of motif that can be selected.

    :param graph: starting graph
    :param teams_composition: dictionary with structure {team_id: [node_1, node_2, ...]}
    :param motif_team_size_ratio: the ratio of nodes involved into the repeatedly motifs addition
    :param motifs: the indexes of motifs to use for building internal connectivity
    :param seed:  the random seed
    :return: initial graph with added motif structure and dictionary with label/class {team_id: C}
    """
    np.random.seed(seed)

    # Split the teams into 3 groups of approximately equal length. Each group will have a different motif structure.
    teams_splits = np.array_split(list(teams_composition.keys()), len(motifs))

    teams_label = {}
    for motif, teams in zip(motifs, teams_splits):
        for team in teams:
            # Access team composition.
            members = teams_composition[team]
            n_motif = int(motif_team_size_ratio * len(members))

            if motif == 0:
                teams_label[team] = 0
                for _ in range(n_motif):
                    motif_not_added = True
                    while motif_not_added:
                        n1, n2, n3, n4 = np.random.choice(members, replace=False, size=4)
                        if (not graph.has_edge(n2, n1)) or (not graph.has_edge(n3, n1)) or (not graph.has_edge(n4, n1)):
                            graph.add_edges_from([(n2, n1), (n3, n1), (n4, n1)])
                            motif_not_added = False
            if motif == 1:
                teams_label[team] = 1
                for _ in range(n_motif):
                    motif_not_added = True
                    while motif_not_added:
                        n1, n2, n3, n4 = np.random.choice(members, replace=False, size=4)
                        if (not graph.has_edge(n1, n2)) or (not graph.has_edge(n2, n3)) or (not graph.has_edge(n3, n4)):
                            graph.add_edges_from([(n1, n2), (n2, n3), (n3, n4)])
                            motif_not_added = False
            if motif == 2:
                teams_label[team] = 2
                for _ in range(n_motif):
                    motif_not_added = True
                    while motif_not_added:
                        n1, n2, n3 = np.random.choice(members, replace=False, size=3)
                        if (not graph.has_edge(n1, n2)) or (not graph.has_edge(n2, n1)) or (not graph.has_edge(n1, n3)):
                            graph.add_edges_from([(n1, n2), (n2, n1), (n1, n3)])
                            motif_not_added = False

    return graph, teams_label


def degree_maker(
    graph: nx.DiGraph,
    teams_composition: dict,
    n_classes: int,
    separation: int,
    use_powerlaw: bool = True,
    method: str = "khot",
    degree: str = "in",
    seed: int = 0,
    **kwargs
) -> tuple[nx.DiGraph, dict, list]:
    """
    Assign to each team a particular in-degree/out-degree distribution from outside the team. This is possible adding
    new edges into the network.

    :param graph: starting graph
    :param teams_composition: dictionary with structure {team_id: [node_1, node_2, ...]}
    :param n_classes: number of different classes of degree distributions
    :param separation: the separation value between centers of the normal distributions of the classes
    :param use_powerlaw: if you want to crate a powerlaw distribution for the higher class
    :param method: the name of the method with which decide the nodes of the teams that receive edges from the outside;
        allowed methods: 'arithmetic', 'geometric', 'uniform', 'triangular', 'khot'
    :param degree: the type of degree to consider in adding edges into the network; allowed degrees: 'in', 'out'
    :param seed: the random seed
    :param kwargs: extra parameters
    :return: starting graph with added new edges, dictionary with structure {team_id: label} and list of more
        discriminative nodes (based on selected degree type) for each team
    """
    np.random.seed(seed)

    # Define all nodes into the graph.
    all_nodes = set(graph.nodes())
    # Define the max class.
    max_class = n_classes - 1

    # Start from 'n_classes' to avoid negative values.
    centers = np.ones(n_classes) * n_classes
    # Separation of the centers and the corresponding distributions.
    centers = centers + np.arange(n_classes, dtype=int) * separation
    distribution = [np.random.normal] * n_classes

    # Split the teams into approximately equal n_classes groups. We obtain three list of team ids.
    all_teams = list(teams_composition.keys())
    np.random.shuffle(all_teams)
    teams_splits = np.array_split(all_teams, n_classes)

    # Save dictionary: {team_id: label}.
    teams_label = {}
    discriminative_nodes = []
    for split, teams in enumerate(teams_splits):
        for team in teams:
            teams_label[team] = split
            members = teams_composition[team]
            # Get the number of new edge to add based on the corresponding distribution.
            if (split == max_class) & use_powerlaw:
                n_links = int(
                    powerlaw.Power_Law(xmin=centers[split], parameters=[3], discrete=True).generate_random()[0]
                )
            else:
                n_links = max(2, int(distribution[split](loc=centers[split])))

            # Assign with selected rules.
            alpha = alpha_generator(len(members), method=method, **kwargs)
            probs = np.random.dirichlet(alpha)

            if degree == "in":
                # Keep possible sources: all nodes of the graph except the nodes that belong to the current team.
                src = list(all_nodes.difference(set(members)))
                src = np.random.choice(src, size=n_links, replace=False)
                # Keep destination: some nodes of the current team based on the above rule.
                dst = np.random.choice(members, p=probs, size=n_links)
                discriminative_nodes.append(set(dst))
            elif degree == "out":
                # Keep destination: all nodes of the graph except the nodes that belong to the current team.
                dst = list(all_nodes.difference(set(members)))
                dst = np.random.choice(dst, size=n_links, replace=False)
                # Keep possible sources: some nodes of the current team based on the above rule.
                src = np.random.choice(members, p=probs, size=n_links)
                discriminative_nodes.append(set(src))
            else:
                raise ValueError("The provided 'degree' parameter not exist")

            graph.add_edges_from(zip(src, dst))

    # Ordered dictionary.
    teams_label = dict(collections.OrderedDict(sorted(teams_label.items())))

    return graph, teams_label, discriminative_nodes


def alpha_generator(n, method="khot", d=2, pw=2, bl=10, k=1):
    """The method with which decide how the nodes of the teams receive edges from the outside."""
    if method == "arithmetic":
        alpha = arithmetic(n, d)
    elif method == "geometric":
        alpha = geometric(n, pw)
    elif method == "uniform":
        alpha = uniform(n, bl)
    elif method == "triangular":
        alpha = triangular(n, pw)
    elif method == "khot":
        alpha = khot(n, k, bl)
    else:
        raise ValueError("The provided 'method' parameter not exist")

    return alpha


def arithmetic(n_terms, d=2):
    seq = np.ones(n_terms, dtype=int)
    increase = np.arange(n_terms, dtype=int) * d
    return seq + increase


def geometric(n_terms, pw=2):
    seq = np.ones(n_terms, dtype=int)
    increase = np.arange(n_terms, dtype=int) ** pw
    return seq + increase


def uniform(n_terms, bl=10):
    return np.ones(n_terms, dtype=int) * bl


def triangular(n_terms, pw=1):
    r = np.arange(n_terms)
    kernel1d = (n_terms + 1 - np.abs(r - r[::-1])) / 2
    return (kernel1d**pw).astype(int)


def khot(n_terms, k=3, bl=10):
    seq = np.zeros(n_terms) + 0.01
    seq[:k] = np.ones(k) * bl
    return seq
