import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data

from user_utility_functions import get_ipr


def _graph_to_data(G):
    A = nx.adjacency_matrix(G)
    ipr = get_ipr(A.todense())
    A_tensor = torch.tensor(A.todense())

    row, col = torch.nonzero(A_tensor, as_tuple=True)
    edge_index = torch.stack([row, col], dim=0)

    x = customized_features(G)
    y = torch.tensor(ipr)

    return Data(x=x, edge_index=edge_index.contiguous(), edge_attr=None, y=y, pos=None)


def star(ngraphs, num_features, start, end):
    dataset = []
    for _ in range(ngraphs):
        n = np.random.randint(start, end)
        G = nx.star_graph(n - 1)
        dataset.append(_graph_to_data(G))
    return dataset


def cycle(ngraphs, num_features, start, end):
    dataset = []
    for _ in range(ngraphs):
        n = np.random.randint(start, end)
        G = nx.cycle_graph(n)
        dataset.append(_graph_to_data(G))
    return dataset


def wheel(ngraphs, num_features, start, end):
    dataset = []
    for _ in range(ngraphs):
        n = np.random.randint(start, end)
        G = nx.wheel_graph(n)
        dataset.append(_graph_to_data(G))
    return dataset


def path(ngraphs, num_features, start, end):
    dataset = []
    for _ in range(ngraphs):
        n = np.random.randint(start, end)
        G = nx.path_graph(n)
        dataset.append(_graph_to_data(G))
    return dataset


def erdos(numgraphs, num_features, start, end):
    dataset = []
    for _ in range(numgraphs):
        n = np.random.randint(start, end)
        p_c = (np.log(n) + 5) / n
        p = np.random.uniform(p_c, p_c + 0.1)
        connected = False
        while not connected:
            G = nx.fast_gnp_random_graph(n, p)
            connected = nx.is_connected(G)
        dataset.append(_graph_to_data(G))
    return dataset


def barabasi(numgraphs, num_features, start, end):
    dataset = []
    for _ in range(numgraphs):
        n = np.random.randint(start, end)
        m = np.random.randint(2, 4)
        G = nx.barabasi_albert_graph(n, m)
        dataset.append(_graph_to_data(G))
    return dataset


# -Customized feature generation--
def customized_features(G):
    # Feature 1: Degree centrality
    degree_centrality = torch.tensor(
        list(nx.degree_centrality(G).values()), dtype=torch.float
    ).view(-1, 1)

    # Feature 2: Clustering Coefficient
    clustering_coeff = torch.tensor(
        [nx.clustering(G, node) for node in G.nodes()], dtype=torch.float
    ).view(-1, 1)

    # Feature 3: Betweenness Centrality
    betweenness = torch.tensor(
        list(nx.betweenness_centrality(G).values()), dtype=torch.float
    ).view(-1, 1)

    # Feature 4: PageRank for nodes
    pagerank = nx.pagerank(G)
    pagerank_tensor = torch.tensor(
        [pagerank[node] for node in G.nodes()], dtype=torch.float
    ).view(-1, 1)

    # Feature 5: Closeness for nodes
    closeness = torch.tensor(
        list(nx.closeness_centrality(G).values()), dtype=torch.float
    ).view(-1, 1)

    x = torch.cat(
        [degree_centrality, clustering_coeff, betweenness, pagerank_tensor, closeness],
        dim=1,
    )
    return x
