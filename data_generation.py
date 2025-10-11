import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from random import shuffle
from torch.nn import Linear
from torch_geometric.data import Data  # Dataset
from numpy import linalg as LA
from user_utility_functions import *

def star(ngraphs, num_features, start, end):

  dataset = []

  for i in range(0, ngraphs):
      
      n = np.random.randint(start, end)
      a = nx.star_graph(n-1)
      A = nx.adjacency_matrix(a)
      ipr = get_ipr(A.todense())
      A = torch.tensor(A.todense()) # Creating Adjacency matrix from graph

      row, col = torch.nonzero(A, as_tuple = True)
      edge_index1 = torch.stack([row, col], dim = 0) # Creating Edge_index from graph

      x = customized_features(a)  # Passing the networkx graph object
      y = torch.tensor(ipr)   # Creating Labels

      data = Data(x = x, edge_index = edge_index.contiguous(), edge_attr = None, y = y, pos =  None)
      dataset.append(data)
      
  return dataset

def cycle(ngraphs, num_features, start, end):

  dataset = []

  for i in range(0, ngraphs):
      n = np.random.randint(start, end)
      a = nx.cycle_graph(n)
      A = nx.adjacency_matrix(a)
      ipr = get_ipr(A.todense())
      A = torch.tensor(A.todense()) # Creating Adjacency matrix from graph

      row, col = torch.nonzero(A, as_tuple = True)
      edge_index = torch.stack([row, col], dim = 0) # Creating Edge_index from graph

      x = customized_features(a)  # Passing the networkx graph object
      y = torch.tensor(ipr)   # Creating Labels

      data = Data(x = x, edge_index = edge_index.contiguous(), edge_attr = None, y = y, pos =  None)
      dataset.append(data)

  return dataset

def wheel(ngraphs, num_features, start, end):

  dataset = []

  for i in range(0, ngraphs):
      n = np.random.randint(start, end)
      a = nx.wheel_graph(n)
      A = nx.adjacency_matrix(a)
      ipr = get_ipr(A.todense())
      A = torch.tensor(A.todense()) # Creating Adjacency matrix from graph

      row, col = torch.nonzero(A, as_tuple = True)
      edge_index = torch.stack([row, col], dim = 0) # Creating Edge_index from graph

      x = customized_features(a)  # Passing the networkx graph object
      y = torch.tensor(ipr)   # Creating Labels

      data = Data(x = x, edge_index = edge_index.contiguous(), edge_attr = None, y = y, pos =  None)
      dataset.append(data)
      
  return dataset

def path(ngraphs, num_features, start, end):

  dataset = []

  for i in range(0, ngraphs):
      n = np.random.randint(start, end)
      a = nx.path_graph(n)
      A = nx.adjacency_matrix(a)
      ipr = get_ipr(A.todense())
      A = torch.tensor(A5.todense()) # Creating Adjacency matrix from graph

      row, col = torch.nonzero(A, as_tuple = True)
      edge_index = torch.stack([row, col], dim = 0) # Creating Edge_index from graph

      x = customized_features(a)  # Passing the networkx graph object
      y = torch.tensor(ipr)   # Creating Labels

      data = Data(x = x, edge_index = edge_index.contiguous(), edge_attr = None, y = y, pos =  None)
      dataset.append(data)

  return dataset

def erdos(numgraphs, num_features, start, end):
    
    dataset = []
    
    for i in range(0, numgraphs):
        n = np.random.randint(start, end)
        p_c = (np.log(n)+5)/n
        p = np.random.uniform(p_c, p_c + 0.1)
        connected = False
        while not connected:
            G = nx.fast_gnp_random_graph(n, p)
            connected = nx.is_connected(G)

    A = nx.adjacency_matrix(G)
    ipr = get_ipr(Ag.todense())
    A = torch.tensor(A.todense())   # Creating Adjacency matrix from graph
    row, col = torch.nonzero(A, as_tuple = True)
    edge_indexg = torch.stack([row, col], dim = 0) # Creating Edge_index from graph
    x = customized_features(G)  # Passing the networkx graph object
    y = torch.tensor(ipr)   # Creating Labels
    data = Data(x = x, edge_index = edge_indexg.contiguous(), edge_attr = None, y = y, pos =  None)
    dataset.append(data)

    return dataset


def barabasi(numgraphs, num_features, start, end):
  
  dataset = []

  for i in range(0, numgraphs):
      n = np.random.randint(start, end)
      m = np.random.randint(2, 4)
      G = nx.barabasi_albert_graph(n, m)
      A = nx.adjacency_matrix(G)

      ipr = get_ipr(A.todense())
      A = torch.tensor(A.todense())   # Creating Adjacency matrix from graph
      row, col = torch.nonzero(A, as_tuple = True)
      edge_index = torch.stack([row, col], dim = 0) # Creating Edge_index from graph

      x = customized_features(G)
      y = torch.tensor(ipr)  # Creating Labels

      data = Data(x = x, edge_index = edge_index.contiguous(), edge_attr = None, y = y, pos =  None)
      dataset.append(data)
      
  return dataset


################################# Customized feature generation ################################
def customized_features(G):

    # Feature 1: Degree centrality
    degree_centrality = torch.tensor(list(nx.degree_centrality(G).values()), dtype = torch.float).view(-1, 1)

    # Feature 2: Clustering Coefficient
    clustering_coeff = torch.tensor([nx.clustering(G, node) for node in G.nodes()], dtype = torch.float).view(-1, 1)

    # Feature 3: Betweenness Centrality
    betweenness = torch.tensor(list(nx.betweenness_centrality(G).values()), dtype = torch.float).view(-1, 1)

    # Feature 4: PageRank for nodes
    pagerank = nx.pagerank(G)

    # Convert PageRank dictionary to a tensor
    pagerank_tensor = torch.tensor([pagerank[node] for node in G.nodes()], dtype=torch.float).view(-1, 1)
    
    # Feature 5: Closeness for nodes
    closeness = torch.tensor(list(nx.closeness_centrality(G).values()), dtype=torch.float).view(-1, 1)

    x = torch.cat([degree_centrality, clustering_coeff, betweenness, pagerank_tensor, closeness], dim = 1)

    return x


   
  
    
