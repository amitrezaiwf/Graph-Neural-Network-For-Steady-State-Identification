# Predicting Steady-State Behavior in Complex Networks with Graph Neural Networks

[![Paper](https://shields.io)](https://journals.aps.org/pre/abstract/10.1103/rkp2-3d1r)
[![License: MIT](https://shields.io)](https://opensource.org)
[![Python 3.10+](https://shields.io)](https://python.org)

Official implementation and synthetic data generation pipeline for **"Predicting steady-state behavior in complex networks with graph neural networks"** (*Priodyuti Pradhan and Amit Reza, Physical Review E 112, 054303*).

## Overview
This repository provides a Graph Neural Network (GNN) framework to predict steady-state regimes in nonlinear complex networks directly from topology and system parameters, bypassing costly numerical integration.

### Key Features
* **Synthetic Data Generation:** Simulates network dynamics to create labeled steady-state datasets.
* **GNN Architecture:** Message-passing neural network capturing local and global steady-state properties.
* **Full Reproducibility:** Includes Docker support and automated CI tests.

---

## Methodology & Pipeline

This framework predicts steady-state eigenvector localization in complex networks—measured by the **Inverse Participation Ratio (IPR)** of the adjacency matrix's principal eigenvector—directly from local and global topological descriptors.

### 1. Synthetic Graph Generation & Feature Extraction (`data_generation.py`)
* **Network Topologies:** Generates synthetic graphs across diverse structural families, including **Star**, **Cycle**, **Wheel**, **Path**, **Erdős–Rényi** ($G(n,p)$), and **Barabási–Albert** scale-free networks.
* **Node Input Features ($5$ dimensions):** Extracts five structural centrality metrics per node:
  1. Degree Centrality
  2. Clustering Coefficient
  3. Betweenness Centrality
  4. PageRank
  5. Closeness Centrality
* **Ground-Truth Target:** Computes the exact IPR from the leading eigenvector of the graph adjacency matrix (`user_utility_functions.py`).

### 2. Graph Neural Network Models (`model.py`)
All architectures take 5-dimensional node features, apply two message-passing layers with `ReLU` activations, aggregate graph embeddings via `global_mean_pool`, apply dropout ($p = 0.6$), and output a scalar IPR prediction via a linear readout layer:
* **GCN (`GCN`):** Two `GCNConv` layers (`hidden_channels = 64`).
* **GAT (`GAT`):** Multi-head `GATConv` layer ($4$ attention heads, concatenated) followed by a single-head `GATConv` layer.
* **Graph Transformer (`GraphTranformer`):** Multi-head `TransformerConv` layer ($2$ heads) followed by a single-head `TransformerConv` layer.

### 3. Training & Localization Classification (`train.py`, `test.py`)
* **Training Objective:** Optimizes logarithmic Mean Squared Error ($\text{MSE}(\log(1 + \hat{y}), \log(1 + y))$) using `AdamW` ($\text{lr} = 10^{-3}$, weight decay $= 5 \times 10^{-4}$).
* **Localization Regime Classification:** Maps continuous IPR predictions into binary (delocalized vs. localized using size-dependent $1/N + \epsilon$ or $\tau_1 = 0.05$ thresholds) and multi-class localization regimes ($\tau_1 = 0.05$, $\tau_2 = 0.20$), generating confusion matrices and relative error diagnostics.




## Quickstart with Docker

```bash
# Build image
docker build -t gnn-steady-state .

# Run automated tests
docker run --rm gnn-steady-state

# Generate data and train models (mounting local directory)
docker run --rm -v \$(pwd):/app gnn-steady-state python -m gnn_steady_state.data_generation
docker run --rm -v \$(pwd):/app gnn-steady-state python -m gnn_steady_state.train
```

---

## Local Installation

```bash
git clone https://github.com
cd Graph-Neural-Network-For-Steady-State-Identification
pip install -r requirements.txt
pip install -e .
```

---

## Citation

```bibtex
@article{pradhan2025predicting,
  title={Predicting steady-state behavior in complex networks with graph neural networks},
  author={Pradhan, Priodyuti and Reza, Amit},
  journal={Physical Review E},
  volume={112},
  number={5},
  pages={054303},
  year={2025},
  publisher={APS},
  doi = {https://doi.org/10.1103/rkp2-3d1r}
}

```

