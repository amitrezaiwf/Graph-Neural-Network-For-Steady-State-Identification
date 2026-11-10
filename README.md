# Predicting Steady-State Behavior in Complex Networks with Graph Neural Networks

[![Paper](https://shields.io)](https://aps.org)
[![License: MIT](https://shields.io)](https://opensource.org)
[![Python 3.10+](https://shields.io)](https://python.org)

Official implementation and synthetic data generation pipeline for **"Predicting steady-state behavior in complex networks with graph neural networks"** (*Physical Review E*).

## Overview
This repository provides a Graph Neural Network (GNN) framework to predict steady-state regimes in nonlinear complex networks directly from topology and system parameters, bypassing costly numerical integration.

### Key Features
* **Synthetic Data Generation:** Simulates network dynamics to create labeled steady-state datasets.
* **GNN Architecture:** Message-passing neural network capturing local and global steady-state properties.
* **Full Reproducibility:** Includes Docker support and automated CI tests.

---

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

