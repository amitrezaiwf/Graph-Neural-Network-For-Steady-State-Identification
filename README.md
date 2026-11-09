#### This repository contains simulated data generation and graph neural network  models developed for the work presented in "Predicting steady-state behavior in complex networks with graph neural networks" [Phys. Rev. E. --> https://journals.aps.org/pre/accepted/10.1103/rkp2-3d1r].

#### Docker Setup

We can build and run this project in an isolated container using Docker.

##### Build the Image
```bash
docker build -t gnn-steady-state .
```

##### Run Tests
```bash
docker run --rm gnn-steady-state
```

##### Generate Data & Train Models
Mount the current directory to save generated data and model outputs locally:
```bash
docker run --rm -v \$(pwd):/app gnn-steady-state python data_generation.py
docker run --rm -v \$(pwd):/app gnn-steady-state python train.py
```

