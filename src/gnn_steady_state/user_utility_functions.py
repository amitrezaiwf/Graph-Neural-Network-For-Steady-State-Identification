import numpy as np
import torch
from numpy import linalg as LA

# -

# Batch size
b_size = 64


def get_ipr(A):
    e_val, e_vec = LA.eigh(A)
    d = np.shape(e_vec)
    n = d[0]
    Ev1 = e_vec[:, n - 1]

    IPR = 0.0
    for i in range(n):
        IPR = IPR + pow(Ev1[i], 4)
    del e_val
    del e_vec
    del Ev1

    return IPR


def get_label_batches(batch_dataset):
    labels = []
    for i in range(len(batch_dataset)):
        data = batch_dataset[i]
        nodes = data.num_nodes
        m_edges = data.num_edges
        edges = data.edge_index
        M = edges.cpu().detach().numpy()
        A = np.zeros((nodes, nodes), dtype=int)
        for j in range(m_edges):
            x = M[0][j]
            y = M[1][j]
            A[x][y] = 1
            A[y][x] = 1
        ipr = get_ipr(A)

        labels.append(ipr)
        del A, M, data

    return torch.tensor(labels)


def decides_class_batches(batch_dataset, out_train):
    labels = []
    graph_size = []
    predicted_labels = []

    # print(len(batch_dataset))
    for i in range(len(batch_dataset)):
        predicted_ipr = out_train[i, 0].detach().numpy().tolist()
        data = batch_dataset[i]
        nodes = data.num_nodes
        ipr = data.y
        # print(ipr,predicted_ipr)
        epsilon = 0.003
        flag = 0
        if ipr >= (1 / nodes + epsilon):
            flag = 1

        predicted_flag = 0
        if np.abs(predicted_ipr) >= (
            1 / nodes + epsilon
        ):  # predicted_ipr may be negative
            predicted_flag = 1

        labels.append(flag)
        predicted_labels.append(predicted_flag)
        graph_size.append(nodes)
    return (
        torch.tensor(labels),
        torch.tensor(graph_size),
        torch.tensor(predicted_labels),
    )


def decides_predicted_classes(actual_test_ipr, pred_test_ipr):
    actual_test_class = np.zeros((len(actual_test_ipr),))
    pred_test_class = np.zeros((len(actual_test_ipr),))
    tau1 = 0.05
    eps = 0.000001

    for i in range(len(actual_test_ipr)):
        if actual_test_ipr[i] >= (tau1 + eps):  # Issue in float vale access
            actual_test_class[i] = 1
        else:
            actual_test_class[i] = 0

    for i in range(len(pred_test_ipr)):
        if pred_test_ipr[i] >= (tau1 + eps):
            pred_test_class[i] = 1
        else:
            pred_test_class[i] = 0

    return actual_test_class, pred_test_class


def decides_predicted_multi_classes(actual_test_ipr, pred_test_ipr):
    actual_test_class = np.zeros((len(actual_test_ipr),))
    pred_test_class = np.zeros((len(actual_test_ipr),))
    tau1 = 0.05
    tau2 = 0.2
    eps = 0.000001

    for i in range(len(actual_test_ipr)):
        if actual_test_ipr[i] >= (tau2 + eps):
            actual_test_class[i] = 2

        elif actual_test_ipr[i] > (tau1 - eps) and actual_test_ipr[i] < (tau2 + eps):
            actual_test_class[i] = 1

        else:
            actual_test_class[i] = 0

    for i in range(len(pred_test_ipr)):
        if pred_test_ipr[i] >= (tau2 + eps):
            pred_test_class[i] = 2

        elif pred_test_ipr[i] > (tau1 - eps) and pred_test_ipr[i] < (tau2 + eps):
            pred_test_class[i] = 1

        else:
            pred_test_class[i] = 0

    return actual_test_class, pred_test_class
