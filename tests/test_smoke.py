import networkx as nx
import numpy as np
from torch_geometric.loader import DataLoader

from data_generation import (
    barabasi,
    customized_features,
    cycle,
    erdos,
    path,
    star,
    wheel,
)
from model import GAT, GCN, GraphTranformer
from user_utility_functions import (
    decides_class_batches,
    decides_predicted_classes,
    decides_predicted_multi_classes,
    get_ipr,
)


def test_ipr_and_features():
    G = nx.barabasi_albert_graph(15, 2)
    A = nx.adjacency_matrix(G).todense()
    ipr = get_ipr(A)
    assert isinstance(float(ipr), float)
    assert 0.0 < ipr <= 1.0

    x = customized_features(G)
    assert x.shape == (15, 5)


def test_all_graph_generators():
    generators = [star, cycle, wheel, path, erdos, barabasi]
    for gen_fn in generators:
        dataset = gen_fn(2, 5, 6, 10)
        assert len(dataset) == 2
        assert dataset[0].x.shape[1] == 5
        assert dataset[0].edge_index.shape[0] == 2


def test_gnn_and_batch_classification():
    dataset = cycle(ngraphs=4, num_features=5, start=6, end=10)
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))

    for ModelClass in [GCN, GAT, GraphTranformer]:
        model = ModelClass(hidden_channels=16)
        out = model(batch.x, batch.edge_index, batch.batch)
        assert out.shape == (2, 1)

        labels, graph_size, pred_labels = decides_class_batches(batch, out)
        assert len(labels) == 2
        assert len(graph_size) == 2
        assert len(pred_labels) == 2


def test_threshold_metrics():
    actual = [0.01, 0.08, 0.25]
    pred = [0.02, 0.10, 0.30]

    bin_actual, bin_pred = decides_predicted_classes(actual, pred)
    assert np.array_equal(bin_actual, np.array([0.0, 1.0, 1.0]))
    assert np.array_equal(bin_pred, np.array([0.0, 1.0, 1.0]))

    multi_actual, multi_pred = decides_predicted_multi_classes(actual, pred)
    assert np.array_equal(multi_actual, np.array([0.0, 1.0, 2.0]))
    assert np.array_equal(multi_pred, np.array([0.0, 1.0, 2.0]))

