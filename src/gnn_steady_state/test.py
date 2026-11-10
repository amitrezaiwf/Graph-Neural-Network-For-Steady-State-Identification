import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch_geometric.loader import DataLoader

from gnn_steady_state.model import GCN
from gnn_steady_state.user_utility_functions import (
    decides_class_batches,
    decides_predicted_classes,
    decides_predicted_multi_classes,
)


def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    actual_test_ipr = []
    pred_test_ipr = []

    with torch.no_grad():
        for data in test_loader:
            out_test = model(data.x, data.edge_index, data.batch)
            decides_class_batches(data, out_test)

            out_flat = out_test.flatten()
            actual_test_ipr.extend(data.y.detach().numpy().tolist())
            pred_test_ipr.extend(out_flat.detach().numpy().tolist())

            test_loss = criterion(torch.log(1 + out_flat), torch.log(1 + data.y))
            total_loss += test_loss.item()

    return total_loss / len(test_loader), actual_test_ipr, pred_test_ipr


def save_confusion_matrix(y_true, y_pred, base_name):
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f"{base_name}_raw.png", dpi=300)
    plt.close()

    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt=".0f", cmap="crest", cbar=True)
    cbar = plt.gca().collections.colorbar
    cbar.set_label("Percentage")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(f"{base_name}_pct.png", dpi=300)
    plt.close()


def main():
    torch.manual_seed(12275)
    b_size = 64
    hc = 64

    test_data = torch.load(
        "test_dataset_samples.pt", map_location=torch.device("cpu"), weights_only=False
    )
    random.shuffle(test_data)
    test_loader = DataLoader(test_data, batch_size=b_size, shuffle=False)

    load_model = GCN(hidden_channels=hc)
    load_model.load_state_dict(torch.load("GCN_model.pth"))
    criterion = torch.nn.MSELoss()

    _, actual_test_ipr, pred_test_ipr = evaluate(load_model, test_loader, criterion)

    plt.figure(figsize=(6, 4))
    plt.plot(
        actual_test_ipr,
        actual_test_ipr,
        color="#DE3163",
        marker="s",
        markersize=8,
        linestyle="None",
        markerfacecolor="none",
    )
    plt.plot(
        actual_test_ipr,
        pred_test_ipr,
        color="#229954",
        marker="^",
        markersize=8,
        linestyle="None",
        markerfacecolor="none",
    )
    plt.legend(["test-actual", "test-predicted"], loc="lower right")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.rcParams.update({"font.size": 22})
    plt.savefig("test_actual_vs_prediction.png", dpi=300)
    plt.close()

    actual_arr = np.array(actual_test_ipr)
    pred_arr = np.array(pred_test_ipr)
    err1 = (actual_arr - pred_arr) / actual_arr
    err2 = pred_arr / actual_arr

    for data_arr, fname in [
        (actual_arr - pred_arr, "abs_err.png"),
        (err1 * 100, "per_err.png"),
        (err2, "ratio_err.png"),
    ]:
        plt.figure(figsize=(6, 4))
        plt.hist(data_arr, bins=20)
        plt.savefig(fname, dpi=300)
        plt.close()

    actual_bin, pred_bin = decides_predicted_classes(actual_test_ipr, pred_test_ipr)
    save_confusion_matrix(actual_bin, pred_bin, "confusion_matrix")

    actual_multi, pred_multi = decides_predicted_multi_classes(
        actual_test_ipr, pred_test_ipr
    )
    save_confusion_matrix(actual_multi, pred_multi, "confusion_matrix_multiclass")


if __name__ == "__main__":
    main()

