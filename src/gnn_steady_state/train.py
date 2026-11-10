import argparse
import random

import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader

from gnn_steady_state.model import GAT, GCN, GraphTranformer
from gnn_steady_state.user_utility_functions import decides_class_batches


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN for IPR prediction.")
    parser.add_argument(
        "--arch",
        type=str,
        default="gcn",
        choices=["gcn", "gat", "transformer"],
        help="Model architecture to train (default: gcn)",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--data-path",
        type=str,
        default="train_dataset_sample.pt",
        help="Path to training dataset",
    )
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, criterion, is_transformer=False):
    model.train()
    total_loss = 0.0
    actual_ipr = []
    pred_ipr = []

    for data in train_loader:
        out_train = model(data.x, data.edge_index, data.batch)
        decides_class_batches(data, out_train)

        actual_ipr.extend(data.y.detach().numpy().tolist())
        out_flat = out_train.flatten()
        pred_ipr.extend(out_flat.detach().numpy().tolist())

        if is_transformer:
            out_flat = out_flat.to(torch.float32)
            target = data.y.to(torch.float32)
        else:
            target = data.y

        loss = criterion(torch.log(1 + out_flat), torch.log(1 + target)).float()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss / len(train_loader), actual_ipr, pred_ipr


def main():
    args = parse_args()
    torch.manual_seed(12275)

    train_data = torch.load(
        args.data_path, map_location=torch.device("cpu"), weights_only=False
    )
    random.shuffle(train_data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    if args.arch == "gcn":
        model = GCN(hidden_channels=args.batch_size)
    elif args.arch == "gat":
        model = GAT(hidden_channels=args.batch_size)
    else:
        model = GraphTranformer(hidden_channels=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    train_losses = []
    epochs = []

    for epoch in range(1, args.epochs + 1):
        loss, actual_ipr, pred_ipr = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            is_transformer=(args.arch == "transformer"),
        )
        train_losses.append(loss)
        epochs.append(epoch)
        print(f"{epoch} Train Loss: {loss:.6f}")

    torch.save(model.state_dict(), f"{args.arch.upper()}_model.pth")

    plt.figure(figsize=(10, 6))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, train_losses, linewidth=2)
    plt.savefig("train_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        actual_ipr,
        actual_ipr,
        color="#5B2C6F",
        marker="o",
        markersize=16,
        linestyle="None",
        markerfacecolor="none",
    )
    plt.plot(
        actual_ipr,
        pred_ipr,
        color="#DE3163",
        marker="^",
        markersize=8,
        linestyle="None",
        markerfacecolor="none",
    )
    plt.legend(["actual", "predicted"], loc="lower right")
    plt.savefig("train_actual_vs_pred.png", dpi=300)
    plt.close()
    print("Completion of training process...\n")


if __name__ == "__main__":
    main()

