import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from preprocessing import load_and_preprocess
from dataset import RSSIDataset
from models import CNN1D, ResNet1D

FRAME_SIZE = 100   # 10s bei 10pkt/s
OVERLAP = 0.5   # 50% overlap #todo vary and test whats best
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SCENARIO = "node"   # Classification of the transmittor
STRATEGY = 1        # 1 = 75/25 Split, 2 = Leave-One-Env-Out # TODO validate


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_park")


def get_splits_strategy1(dataset):
    """75% train, 25% test out of the same dataset"""
    n = len(dataset)
    n_train = int(0.75 * n)
    return random_split(dataset, [n_train, n - n_train])


def get_splits_strategy2(x, labels_env, labels_node, test_env="park"):
    """Train of 4 Environments, Test of the 5th (Leave-One-Out)"""
    target = labels_env if SCENARIO == "env" else labels_node
    train_idx = np.where(labels_env != test_env)[0]
    test_idx = np.where(labels_env == test_env)[0]
    return train_idx, test_idx


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        out = model(x_batch)
        total_loss += criterion(out, y_batch).item()
        correct += (out.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def run(model_name="cnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Daten laden
    X, labels_env, labels_node = load_and_preprocess(
        DATA_DIR, frame_size=FRAME_SIZE, overlap=OVERLAP
    )
    labels = labels_env if SCENARIO == "env" else labels_node
    dataset = RSSIDataset(X, labels)
    num_classes = len(dataset.classes)
    print(f"Classes: {dataset.classes}  ({num_classes} total)")
    print(f"Total samples: {len(dataset)}")

    # Train/Test split
    if STRATEGY == 1:
        train_ds, test_ds = get_splits_strategy1(dataset)
    else:
        train_idx, test_idx = get_splits_strategy2(X, labels_env, labels_node)
        train_ds = Subset(dataset, train_idx)
        test_ds = Subset(dataset, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model
    if model_name == "cnn":
        model = CNN1D(num_classes, FRAME_SIZE).to(device)
    else:
        model = ResNet1D(num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Acc':>8}")
    print("-" * 45)
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        if epoch % 5 == 0:
            print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>9.3f} | {te_acc:>8.3f}")

    torch.save(model.state_dict(), f"{model_name}_scenario_{SCENARIO}_s{STRATEGY}.pt")
    print(f"\nFinal Test Accuracy: {te_acc:.4f}")


if __name__ == "__main__":
    run("cnn")
    run("resnet")