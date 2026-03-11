import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from preprocessing import load_and_preprocess
from dataset import RSSIDataset
from models import CNN1D, ResNet1D
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import csv

FRAME_SIZE = 100   # 10s bei 10pkt/s
OVERLAP = 0.5   # 50% overlap #todo vary and test whats best
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
SCENARIO = ("node")   # "node" or "environment"
STRATEGY = 1        # 1 = 75/25 Split, 2 = Leave-One-Env-Out # TODO validate


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_data")


def get_splits_strategy1(dataset):
    """75% train, 25% test out of the same dataset"""
    n = len(dataset)
    n_train = int(0.75 * n)
    return random_split(dataset, [n_train, n - n_train])


def get_splits_strategy2(x, labels_env, labels_node, test_env="river"):
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


def run_with_export(model_name="cnn"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, labels_env, labels_node = load_and_preprocess(
        DATA_DIR, frame_size=FRAME_SIZE, overlap=OVERLAP
    )
    labels = labels_env if SCENARIO == "env" else labels_node
    dataset = RSSIDataset(X, labels)
    num_classes = len(dataset.classes)

    train_ds, test_ds = get_splits_strategy1(dataset)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = (CNN1D(num_classes, FRAME_SIZE) if model_name == "cnn" else ResNet1D(num_classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # training and history is recorded
    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        if epoch % 5 == 0:
            print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>9.3f} | {te_acc:>8.3f}")

    # save model
    model_path = f"{model_name}_{SCENARIO}_s{STRATEGY}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Modell gespeichert: {model_path}")

    # plotting the learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_acc"], label="Train Accuracy", color="steelblue")
    plt.plot(history["test_acc"],  label="Test Accuracy",  color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name.upper()} — {SCENARIO} classification (Strategy {STRATEGY})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f"plot_training_{model_name}_{SCENARIO}_s{STRATEGY}.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Training-Plot gespeichert: {plot_path}")

    # plot of confusion matrix
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            pred = model(X_batch.to(device)).argmax(1).cpu()
            all_pred.extend(pred.numpy())
            all_true.extend(y_batch.numpy())

    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset.classes,
                yticklabels=dataset.classes)
    plt.title(f"Confusion Matrix — {model_name.upper()} ({SCENARIO})")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = f"plot_cm_{model_name}_{SCENARIO}_s{STRATEGY}.png"
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"Confusion Matrix gespeichert: {cm_path}")

    # Classification report
    report = classification_report(all_true, all_pred,
                                   target_names=dataset.classes)
    print(f"\nClassification Report:\n{report}")

    report_path = f"report_{model_name}_{SCENARIO}_s{STRATEGY}.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {model_name.upper()}\n")
        f.write(f"Scenario: {SCENARIO}\n")
        f.write(f"Strategy: {STRATEGY}\n")
        f.write(f"Frame size: {FRAME_SIZE}, Overlap: {OVERLAP}\n")
        f.write(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}\n")
        f.write(f"Best Test Accuracy:  {max(history['test_acc']):.4f}\n\n")
        f.write(report)
    print(f"Report gespeichert: {report_path}")

    return max(history["test_acc"])


if __name__ == "__main__":
    cnn_acc = run_with_export("cnn")
    resnet_acc = run_with_export("resnet")
    print(f"\n{'='*45}")
    print(f"  CNN    beste Accuracy: {cnn_acc:.4f}")
    print(f"  ResNet beste Accuracy: {resnet_acc:.4f}")
    print(f"{'='*45}")
