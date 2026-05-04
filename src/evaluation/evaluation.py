"""
Evaluation and plotting utilities for training and testing classification models.

This file includes:
1. Basic accuracy and evaluation helpers
2. Confusion matrix and per-class metric functions
3. Evaluation on a new dataset or subject
4. Plotting functions for losses, accuracy, and confusion matrices
"""

from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


def safe_save_path(base_dir: Path, filename: str) -> Path:
    """
    Make sure the output directory exists and return a safe path for saving
    a file inside it.
    """
    base_dir = Path(base_dir)

    # If a file path is passed instead of a directory, use its parent folder.
    if base_dir.suffix != "":
        print("[WARNING] save_path was a file. Using its parent directory instead.")
        base_dir = base_dir.parent

    base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir / filename


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy for one batch of model outputs.
    """
    preds = outputs.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def compute_confusion_matrix(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    normalize: bool = False,
) -> np.ndarray:
    """
    Compute a confusion matrix over all samples in a dataloader.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    cm = confusion_matrix(
        all_targets,
        all_preds,
        normalize="true" if normalize else None,
    )
    return cm


def compute_per_class_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    target_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute per-class precision, recall, F1 score, and support for all
    samples in a dataloader.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    report = classification_report(
        all_targets,
        all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    return report


def evaluate_on_new_subject(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    Evaluate a trained model on a dataloader and return average loss and
    accuracy.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    n_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            running_correct += (preds == targets).sum().item()
            n_samples += batch_size

    avg_loss = running_loss / n_samples
    accuracy = running_correct / n_samples
    return avg_loss, accuracy


def plot_losses_errors(
    train_losses: List[float],
    test_losses: List[float],
    train_errors: List[float],
    test_errors: List[float],
    error_is_fraction: bool = True,
) -> None:
    """
    Plot train and test loss curves together with train and test error curves.
    """
    train_losses = np.asarray(train_losses)
    test_losses = np.asarray(test_losses)
    train_errors = np.asarray(train_errors)
    test_errors = np.asarray(test_errors)

    if error_is_fraction:
        train_err_plot = train_errors * 100.0
        test_err_plot = test_errors * 100.0
    else:
        train_err_plot = train_errors
        test_err_plot = test_errors

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    # Plot loss across epochs.
    ax[0].plot(train_losses, 's-', label='Train')
    ax[0].plot(test_losses, 'o-', label='Test')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Model loss')
    ax[0].legend()

    # Plot error rate across epochs.
    ax[1].plot(train_err_plot, 's-', label='Train')
    ax[1].plot(test_err_plot, 'o-', label='Test')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Error rate (%)')
    ax[1].set_title(f'Final model test error rate: {test_err_plot[-1]:.2f}%')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(
    train_targets,
    train_preds,
    test_targets,
    test_preds,
    class_names,
    save_dir: str | Path | None = None,
    show: bool = False,
):
    """
    Plot and optionally save confusion matrices for the training set and test set.

    For training, only the last 100 predictions are shown to keep the plot
    focused on recent training performance.
    """
    train_targets = np.asarray(train_targets)
    train_preds = np.asarray(train_preds)
    test_targets = np.asarray(test_targets)
    test_preds = np.asarray(test_preds)

    train_targets_last = train_targets[-100:]
    train_preds_last = train_preds[-100:]

    labels = np.arange(len(class_names))
    cm_train = confusion_matrix(train_targets_last, train_preds_last, labels=labels)
    cm_test = confusion_matrix(test_targets, test_preds, labels=labels)

    save_dir = Path(save_dir) if save_dir is not None else None

    def _plot(cm, title, filename: str):
        """Internal helper for drawing and saving one confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6), num=title)

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predictions")
        ax.set_ylabel("Actual")

        thresh = cm.max() / 2.0 if cm.size else 0.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, f"{cm[i, j]:d}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        fig.tight_layout()

        if save_dir is not None:
            full_path = safe_save_path(save_dir, filename)
            fig.savefig(full_path, dpi=300, bbox_inches="tight")
            print(f"Saved confusion matrix to {full_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    _plot(cm_train, "Confusion Matrix (Train - Last 100)", "confusion_matrix_train.png")
    _plot(cm_test, "Confusion Matrix (Test)", "confusion_matrix_test.png")


def plot_train_test_accuracy(
    train_accuracy,
    test_accuracy,
    title="Accuracy over epochs",
    save_dir: str | Path | None = None,
    filename: str = "accuracy.png",
    show: bool = True,
):
    """
    Plot train and test accuracy across epochs and mark the best epoch for each.
    """
    train_accuracy = np.asarray(train_accuracy, dtype=float)
    test_accuracy = np.asarray(test_accuracy, dtype=float)

    n_epochs = min(len(train_accuracy), len(test_accuracy))
    train_accuracy = train_accuracy[:n_epochs]
    test_accuracy = test_accuracy[:n_epochs]
    epochs = np.arange(1, n_epochs + 1)

    train_best_idx = int(np.argmax(train_accuracy))
    test_best_idx = int(np.argmax(test_accuracy))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, marker="o", label="Train")
    plt.plot(epochs, test_accuracy, marker="o", label="Test")

    plt.scatter(epochs[train_best_idx], train_accuracy[train_best_idx], c="red", zorder=5)
    plt.text(
        epochs[train_best_idx],
        train_accuracy[train_best_idx],
        f"  epoch {epochs[train_best_idx]}",
        color="red",
        fontsize=10,
        va="center",
    )

    plt.scatter(epochs[test_best_idx], test_accuracy[test_best_idx], c="red", zorder=5)
    plt.text(
        epochs[test_best_idx],
        test_accuracy[test_best_idx],
        f"  epoch {epochs[test_best_idx]}",
        color="red",
        fontsize=10,
        va="center",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        full_path = safe_save_path(save_dir, filename)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved accuracy plot to {full_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_train_test_loss(
    train_losses,
    test_losses,
    title="Loss over epochs",
    save_dir: str | Path | None = None,
    filename: str = "loss.png",
    show: bool = True,
):
    """
    Plot train and test loss across epochs and mark the lowest-loss epoch for each.
    """
    train_losses = np.asarray(train_losses, dtype=float)
    test_losses = np.asarray(test_losses, dtype=float)

    n_epochs = min(len(train_losses), len(test_losses))
    train_losses = train_losses[:n_epochs]
    test_losses = test_losses[:n_epochs]
    epochs = np.arange(1, n_epochs + 1)

    train_best_idx = int(np.argmin(train_losses))
    test_best_idx = int(np.argmin(test_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, marker="o", label="Train")
    plt.plot(epochs, test_losses, marker="o", label="Test")

    plt.scatter(epochs[train_best_idx], train_losses[train_best_idx], c="red", zorder=5)
    plt.text(
        epochs[train_best_idx],
        train_losses[train_best_idx],
        f"  epoch {epochs[train_best_idx]}",
        color="red",
        fontsize=10,
        va="center",
    )

    plt.scatter(epochs[test_best_idx], test_losses[test_best_idx], c="red", zorder=5)
    plt.text(
        epochs[test_best_idx],
        test_losses[test_best_idx],
        f"  epoch {epochs[test_best_idx]}",
        color="red",
        fontsize=10,
        va="center",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        full_path = safe_save_path(save_dir, filename)
        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        print(f"Saved loss plot to {full_path}")

    if show:
        plt.show()
    else:
        plt.close()