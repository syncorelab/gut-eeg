"""
Create PyTorch dataloaders for one fold of cross-validation.

The selected fold is used as the test set, while all remaining folds are
combined into one training set.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_dataloaders(folds: dict, k: int, batch_size: int):
    """
    Build training and test dataloaders from a fold dictionary.

    The chosen fold k is used as the test set, and all other folds are
    concatenated into the training set.
    """
    fold_key = f"fold_{k}"
    assert fold_key in folds, f"{fold_key} not found in folds."

    # Use the selected fold as the test set.
    test_data = folds[fold_key]["data"]
    test_labels = folds[fold_key]["labels"]

    # Combine all remaining folds into one training set.
    train_data_list = []
    train_label_list = []

    for this_key, fold in folds.items():
        if this_key == fold_key:
            continue
        train_data_list.append(fold["data"])
        train_label_list.append(fold["labels"])

    train_data = np.concatenate(train_data_list, axis=0)
    train_labels = np.concatenate(train_label_list, axis=0)

    # Convert numpy arrays into PyTorch tensors.
    train_X = torch.tensor(train_data, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.long)

    test_X = torch.tensor(test_data, dtype=torch.float32)
    test_y = torch.tensor(test_labels, dtype=torch.long)

    # Wrap tensors in TensorDataset so they can be used by DataLoader.
    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    # Shuffle the training data each epoch. The test loader uses the full
    # test set in one batch and keeps the order fixed.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_dataset.tensors[0].shape[0],
        shuffle=False,
    )

    return train_loader, test_loader