"""
Train a GUTNet classification model and track training and test performance
across epochs.

This function:
1. Builds the loss function and optimizer
2. Optionally enables early stopping based on test loss
3. Trains the model epoch by epoch
4. Stores losses, accuracies, and predictions for later evaluation
5. Restores the best model if early stopping is used
6. Saves the trained model weights if a save directory is provided
"""

from typing import Dict, Tuple, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.training.early_stopping import EarlyStopping


def train_GUTNet(
    model: nn.Module,
    train_loader,
    test_loader,
    TRAINING_HPARAMS: Dict,
    device: torch.device,
    save_dir: Path = None,
):
    # Read training settings from the hyperparameter dictionary.
    learning_rate = TRAINING_HPARAMS["learning_rate"]
    num_epochs = TRAINING_HPARAMS.get("num_epochs", 150)
    betas = TRAINING_HPARAMS.get("betas", (0.9, 0.999))
    weight_decay = TRAINING_HPARAMS.get("weight_decay", 0.0)

    # Early stopping is optional and monitors test loss.
    use_early_stopping = TRAINING_HPARAMS.get("use_early_stopping", False)
    early_stopping_patience = TRAINING_HPARAMS.get("early_stopping_patience", 20)
    early_stopping_min_delta = TRAINING_HPARAMS.get("early_stopping_min_delta", 0.001)

    # Cross-entropy loss is used for multi-class classification.
    criterion = nn.CrossEntropyLoss().to(device)

    # Adam optimizer updates the model weights during training.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )

    # Create the early stopping helper only if it is enabled.
    early_stopper = None
    if use_early_stopping:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
            restore_best_weights=True,
        )

    # Store loss and accuracy values from each epoch.
    train_losses: List[float] = []
    train_accuracy: List[float] = []
    test_losses: List[float] = []
    test_accuracy: List[float] = []

    # Store final test predictions and targets for later evaluation plots.
    test_preds_all = []
    test_targets_all = []

    # Store training predictions and targets from the last fully completed epoch.
    last_epoch_train_preds_all = []
    last_epoch_train_targets_all = []

    for epoch in range(num_epochs):
        # Switch the model to training mode.
        model.train()
        running_loss = 0.0
        running_errors = 0
        n_train_samples = 0

        # Collect predictions from this training epoch.
        epoch_train_preds_all = []
        epoch_train_targets_all = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            running_errors += (preds != targets).sum().item()
            n_train_samples += batch_size

            epoch_train_preds_all.append(preds.detach().cpu().numpy())
            epoch_train_targets_all.append(targets.detach().cpu().numpy())

        avg_train_loss = running_loss / n_train_samples
        avg_train_accuracy = 1.0 - (running_errors / n_train_samples)

        train_losses.append(avg_train_loss)
        train_accuracy.append(avg_train_accuracy)

        # Keep the most recent full epoch of training predictions.
        last_epoch_train_preds_all = epoch_train_preds_all.copy()
        last_epoch_train_targets_all = epoch_train_targets_all.copy()

        # Switch the model to evaluation mode for the test set.
        model.eval()
        test_running_loss = 0.0
        test_running_errors = 0
        n_test_samples = 0

        # Reset test predictions for this epoch.
        test_preds_all = []
        test_targets_all = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_size = inputs.size(0)
                test_running_loss += loss.item() * batch_size

                preds = outputs.argmax(dim=1)
                test_running_errors += (preds != targets).sum().item()
                n_test_samples += batch_size

                test_preds_all.append(preds.detach().cpu().numpy())
                test_targets_all.append(targets.detach().cpu().numpy())

        avg_test_loss = test_running_loss / n_test_samples
        avg_test_accuracy = 1.0 - (test_running_errors / n_test_samples)

        test_losses.append(avg_test_loss)
        test_accuracy.append(avg_test_accuracy)

        # Let early stopping inspect the current test loss.
        if use_early_stopping:
            early_stopper.step(avg_test_loss, model, epoch)

            if early_stopper.should_stop:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    # Restore the best saved weights if early stopping was active.
    if use_early_stopping:
        early_stopper.restore(model)

        if early_stopper.best_epoch is not None and early_stopper.best_score is not None:
            print(f"Best epoch: {early_stopper.best_epoch + 1}")
            print(f"Best validation loss: {early_stopper.best_score:.6f}")
    else:
        print("Early stopping is OFF. Using final epoch weights.")

    # Run one final evaluation using the restored best model or final epoch model.
    model.eval()
    test_running_loss = 0.0
    test_running_errors = 0
    n_test_samples = 0

    test_preds_all = []
    test_targets_all = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            test_running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            test_running_errors += (preds != targets).sum().item()
            n_test_samples += batch_size

            test_preds_all.append(preds.detach().cpu().numpy())
            test_targets_all.append(targets.detach().cpu().numpy())

    test_preds_flat = np.concatenate(test_preds_all, axis=0)
    test_targets_flat = np.concatenate(test_targets_all, axis=0)

    # Flatten the stored training predictions from the last full epoch.
    last_epoch_train_preds_flat = np.concatenate(last_epoch_train_preds_all, axis=0)
    last_epoch_train_targets_flat = np.concatenate(last_epoch_train_targets_all, axis=0)

    # Keep only the last 100 training predictions for the training confusion matrix.
    last_epoch_train_preds_last100 = last_epoch_train_preds_flat[-100:]
    last_epoch_train_targets_last100 = last_epoch_train_targets_flat[-100:]

    # Save the trained model weights if an output folder is provided.
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        weights_path = save_dir / "trained_model_state_dict.pt"
        torch.save({"model_state_dict": model.state_dict()}, weights_path)
        print(f"Saved trained model weights to: {weights_path}")

    return (
        model,
        train_losses,
        train_accuracy,
        test_losses,
        test_accuracy,
        last_epoch_train_targets_last100,
        last_epoch_train_preds_last100,
        test_targets_flat,
        test_preds_flat,
    )