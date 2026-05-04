"""
Run random hyperparameter search for the CNN classifier on the 30 Hz preprocessed tensor data, then evaluate the best trial on a separate
holdout subject set.

Structure:
1. Imports
2. Helper functions
3. Main pipeline
4. Script entry point
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from configs.CNN_Hparameters_distributions import HPARAM_DISTS
from src.models.cnn_classifier_model import GUTNet

from src.dataloading.splits import load_eeg_and_triggers, make_stratified_group_folds
from src.dataloading.dataloader import make_dataloaders
from src.training.train_loop import train_GUTNet

from src.evaluation.evaluation import (
    plot_confusion_matrices,
    plot_train_test_accuracy,
    plot_train_test_loss,
)
from src.utils.randomsearch import sample_hparams, is_valid_config


def get_device() -> torch.device:
    """Use GPU if available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_experiment_dir(root: str = "results") -> Path:
    """
    Create a new experiment folder for one full random-search run.

    Expected format:
        results/exp_YYYYMMDD_HHMMSS/

    If that name already exists, a counter is added.
    """
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = root_path / f"exp_{timestamp}"

    counter = 1
    while exp_dir.exists():
        exp_dir = root_path / f"exp_{timestamp}_{counter}"
        counter += 1

    exp_dir.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] Created experiment directory: {exp_dir}")
    return exp_dir


def create_trial_dir(exp_dir: Path, trial_idx: int) -> Path:
    """
    Create a folder for one sampled hyperparameter trial inside the
    experiment directory.
    """
    exp_dir = Path(exp_dir)

    if not exp_dir.exists():
        print("[WARNING] Experiment directory missing. Creating new experiment folder.")
        exp_dir = create_experiment_dir(exp_dir.parent)

    trial_dir = exp_dir / f"trial_{trial_idx:03d}"

    counter = 1
    while trial_dir.exists():
        trial_dir = exp_dir / f"trial_{trial_idx:03d}_{counter}"
        counter += 1

    trial_dir.mkdir(parents=True, exist_ok=False)

    print(f"[INFO] Created trial directory: {trial_dir}")
    return trial_dir


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """Save a dictionary as JSON and create parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(obj, f, indent=2)

    print(f"[INFO] Saved JSON to: {path}")


def start_timer() -> float:
    """Start a high-resolution timer."""
    return time.perf_counter()


def end_timer(start_time: float, trial_dir: Path, label: str = "train_test") -> float:
    """
    Stop a timer, print the duration, and save it to a text file inside the
    trial folder.
    """
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    end_time = time.perf_counter()
    duration_sec = end_time - start_time
    duration_min = duration_sec / 60

    print(f"\n[⏱] {label} took {duration_sec:.2f} seconds ({duration_min:.2f} minutes)\n")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_path = trial_dir / f"{label}_timing.txt"

    with save_path.open("w") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Duration (seconds): {duration_sec:.4f}\n")
        f.write(f"Duration (minutes): {duration_min:.4f}\n")

    return duration_sec


def make_holdout_loader(holdout_subject: Dict[str, Any], batch_size: int):
    """Build a dataloader for the final holdout evaluation."""
    X_holdout = torch.tensor(holdout_subject["data"], dtype=torch.float32)
    y_holdout = torch.tensor(holdout_subject["labels"], dtype=torch.long)

    dataset = TensorDataset(X_holdout, y_holdout)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return loader


def evaluate_no_backprop(
    model: nn.Module,
    data_loader,
    device: torch.device,
):
    """
    Evaluate a trained model without gradient updates and return loss,
    accuracy, targets, and predictions.
    """
    criterion = nn.CrossEntropyLoss().to(device)

    model.eval()
    running_loss = 0.0
    running_errors = 0
    n_samples = 0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size

            preds = outputs.argmax(dim=1)
            running_errors += (preds != targets).sum().item()
            n_samples += batch_size

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())

    avg_loss = running_loss / n_samples
    avg_accuracy = 1.0 - (running_errors / n_samples)

    preds_flat = np.concatenate(preds_all, axis=0)
    targets_flat = np.concatenate(targets_all, axis=0)

    return avg_loss, avg_accuracy, targets_flat, preds_flat


def main() -> None:
    """Run random search training and evaluate the best trial on the holdout set."""
    device = get_device()
    print(f"Using device: {device}")

    N_TRIALS = 10
    MAX_RESAMPLE_ATTEMPTS = 200

    exp_dir = create_experiment_dir(root="results")
    print(f"Saving results to: {exp_dir}")

    save_json(HPARAM_DISTS, exp_dir / "hparam_distributions.json")

    holdout_subject = None
    best_trial_info = None

    # Repeatedly sample hyperparameters, train a model, and store results
    # for each valid trial.
    for trial in range(N_TRIALS):
        sampled_cfg = None
        reason = "not_sampled"

        for attempt in range(MAX_RESAMPLE_ATTEMPTS):
            cfg = sample_hparams(HPARAM_DISTS)
            ok, reason = is_valid_config(cfg)
            if ok:
                sampled_cfg = cfg
                break

        if sampled_cfg is None:
            print(
                f"[trial {trial}] FAILED to find valid config after "
                f"{MAX_RESAMPLE_ATTEMPTS} attempts. Last reason: {reason}"
            )
            continue

        trial_dir = create_trial_dir(exp_dir, trial)
        save_json(sampled_cfg, trial_dir / "hparams_sampled.json")

        DATA_HPARAMS = sampled_cfg["DATA_HPARAMS"]
        MODEL_HPARAMS = sampled_cfg["MODEL_HPARAMS"]
        TRAINING_HPARAMS = sampled_cfg["TRAINING_HPARAMS"]

        print(f"\n=== Trial {trial} ===")
        print(f"DATA_HPARAMS: {DATA_HPARAMS}")
        print(f"MODEL_HPARAMS: {MODEL_HPARAMS}")
        print(f"TRAINING_HPARAMS: {TRAINING_HPARAMS}")

        start_time = start_timer()

        # Load the three condition-specific concatenated tensors and their
        # matching subject-origin files.
        concat_files = [
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/concatenated_withICA_v2/concat_s1.mat",
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/concatenated_withICA_v2/concat_s2.mat",
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/concatenated_withICA_v2/concat_s4.mat",
        ]
        origin_files = [
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/concatenated_withICA_v2/origin_s1.mat",
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/concatenated_withICA_v2/origin_s2.mat",
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/concatenated_withICA_v2/origin_s4.mat",
        ]

        print("Loading data...")

        X, y, subjects = load_eeg_and_triggers(
            concat_files=concat_files,
            origin_files=origin_files,
            cond_nums=(1, 2, 4),
        )

        print(f"X shape: {X.shape}, y shape: {y.shape}, subjects shape: {subjects.shape}")
        print("phh it worked.. that is a miracle!")

        n_channels = X.shape[1]
        n_time = X.shape[2]
        n_classes = int(len(np.unique(y)))
        print(f"Detected: {n_channels} channels, {n_time} timepoints, {n_classes} classes")

        # Create subject-grouped folds and reserve a separate holdout set.
        folds, holdout_subject = make_stratified_group_folds(
            X=X,
            y=y,
            groups=subjects,
            n_splits=5,
            random_state=3,
        )

        # Use one fold split to build train and test loaders for this trial.
        k = 0
        batch_size = DATA_HPARAMS["batch_size"]

        train_loader, test_loader = make_dataloaders(
            folds=folds,
            k=k,
            batch_size=batch_size,
        )

        # Initialize the classifier using the sampled model hyperparameters.
        model = GUTNet(**MODEL_HPARAMS)
        model.to(device)

        # Train the model and collect the metrics needed for evaluation plots.
        (
            model,
            train_losses,
            train_accuracy,
            test_losses,
            test_accuracy,
            last_epoch_train_targets_last100,
            last_epoch_train_preds_last100,
            test_targets_flat,
            test_preds_flat,
        ) = train_GUTNet(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            save_dir=trial_dir,
            TRAINING_HPARAMS=TRAINING_HPARAMS,
        )

        # Save confusion matrices and learning-curve plots for this trial.
        class_names = ["fear", "hunger", "nature"]

        plot_confusion_matrices(
            train_targets=last_epoch_train_targets_last100,
            train_preds=last_epoch_train_preds_last100,
            test_targets=test_targets_flat,
            test_preds=test_preds_flat,
            class_names=class_names,
            save_dir=trial_dir,
            show=False,
        )

        plot_train_test_accuracy(
            train_accuracy=train_accuracy,
            test_accuracy=test_accuracy,
            title="Train/Test Accuracy",
            save_dir=trial_dir,
            filename="accuracy.png",
        )

        plot_train_test_loss(
            train_losses=train_losses,
            test_losses=test_losses,
            title="Train/Test Loss",
            save_dir=trial_dir,
            filename="loss.png",
        )

        # Track the trial with the highest test accuracy so it can be
        # reloaded for final holdout evaluation.
        peak_test_acc = float(max(test_accuracy))
        peak_test_epoch = int(np.argmax(test_accuracy))
        loss_at_peak_test_acc = float(test_losses[peak_test_epoch])

        if (best_trial_info is None) or (peak_test_acc > best_trial_info["peak_test_accuracy"]):
            best_trial_info = {
                "trial": trial,
                "trial_dir": str(trial_dir),
                "peak_test_accuracy": peak_test_acc,
                "peak_test_epoch": peak_test_epoch,
                "loss_at_peak_test_accuracy": loss_at_peak_test_acc,
                "model_hparams": MODEL_HPARAMS,
                "batch_size": batch_size,
            }

        save_json(
            {
                "trial": trial,
                "validity_reason": reason,
                "peak_test_accuracy": peak_test_acc,
                "peak_test_epoch": peak_test_epoch,
                "loss_at_peak_test_accuracy": loss_at_peak_test_acc,
            },
            trial_dir / "trial_meta.json",
        )

        end_timer(start_time, trial_dir, label="model_run")

    # After random search, reload the best trial and test it once on the
    # untouched holdout subject set.
    if holdout_subject is not None:
        print(f'holdout subjects are {holdout_subject["subject_ids"]}, random state is 3')

    if best_trial_info is None:
        print("No valid trials completed, so no holdout evaluation could be run.")
        print("\nDone. Random search finished.")
        return

    print("\nBest trial selected from random search:")
    print(best_trial_info)

    best_trial_dir = Path(best_trial_info["trial_dir"])
    weights_path = best_trial_dir / "trained_model_state_dict.pt"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Could not find saved weights for best trial at: {weights_path}"
        )

    holdout_eval_dir = exp_dir / "best_trial_holdout_eval"
    holdout_eval_dir.mkdir(parents=True, exist_ok=True)

    holdout_start_time = start_timer()

    best_model = GUTNet(**best_trial_info["model_hparams"])
    best_model.to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        best_model.load_state_dict(checkpoint["model_state_dict"])
    else:
        best_model.load_state_dict(checkpoint)

    holdout_loader = make_holdout_loader(
        holdout_subject=holdout_subject,
        batch_size=best_trial_info["batch_size"],
    )

    holdout_loss, holdout_accuracy, holdout_targets_flat, holdout_preds_flat = evaluate_no_backprop(
        model=best_model,
        data_loader=holdout_loader,
        device=device,
    )

    holdout_macro_f1 = float(f1_score(holdout_targets_flat, holdout_preds_flat, average="macro"))

    print(f"Best-trial holdout loss: {holdout_loss:.6f}")
    print(f"Best-trial holdout accuracy: {holdout_accuracy:.6f}")
    print(f"Best-trial holdout macro F1: {holdout_macro_f1:.6f}")

    save_json(
        {
            "best_trial": best_trial_info["trial"],
            "best_trial_dir": str(best_trial_dir),
            "weights_path": str(weights_path),
            "holdout_subject_ids": np.asarray(holdout_subject["subject_ids"]).tolist(),
            "holdout_loss": float(holdout_loss),
            "holdout_accuracy": float(holdout_accuracy),
            "holdout_macro_f1": float(holdout_macro_f1),
            "selection_metric": "max(test_accuracy)",
            "best_trial_peak_test_accuracy": float(best_trial_info["peak_test_accuracy"]),
            "best_trial_peak_test_epoch": int(best_trial_info["peak_test_epoch"]),
            "best_trial_loss_at_peak_test_accuracy": float(best_trial_info["loss_at_peak_test_accuracy"]),
        },
        holdout_eval_dir / "holdout_eval_summary.json"
    )

    np.save(holdout_eval_dir / "holdout_targets.npy", holdout_targets_flat)
    np.save(holdout_eval_dir / "holdout_preds.npy", holdout_preds_flat)

    # Reuse the plotting helpers to save single-point holdout summaries.
    plot_train_test_accuracy(
        train_accuracy=[holdout_accuracy],
        test_accuracy=[holdout_accuracy],
        title="Holdout Accuracy",
        save_dir=holdout_eval_dir,
        filename="holdout_accuracy.png",
    )

    plot_train_test_loss(
        train_losses=[holdout_loss],
        test_losses=[holdout_loss],
        title="Holdout Loss",
        save_dir=holdout_eval_dir,
        filename="holdout_loss.png",
    )

    end_timer(holdout_start_time, holdout_eval_dir, label="holdout_eval")

    print("\nDone. Random search finished.")


if __name__ == "__main__":
    main()

print("Ran smuuuuudly.. first try :)")