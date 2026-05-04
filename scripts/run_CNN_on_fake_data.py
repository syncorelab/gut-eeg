"""
Structure:
1. Imports
2. Helper functions
3. Main pipeline
4. Run main
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch

# --------------------------------------------------
# CHANGE THESE IMPORTS TO YOUR CNN-ONLY FILES
# --------------------------------------------------
from configs.CNN_Hparameters_distributions import HPARAM_DISTS
from src.models.cnn_classifier_model import GUTNet
# --------------------------------------------------

from src.dataloading.splits import load_eeg_and_triggers, make_stratified_group_folds
from src.dataloading.dataloader import make_dataloaders

# You can keep this IF the train loop is model-agnostic and just does:
# outputs = model(inputs)
from src.training.train_loop import train_GUTNet

from src.evaluation.evaluation import (
    plot_confusion_matrices,
    plot_train_test_accuracy,
    plot_train_test_loss,
)
from src.utils.randomsearch import sample_hparams, is_valid_config


# -------------------------
# Helper functions
# -------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_experiment_dir(root: str = "results") -> Path:
    """
    Creates:
        results/exp_YYYYMMDD_HHMMSS/

    If folder already exists, appends counter.
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
    Creates:
        results/exp_xxx/trial_000/
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
    """
    Safe JSON saving.
    Creates parent directories automatically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(obj, f, indent=2)

    print(f"[INFO] Saved JSON to: {path}")


def start_timer() -> float:
    return time.perf_counter()


def end_timer(start_time: float, trial_dir: Path, label: str = "train_test") -> float:
    """
    Stops timer, prints duration, and saves it to trial_dir.
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


# -------------------------
# Main pipeline
# -------------------------
def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    N_TRIALS = 1
    MAX_RESAMPLE_ATTEMPTS = 200

    exp_dir = create_experiment_dir(root="results")
    print(f"Saving results to: {exp_dir}")

    save_json(HPARAM_DISTS, exp_dir / "hparam_distributions.json")

    holdout_subject = None

    # -------------------------
    # Random search loop
    # -------------------------
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

        # -------------------------
        # Load data
        # -------------------------
        concat_files = [
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/concat_s1.mat",
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/concat_s2.mat",
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/concat_s4.mat",
        ]
        origin_files = [
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/origin_s1.mat",
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/origin_s2.mat",
            "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/origin_s4.mat",
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

        # -------------------------
        # Make folds
        # -------------------------
        folds, holdout_subject = make_stratified_group_folds(
            X=X,
            y=y,
            groups=subjects,
            n_splits=5,
            random_state=3,
        )

        # -------------------------
        # Make dataloaders
        # -------------------------
        k = 0
        batch_size = DATA_HPARAMS["batch_size"]

        train_loader, test_loader = make_dataloaders(
            folds=folds,
            k=k,
            batch_size=batch_size,
        )

        # -------------------------
        # Initialize CNN model
        # -------------------------
        model = GUTNet(**MODEL_HPARAMS)
        model.to(device)

        # -------------------------
        # Train model
        # -------------------------
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

        # -------------------------
        # Evaluate + save plots
        # -------------------------
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

        save_json(
            {"trial": trial, "validity_reason": reason},
            trial_dir / "trial_meta.json",
        )

        end_timer(start_time, trial_dir, label="model_run")

    if holdout_subject is not None:
        print(f'holdout subject is {holdout_subject["subject_ids"]}, random state is 3')

    print("\nDone. Random search finished.")


if __name__ == "__main__":
    main()

print("Ran smuuuuudly.. first try ;)")