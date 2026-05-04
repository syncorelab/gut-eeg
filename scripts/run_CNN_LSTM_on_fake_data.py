"""

this script is only meant to test if the model runs without errors, not to acctually train a good model or get any meaningfull results.

the structure is folder based and the functions and classes used are listed in the import section.

script structure:
1. Imports
2. helper functions
    2.1 get_device
    2.2 create_experiment_dir
    2.3 save_json
3. main pipeline
    3.1 set up device, experiment folder, save configs, random search loop (sample config, check validity, save config)
    3.2 load data 
    3.3 make folds(I will skip stratified group k-fold CV for now and just use the first fold, mostly because it will take too long when this is just a test anyway)
    3.4 make dataloaders
    3.5 initialize model instance
    3.6 train model
    3.7 evaluate model
4. run main"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch

from configs.CNN_LSTM_Hparameters_distributions import HPARAM_DISTS
from src.dataloading.splits import load_eeg_and_triggers, make_stratified_group_folds
from src.dataloading.dataloader import make_dataloaders
from src.models.cnn_lstm_model import GUTNet
from src.training.train_loop_for_lstm import train_GUTNet
from src.evaluation.evaluation import (
    plot_confusion_matrices,
    plot_train_test_accuracy,
    plot_train_test_loss,
    safe_save_path,
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

    If folder already exists (very rare), appends counter.
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
    """
    Starts a high-precision timer.
    Returns
    -------
    float : start time reference
    """
    return time.perf_counter()


def end_timer(start_time: float, trial_dir: Path, label: str = "train_test") -> float:
    """
    Stops timer, prints duration, and saves it to trial_dir.

    Parameters
    ----------
    start_time : float
        Value returned from start_timer()
    trial_dir : Path
        Directory where timing info should be saved
    label : str
        Name prefix for saved file

    Returns
    -------
    float : duration in seconds
    """
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)

    end_time = time.perf_counter()
    duration_sec = end_time - start_time

    duration_min = duration_sec / 60

    # ---- Pretty print ----
    print(f"\n[⏱] {label} took {duration_sec:.2f} seconds ({duration_min:.2f} minutes)\n")

    # ---- Save to file ----
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

    # How many random configs to try:
    N_TRIALS = 1

    # Avoid infinite loops if constraints are tight:
    MAX_RESAMPLE_ATTEMPTS = 200

    exp_dir = create_experiment_dir(root="results")
    print(f"Saving results to: {exp_dir}")

    # Save the distributions used (so you know what space you searched)
    save_json(HPARAM_DISTS, exp_dir / "hparam_distributions.json")

    # -------------------------
    # Random search loop
    # -------------------------
    for trial in range(N_TRIALS):
        # --- sample until valid ---
        sampled_cfg = None
        reason = "not_sampled"

        for attempt in range(MAX_RESAMPLE_ATTEMPTS):
            cfg = sample_hparams(HPARAM_DISTS)
            ok, reason = is_valid_config(cfg)
            if ok:
                sampled_cfg = cfg
                break

        if sampled_cfg is None:
            print(f"[trial {trial}] FAILED to find valid config after {MAX_RESAMPLE_ATTEMPTS} attempts. Last reason: {reason}")
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

        # ---- START TIMER ----
        start_time = start_timer()

        # -------------------------
        # 3.2 Load data
        # -------------------------
        concat_files = [
        "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/concat_s1.mat",
        "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/concat_s2.mat",
        "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/concat_s4.mat"
        ]
        origin_files = [
        "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/origin_s1.mat",
        "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/origin_s2.mat",
        "/home/gutproject/Desktop/guteeg/gut-eeg/data/fake_windowed/origin_s4.mat"
        ]    

        print("Loading data...")

        X, y, subjects = load_eeg_and_triggers(
            concat_files=concat_files,
            origin_files=origin_files,
            cond_nums=(1, 2, 4),
            #window_size=DATA_HPARAMS["window_size"],
            #overlap=DATA_HPARAMS["overlap"],
        )

        print(f"X shape: {X.shape}, y shape: {y.shape}, subjects shape: {subjects.shape}")
        print("phh it worked.. that is a miracle!")
        n_channels = X.shape[1]
        n_time     = X.shape[2]
        n_classes  = int(len(np.unique(y)))
        print(f"Detected: {n_channels} channels, {n_time} timepoints, {n_classes} classes")

        # 3.3 Make folds (you said: just use the first fold)
        all_results = []

        # Make folds (by subject, stratified on y)
        folds, holdout_subject = make_stratified_group_folds(
            X=X,
            y=y,
            groups=subjects,       # or 'subjects' if your arg is named that
            n_splits= 5,
            random_state=3,
            )
        # 3.4 Make dataloaders
        k = 0                      #REMEMBER: just use the first fold for testing purposes, not actual CV
        batch_size = DATA_HPARAMS["batch_size"]
        train_loader, test_loader = make_dataloaders(
            folds=folds,
            k=k,                # fold index
            batch_size=batch_size,
        )

        # 3.5 Initialize model instance
        model = GUTNet(**MODEL_HPARAMS)
        model.to(device)

        # 3.6 Train model
        model, train_losses,train_accuracy,test_losses,test_accuracy, last_epoch_train_targets_last100,last_epoch_train_preds_last100,test_targets_flat,test_preds_flat = train_GUTNet(
             model=model,
             train_loader=train_loader,
             test_loader=test_loader,
             device=device,
             save_dir=trial_dir,
             TRAINING_HPARAMS=TRAINING_HPARAMS
             )

        # 3.7 Evaluate model + plots saved into trial_dir
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
            save_dir= trial_dir,
            filename="accuracy.png"
        )

        plot_train_test_loss(
            train_losses=train_losses,
            test_losses=test_losses,
            title="Train/Test Loss",
            save_dir= trial_dir,
            filename="loss.png"
        )

        # Optional: also save a quick “status” file per trial
        save_json(
            {"trial": trial, "validity_reason": reason},
            trial_dir / "trial_meta.json"
        )

        # ---- STOP TIMER ----
        duration = end_timer(start_time, trial_dir, label="model_run")

    print(f"holdout subject is {holdout_subject["subject_ids"]} random state is 3")
    print("\nDone. Random search finished.")


if __name__ == "__main__":
    main()

print("Ran smuuuuudly.. first try ;)")