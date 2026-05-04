"""
Load concatenated EEG tensors and subject-origin files, then create
subject-grouped stratified folds with a separate holdout subject set.

This file:
1. Loads MATLAB .mat files, including v7.3 files when needed
2. Matches concat and origin files by condition
3. Builds X, y, and subject arrays for model training
4. Creates stratified group folds after removing holdout subjects
"""

import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold


def load_mat_smart(path: str):
    """
    Load a MATLAB file using scipy, and fall back to h5py for v7.3 files.
    """
    try:
        return loadmat(path)
    except NotImplementedError as e:
        if "v7.3" not in str(e).lower():
            raise
        import h5py
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.array(f[k])
        return out


def find_file_for_condition(files, prefix, cond):
    """
    Find a condition-specific file such as concat_s1.mat or origin_s1.mat
    from a list of file paths.
    """
    target = f"{prefix}_s{cond}"
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0].lower()
        if base == target.lower():
            return p
    return None


def load_eeg_and_triggers(
    concat_files: list,
    origin_files: list,
    cond_nums=(1, 2, 4),
):
    """
    Load concatenated condition files and matching subject-origin files.

    Returns
    -------
    X : array
        Shape (n_windows_total, n_channels, window_len)
    y : array
        Shape (n_windows_total,)
    subjects : array
        Shape (n_windows_total,)
    """
    label_map = {1: 0, 2: 1, 4: 2}

    X_all, y_all, subj_all = [], [], []

    for cond in cond_nums:
        y_label = label_map[cond]

        concat_path = find_file_for_condition(concat_files, "concat", cond)
        origin_path = find_file_for_condition(origin_files, "origin", cond)

        if concat_path is None:
            raise FileNotFoundError(f"Could not find concat_s{cond}.mat in concat_files")
        if origin_path is None:
            raise FileNotFoundError(f"Could not find origin_s{cond}.mat in origin_files")

        x_mat = load_mat_smart(concat_path)
        o_mat = load_mat_smart(origin_path)

        if "concat" not in x_mat:
            raise KeyError(
                f"'concat' not found inside {concat_path}. Keys found: {list(x_mat.keys())}"
            )
        if "origin" not in o_mat:
            raise KeyError(
                f"'origin' not found inside {origin_path}. Keys found: {list(o_mat.keys())}"
            )

        # concat is expected as (window_len, channels, windows).
        Xc = np.asarray(x_mat["concat"])
        subj = np.asarray(o_mat["origin"]).squeeze()

        if Xc.ndim != 3 or Xc.shape[1] != 19:
            raise ValueError(
                f"'concat' in {concat_path} must be 3D (win_len, channels, windows). Got {Xc.shape}"
            )
        print(f"Loaded {concat_path}: concat shape {Xc.shape}, origin shape {subj.shape}")

        # Reorder once to inspect the window count using the last dimension.
        Xc = np.transpose(Xc, (2, 1, 0))
        n_win, n_ch, win_len = Xc.shape

        if subj.shape[0] != n_win:
            raise ValueError(
                f"'origin' in {origin_path} length {subj.shape[0]} does not match "
                f"'concat' windows {n_win}"
            )

        y = np.full(n_win, y_label, dtype=int)
        subj = subj.astype(int)

        X_all.append(Xc)
        y_all.append(y)
        subj_all.append(subj)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    subjects = np.concatenate(subj_all, axis=0)

    # Final shape and label checks before returning the data.
    assert X.shape[0] == y.shape[0] == subjects.shape[0], "X/y/subjects length mismatch"
    assert X.ndim == 3 and X.shape[1] == 19, f"Expected X as (N, 19, win_len). Got {X.shape}"
    assert set(np.unique(y)).issubset({0, 1, 2}), f"Labels must be in {{0,1,2}}. Got {np.unique(y)}"

    print("X:", X.shape, "y:", y.shape, "subjects:", subjects.shape)
    print("y unique:", np.unique(y), "subjects unique count:", len(np.unique(subjects)))

    return X, y, subjects


def make_stratified_group_folds(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 0,
):
    """
    Create stratified group k-fold splits after first removing three random
    seeded participants as holdout subjects.

    Returns
    -------
    folds : dict
        Each fold contains data, labels, and subject IDs.

    holdout_subject : dict
        Contains the holdout subject IDs and their corresponding data.
    """
    rng = np.random.default_rng(random_state)

    unique_subjects = np.unique(groups)
    if len(unique_subjects) < 4:
        raise ValueError(
            f"Need at least 4 unique subjects to create 3 holdout subjects. Got {len(unique_subjects)}."
        )

    # Randomly choose three subjects that will be completely excluded from
    # cross-validation and used only for final holdout testing.
    holdout_ids = rng.choice(unique_subjects, size=3, replace=False)

    holdout_mask = np.isin(groups, holdout_ids)
    remaining_mask = ~holdout_mask

    holdout_subject = {
        "subject_ids": holdout_ids,
        "data": X[holdout_mask],
        "labels": y[holdout_mask],
        "subjects": groups[holdout_mask],
    }

    X_remaining = X[remaining_mask]
    y_remaining = y[remaining_mask]
    groups_remaining = groups[remaining_mask]

    if len(np.unique(groups_remaining)) < n_splits:
        raise ValueError(
            f"After removing holdout subjects {holdout_ids}, only "
            f"{len(np.unique(groups_remaining))} subjects remain, which is fewer than n_splits={n_splits}."
        )

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    folds = {}

    # Each fold stores only its test portion. The training data is later
    # built by concatenating all remaining folds.
    for fold_idx, (_, idx) in enumerate(cv.split(X_remaining, y_remaining, groups_remaining)):
        fold_key = f"fold_{fold_idx}"

        folds[fold_key] = {
            "data":     X_remaining[idx],
            "labels":   y_remaining[idx],
            "subjects": groups_remaining[idx],
        }

    print(f"Holdout subjects: {holdout_ids}")
    print(
        "Holdout shapes:",
        holdout_subject["data"].shape,
        holdout_subject["labels"].shape,
        holdout_subject["subjects"].shape,
    )

    return folds, holdout_subject