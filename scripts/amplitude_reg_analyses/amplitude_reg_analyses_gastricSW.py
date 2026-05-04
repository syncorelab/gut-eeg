# script1_extract_gastric_power_continuous.py

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfiltfilt, welch
from scipy.integrate import trapezoid as trapz

# ============================================================
# USER SETTINGS
# ============================================================

INPUT_DIR = Path(r"/home/gutproject/Desktop/guteeg/gut-eeg/data/gut_renamed")
OUTPUT_DIR = Path(r"/home/gutproject/Desktop/guteeg/gut-eeg/scripts/amplitude_reg_analyses/amp_gastric_data")

FILE_GLOB = "*.vhdr"

RAW_CHANNELS = [
    "Fp1", "AF4", "AF7", "F1", "F4",
    "F2", "AF8", "F7", "F3", "C1",
    "FCz", "F8", "F5", "FC3", "FT7",
    "C2", "C4", "F6", "FT8",
]

TRIGGER_MARKERS = {"stimulus/s1", "stimulus/s2", "stimulus/s4"}

# Keep continuous data from:
# first target trigger onset
# to last target trigger onset + 15 sec
POST_TRIGGER_DURATION_SEC = 15.0

# Gastric slow-wave oriented downsampling
DOWNSAMPLE_HZ = 10.0

# Gastric filter
HP_FREQ = 0.0083
LP_FREQ = 0.15
BUTTER_ORDER = 4

# QC windows for continuous artifact rejection
QC_WINDOW_SEC = 60.0

# Immediate flat channel exclusion
ABSOLUTE_FLAT_SD_THRESH = 1e-12
CHANNEL_FLAT_RELATIVE_THRESH = 0.05  # exclude if channel std < 5% of participant median channel std

# Continuous-window artifact detection
WINDOW_STD_Z_THRESH = 3.0
WINDOW_PTP_Z_THRESH = 3.0
WINDOW_FLAT_RELATIVE_THRESH = 0.05   # reject window if std < 5% of channel median window std
MAX_REMOVAL_PROP = 0.50

# Welch PSD settings
WELCH_WINDOW_SEC = 60.0
WELCH_OVERLAP = 0.75

# ============================================================
# HELPERS
# ============================================================

def normalize_marker(desc: str) -> str:
    """
    Normalize BrainVision/MNE annotation strings so that
    'Stimulus/S 1', 'Stimulus/s1', etc. can match.
    """
    d = str(desc).strip().lower()
    d = d.replace(" ", "")
    d = d.replace("s ", "s")
    return d


def participant_id_from_filename(path: Path) -> str:
    match = re.match(r"(\d{3})", path.name)
    if not match:
        raise ValueError(f"Could not extract 3-digit participant ID from filename: {path.name}")
    return match.group(1)


def check_required_channels(raw: mne.io.BaseRaw, required_channels: List[str]) -> None:
    missing = [ch for ch in required_channels if ch not in raw.ch_names]
    if missing:
        raise ValueError(
            f"Missing required channels in {raw.filenames[0] if raw.filenames else 'raw'}: {missing}"
        )


def safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)

    if x.size < 2:
        return np.zeros_like(x, dtype=float)

    s = np.std(x, ddof=0)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x, dtype=float)

    return (x - np.mean(x)) / s


def butter_bandpass_filter_zerophase(
    data_2d: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass using sosfiltfilt.
    data_2d shape: (n_channels, n_samples)
    """
    nyq = sfreq / 2.0
    if high_hz >= nyq:
        raise ValueError(f"High cutoff {high_hz} Hz must be below Nyquist {nyq} Hz")

    sos = butter(
        N=order,
        Wn=[low_hz, high_hz],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    return sosfiltfilt(sos, data_2d, axis=1)


def get_target_event_samples(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Returns onset sample indices for the three target markers.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    keep_codes = []
    for desc, code in event_id.items():
        norm = normalize_marker(desc)
        if norm in TRIGGER_MARKERS:
            keep_codes.append(code)

    keep_codes = set(keep_codes)
    if not keep_codes:
        raise ValueError("No matching trigger markers found among annotations.")

    target_events = events[np.isin(events[:, 2], list(keep_codes))]
    if len(target_events) == 0:
        raise ValueError("No target trigger events found after matching event codes.")

    return target_events[:, 0]


def crop_to_continuous_trigger_block(raw: mne.io.BaseRaw) -> Tuple[mne.io.BaseRaw, int]:
    """
    Keep data continuously from first target trigger onset
    to last target trigger onset + 15 sec.
    """
    target_samples = get_target_event_samples(raw)

    first_sample = int(np.min(target_samples))
    last_start_sample = int(np.max(target_samples))
    end_sample = int(last_start_sample + round(POST_TRIGGER_DURATION_SEC * raw.info["sfreq"]))

    if end_sample > raw.n_times:
        end_sample = raw.n_times

    tmin = first_sample / raw.info["sfreq"]
    tmax = (end_sample - 1) / raw.info["sfreq"]

    cropped = raw.copy().crop(tmin=tmin, tmax=tmax)
    return cropped, len(target_samples)


def split_into_windows_2d(data_1d: np.ndarray, window_samples: int) -> Tuple[np.ndarray, int]:
    """
    Split 1D data into non-overlapping windows.
    Drops the leftover tail if it does not fit a full window.
    Returns:
      windows shape: (n_windows, window_samples)
      dropped_tail_samples
    """
    n_samples = data_1d.shape[0]
    n_windows = n_samples // window_samples
    usable = n_windows * window_samples
    tail = n_samples - usable

    if n_windows == 0:
        return np.empty((0, window_samples), dtype=data_1d.dtype), n_samples

    windows = data_1d[:usable].reshape(n_windows, window_samples)
    return windows, tail


def compute_overall_channel_stds(data_2d: np.ndarray) -> np.ndarray:
    """
    data_2d shape: (n_channels, n_samples)
    """
    return np.std(data_2d, axis=1, ddof=1)


def compute_window_artifact_mask(
    channel_data: np.ndarray,
    sfreq: float,
    window_sec: float,
) -> Dict[str, np.ndarray]:
    """
    Continuous artifact rejection for one channel using 60 s windows.

    Flags windows as bad if:
    - window std is an outlier
    - window peak-to-peak amplitude is an outlier
    - window is nearly flat
    """
    window_samples = int(round(window_sec * sfreq))
    windows, dropped_tail = split_into_windows_2d(channel_data, window_samples)

    if windows.shape[0] == 0:
        raise ValueError(
            f"Not enough data for even one full QC window of {window_sec} sec. "
            f"Need at least {window_samples} samples, got {len(channel_data)}."
        )

    win_std = np.std(windows, axis=1, ddof=1)
    win_ptp = np.ptp(windows, axis=1)

    std_z = safe_zscore(win_std)
    ptp_z = safe_zscore(win_ptp)

    median_win_std = np.nanmedian(win_std)
    relative_flat_cutoff = WINDOW_FLAT_RELATIVE_THRESH * median_win_std if np.isfinite(median_win_std) else np.nan

    flat_mask = (
        (win_std <= ABSOLUTE_FLAT_SD_THRESH) |
        (np.isfinite(relative_flat_cutoff) & (win_std <= relative_flat_cutoff))
    )

    bad_mask = (
        (np.abs(std_z) > WINDOW_STD_Z_THRESH) |
        (np.abs(ptp_z) > WINDOW_PTP_Z_THRESH) |
        flat_mask
    )

    return {
        "window_samples": window_samples,
        "windows": windows,
        "dropped_tail_samples": dropped_tail,
        "win_std": win_std,
        "win_ptp": win_ptp,
        "std_z": std_z,
        "ptp_z": ptp_z,
        "flat_mask": flat_mask,
        "bad_mask": bad_mask,
    }


def integrated_band_power_welch(
    signal_1d: np.ndarray,
    sfreq: float,
    low_hz: float,
    high_hz: float,
    welch_window_sec: float,
    overlap_frac: float = 0.75,
) -> float:
    """
    Compute integrated band power using Welch PSD.

    This is the area under the PSD curve within the gastric band.
    """
    nperseg = int(round(welch_window_sec * sfreq))
    if len(signal_1d) < nperseg:
        raise ValueError(
            f"Signal too short for Welch window of {welch_window_sec} sec. "
            f"Need at least {nperseg} samples, got {len(signal_1d)}."
        )

    noverlap = int(round(nperseg * overlap_frac))
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    freqs, psd = welch(
        signal_1d,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )

    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        raise ValueError("No Welch frequency bins fall inside the requested gastric band.")

    band_power = trapz(psd[band_mask], freqs[band_mask])
    return float(band_power)


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_one_file(vhdr_path: Path) -> Tuple[Dict[str, dict], List[dict]]:
    """
    Returns:
      channel_results: dict keyed by raw channel name, each value is a row dict or None
      exclusion_log: list of exclusion dicts
    """
    participant_id = participant_id_from_filename(vhdr_path)

    print("=" * 80)
    print(f"Processing participant {participant_id}: {vhdr_path.name}")

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    print(f"  Loaded raw shape: n_channels={len(raw.ch_names)}, n_times={raw.n_times}, sfreq={raw.info['sfreq']}")

    check_required_channels(raw, RAW_CHANNELS)

    # Keep requested channels and order
    raw.pick(RAW_CHANNELS)
    raw.reorder_channels(RAW_CHANNELS)
    print(f"  Kept channels: {raw.ch_names}")

    # Keep continuous data from first target trigger to last target trigger + 15 s
    raw_continuous, n_target_triggers = crop_to_continuous_trigger_block(raw)
    print(f"  Matched target triggers: {n_target_triggers}")
    print(f"  Continuous retained block shape before resample: {raw_continuous.get_data().shape}")

    # Downsample to gastric-slow-wave-friendly rate
    print(f"  Resampling to {DOWNSAMPLE_HZ} Hz ...")
    raw_continuous.resample(DOWNSAMPLE_HZ, npad='auto', verbose=False)
    print(f"  New sfreq after resample: {raw_continuous.info['sfreq']}")

    # Butterworth bandpass, zero-phase
    print(f"  Zero-phase Butterworth bandpass: {HP_FREQ:.4f} - {LP_FREQ:.2f} Hz, order={BUTTER_ORDER}")
    data = raw_continuous.get_data()
    sfreq = raw_continuous.info["sfreq"]

    filt_data = butter_bandpass_filter_zerophase(
        data_2d=data,
        sfreq=sfreq,
        low_hz=HP_FREQ,
        high_hz=LP_FREQ,
        order=BUTTER_ORDER,
    )

    print(f"  Filtered continuous data shape: {filt_data.shape}")

    channel_results: Dict[str, dict] = {}
    exclusion_log: List[dict] = []

    # Immediate flat-channel exclusion
    overall_channel_stds = compute_overall_channel_stds(filt_data)
    median_channel_std = np.nanmedian(overall_channel_stds)

    print("  Overall filtered channel stds:")
    for ch_name, ch_std in zip(raw_continuous.ch_names, overall_channel_stds):
        print(f"    {ch_name:<4} std={ch_std:.6g}")

    if not np.isfinite(median_channel_std):
        raise ValueError(f"Median channel std is not finite for participant {participant_id}")

    relative_channel_flat_cutoff = CHANNEL_FLAT_RELATIVE_THRESH * median_channel_std
    print(f"  Median channel std: {median_channel_std:.6g}")
    print(f"  Flat-channel cutoff (relative): {relative_channel_flat_cutoff:.6g}")
    print(f"  Flat-channel cutoff (absolute): {ABSOLUTE_FLAT_SD_THRESH:.6g}")

    for ch_idx, ch_name in enumerate(raw_continuous.ch_names):
        ch_data = filt_data[ch_idx, :]
        overall_std = overall_channel_stds[ch_idx]

        # Immediate flat-channel exclusion
        is_flat_channel = (
            (overall_std <= ABSOLUTE_FLAT_SD_THRESH) or
            (overall_std <= relative_channel_flat_cutoff)
        )

        if is_flat_channel:
            print(
                f"    Channel {ch_name:<4} | overall_std={overall_std:.6g} "
                f"-> IMMEDIATE EXCLUSION as flat channel"
            )
            channel_results[ch_name] = None
            exclusion_log.append({
                "participant_id": participant_id,
                "raw_channel_name": ch_name,
                "n_segments_total": np.nan,
                "n_segments_kept": 0,
                "n_segments_removed": np.nan,
                "pct_segments_removed": 100.0,
                "reason": (
                    f"Immediate flat-channel exclusion "
                    f"(overall_std={overall_std:.6g}, "
                    f"median_channel_std={median_channel_std:.6g})"
                ),
            })
            continue

        # Continuous artifact rejection using 60 s windows
        qc = compute_window_artifact_mask(
            channel_data=ch_data,
            sfreq=sfreq,
            window_sec=QC_WINDOW_SEC,
        )

        bad_mask = qc["bad_mask"]
        flat_mask = qc["flat_mask"]
        windows = qc["windows"]
        window_samples = qc["window_samples"]
        dropped_tail_samples = qc["dropped_tail_samples"]

        n_total = int(len(bad_mask))
        n_removed = int(np.sum(bad_mask))
        n_kept = n_total - n_removed
        pct_removed = (n_removed / n_total * 100.0) if n_total > 0 else np.nan
        removal_prop = n_removed / n_total if n_total > 0 else np.nan
        n_flat_windows = int(np.sum(flat_mask))

        print(
            f"    Channel {ch_name:<4} | windows={n_total:>3} kept={n_kept:>3} "
            f"removed={n_removed:>3} ({pct_removed:5.1f}% removed) "
            f"| flat_windows={n_flat_windows:>3} | dropped_tail_samples={dropped_tail_samples}"
        )

        if removal_prop > MAX_REMOVAL_PROP:
            print(f"      -> EXCLUDED for this participant-channel (>50% windows removed)")
            channel_results[ch_name] = None
            exclusion_log.append({
                "participant_id": participant_id,
                "raw_channel_name": ch_name,
                "n_segments_total": n_total,
                "n_segments_kept": n_kept,
                "n_segments_removed": n_removed,
                "pct_segments_removed": pct_removed,
                "reason": ">50% QC windows removed",
            })
            continue

        kept_windows = windows[~bad_mask, :]
        if kept_windows.size == 0:
            print("      -> EXCLUDED (no surviving data after artifact rejection)")
            channel_results[ch_name] = None
            exclusion_log.append({
                "participant_id": participant_id,
                "raw_channel_name": ch_name,
                "n_segments_total": n_total,
                "n_segments_kept": 0,
                "n_segments_removed": n_removed,
                "pct_segments_removed": 100.0,
                "reason": "No surviving data after QC",
            })
            continue

        cleaned_signal = kept_windows.reshape(-1)

        # Final gastric metric: integrated band power in 0.0083 - 0.15 Hz
        gastric_power = integrated_band_power_welch(
            signal_1d=cleaned_signal,
            sfreq=sfreq,
            low_hz=HP_FREQ,
            high_hz=LP_FREQ,
            welch_window_sec=WELCH_WINDOW_SEC,
            overlap_frac=WELCH_OVERLAP,
        )

        channel_results[ch_name] = {
            "participant_id": participant_id,
            "raw_channel_name": ch_name,
            "n_segments_total": n_total,
            "n_segments_kept": n_kept,
            "n_segments_removed": n_removed,
            "pct_segments_removed": pct_removed,
            "gastric_power": gastric_power,
        }

    return channel_results, exclusion_log


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vhdr_files = sorted(INPUT_DIR.rglob(FILE_GLOB))
    if not vhdr_files:
        raise FileNotFoundError(f"No BrainVision .vhdr files found in: {INPUT_DIR}")

    print(f"Found {len(vhdr_files)} BrainVision files.")

    per_channel_rows: Dict[str, List[dict]] = {ch: [] for ch in RAW_CHANNELS}
    all_exclusions: List[dict] = []
    failed_files: List[dict] = []

    for fpath in vhdr_files:
        try:
            channel_results, exclusion_log = process_one_file(fpath)
            all_exclusions.extend(exclusion_log)

            for ch_name, row in channel_results.items():
                if row is not None:
                    per_channel_rows[ch_name].append(row)

        except Exception as e:
            pid = None
            try:
                pid = participant_id_from_filename(fpath)
            except Exception:
                pid = "UNKNOWN"

            print(f"  !!! FAILED: {fpath.name}")
            print(f"      Error: {e}")

            failed_files.append({
                "participant_id": pid,
                "file": str(fpath),
                "error": str(e),
            })

    # Save one CSV per channel
    print("\nSaving per-channel CSV files ...")
    for ch_name, rows in per_channel_rows.items():
        df = pd.DataFrame(rows)

        if not df.empty and "participant_id" in df.columns:
            df = df.sort_values("participant_id").reset_index(drop=True)

        out_csv = OUTPUT_DIR / f"{ch_name}.csv"
        df.to_csv(out_csv, index=False)
        print(f"  Saved {out_csv.name}: {len(df)} participants")

    # Save exclusions
    exclusions_df = pd.DataFrame(all_exclusions)
    exclusions_path = OUTPUT_DIR / "channel_participant_exclusions.csv"
    exclusions_df.to_csv(exclusions_path, index=False)
    print(f"Saved exclusions log: {exclusions_path}")

    # Save failed files
    failed_df = pd.DataFrame(failed_files)
    failed_path = OUTPUT_DIR / "failed_files.csv"
    failed_df.to_csv(failed_path, index=False)
    print(f"Saved failed file log: {failed_path}")

    # Save config
    config = {
        "raw_channels": RAW_CHANNELS,
        "trigger_markers": sorted(TRIGGER_MARKERS),
        "continuous_keep_rule": "from first target trigger onset to last target trigger onset + 15 sec",
        "downsample_hz": DOWNSAMPLE_HZ,
        "filter_type": "zero-phase Butterworth bandpass",
        "butter_order": BUTTER_ORDER,
        "highpass_hz": HP_FREQ,
        "lowpass_hz": LP_FREQ,
        "qc_window_sec": QC_WINDOW_SEC,
        "window_std_z_thresh": WINDOW_STD_Z_THRESH,
        "window_ptp_z_thresh": WINDOW_PTP_Z_THRESH,
        "absolute_flat_sd_thresh": ABSOLUTE_FLAT_SD_THRESH,
        "channel_flat_relative_thresh": CHANNEL_FLAT_RELATIVE_THRESH,
        "window_flat_relative_thresh": WINDOW_FLAT_RELATIVE_THRESH,
        "max_removal_prop": MAX_REMOVAL_PROP,
        "welch_window_sec": WELCH_WINDOW_SEC,
        "welch_overlap": WELCH_OVERLAP,
        "feature_name": "gastric_power",
        "feature_definition": "integrated Welch PSD band power in 0.0083-0.15 Hz",
    }

    config_path = OUTPUT_DIR / "processing_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved processing config: {config_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()