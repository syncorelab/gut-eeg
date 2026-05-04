# script1_extract_channel_variance.py

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mne
from scipy.stats import kurtosis


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_DIR = Path(r"/home/gutproject/Desktop/guteeg/gut-eeg/data/gut_renamed")
OUTPUT_DIR = Path(r"/home/gutproject/Desktop/guteeg/gut-eeg/scripts/amplitude_reg_analyses/amp_reg_data")

# If BrainVision files are spread across subfolders, keep rglob.
FILE_GLOB = "*.vhdr"

RAW_CHANNELS = [
    "Fp1", "AF4", "AF7", "F1", "F4",
    "F2", "AF8", "F7", "F3", "C1",
    "FCz", "F8", "F5", "FC3", "FT7",
    "C2", "C4", "F6", "FT8",
]

TRIGGER_MARKERS = {"stimulus/s1", "stimulus/s2", "stimulus/s4"}

TMIN = -1.0
TMAX = 15.0
EPOCH_LEN_SEC = 16.0

DOWNSAMPLE_HZ = 512.0

# Per your instruction:
# lowest frequency that can "fit" inside a 16-second trial = 1 / 16 = 0.0625 Hz
HP_FREQ = 1.0 / EPOCH_LEN_SEC
LP_FREQ = 30.0

VAR_Z_THRESH = 3.0
KURT_Z_THRESH = 3.0
CORR_THRESH = 0.3
MAX_REMOVAL_PROP = 0.50

# Filtering choices for this first version
FILTER_METHOD = "fir"
FIR_DESIGN = "firwin"

# ------------------------------------------------------------
# NEW: flat channel / flat segment detection settings
# ------------------------------------------------------------
# Absolute tiny SD threshold. This is a last-resort "basically zero" check.
ABSOLUTE_FLAT_SD_THRESH = 1e-12

# Immediate channel exclusion if channel overall std is below this
# fraction of the participant's median channel std.
CHANNEL_FLAT_RELATIVE_THRESH = 0.05   # 5%

# Segment marked bad if its std is below this fraction of that channel's
# median segment std across all segments.
SEGMENT_FLAT_RELATIVE_THRESH = 0.05   # 5%

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
    """
    Extract first 3 digits from filename start.
    """
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


def get_event_id_from_annotations(raw: mne.io.BaseRaw) -> Dict[str, int]:
    """
    Return dict of normalized annotation descriptions -> event code,
    but only for the markers we care about.
    """
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    out = {}
    for desc, code in event_id.items():
        norm = normalize_marker(desc)
        if norm in TRIGGER_MARKERS:
            out[norm] = code
    return out


def make_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    """
    Create 16-second epochs from the three target markers only.
    """
    events, event_id_full = mne.events_from_annotations(raw, verbose=False)

    # Keep only our target markers
    event_id_keep = {}
    for desc, code in event_id_full.items():
        norm = normalize_marker(desc)
        if norm in TRIGGER_MARKERS:
            event_id_keep[norm] = code

    if not event_id_keep:
        raise ValueError("No matching trigger markers found among annotations.")

    print(f"  Matched markers: {sorted(event_id_keep.keys())}")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_keep,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        reject_by_annotation=False,
        verbose=False,
    )

    if len(epochs) == 0:
        raise ValueError("No valid epochs were created.")

    return epochs


def concatenate_epochs_to_raw(epochs: mne.Epochs) -> mne.io.RawArray:
    """
    Concatenate all epochs back into one continuous RawArray.
    Shape in epochs: (n_epochs, n_channels, n_times)
    Output raw data shape: (n_channels, n_epochs * n_times)
    """
    data = epochs.get_data(copy=True)  # (E, C, T)
    n_epochs, n_channels, n_times = data.shape

    concat = np.transpose(data, (1, 0, 2)).reshape(n_channels, n_epochs * n_times)

    info = mne.create_info(
        ch_names=epochs.ch_names,
        sfreq=epochs.info["sfreq"],
        ch_types=["eeg"] * n_channels,
    )
    raw_concat = mne.io.RawArray(concat, info, verbose=False)
    return raw_concat


def split_continuous_back_to_segments(data_2d: np.ndarray, n_segments: int, samples_per_segment: int) -> np.ndarray:
    """
    data_2d shape: (n_channels, n_segments * samples_per_segment)
    returns: (n_segments, n_channels, samples_per_segment)
    """
    n_channels, total_samples = data_2d.shape
    expected = n_segments * samples_per_segment
    if total_samples != expected:
        raise ValueError(
            f"Data length mismatch when re-splitting segments. "
            f"Expected {expected}, got {total_samples}"
        )

    arr = data_2d.reshape(n_channels, n_segments, samples_per_segment)
    arr = np.transpose(arr, (1, 0, 2))  # (segments, channels, samples)
    return arr


def safe_zscore(x: np.ndarray) -> np.ndarray:
    """
    Z-score with protection against zero std and tiny arrays.
    """
    x = np.asarray(x, dtype=float)

    if x.size < 2:
        return np.zeros_like(x, dtype=float)

    s = np.std(x, ddof=0)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x, dtype=float)

    return (x - np.mean(x)) / s


def compute_overall_channel_stds(segments: np.ndarray) -> np.ndarray:
    """
    segments shape: (n_segments, n_channels, n_samples)

    Returns overall std per channel across all concatenated task-only samples.
    """
    n_segments, n_channels, n_samples = segments.shape
    flat = np.transpose(segments, (1, 0, 2)).reshape(n_channels, n_segments * n_samples)
    return np.std(flat, axis=1, ddof=1)


def compute_channel_segment_qc_flags(
    segments: np.ndarray,
    channel_index: int,
) -> Dict[str, np.ndarray]:
    """
    segments shape: (n_segments, n_channels, n_samples)

    For one channel:
    - variance per segment
    - kurtosis per segment
    - corr(channel, mean(other channels)) per segment
    - std per segment
    - flat segment detection

    Then z-score variance and kurtosis across segments for that participant-channel.
    """
    n_segments, n_channels, _ = segments.shape

    x = segments[:, channel_index, :]  # (n_segments, n_samples)

    seg_var = np.var(x, axis=1, ddof=1)
    seg_std = np.std(x, axis=1, ddof=1)
    seg_kurt = kurtosis(x, axis=1, fisher=True, bias=False, nan_policy="omit")

    seg_corr = np.full(n_segments, np.nan, dtype=float)
    other_idx = [i for i in range(n_channels) if i != channel_index]

    for s in range(n_segments):
        this_ch = segments[s, channel_index, :]
        others_mean = np.mean(segments[s, other_idx, :], axis=0)

        this_std = np.std(this_ch)
        others_std = np.std(others_mean)

        if this_std == 0 or others_std == 0:
            seg_corr[s] = 0.0
        else:
            r = np.corrcoef(this_ch, others_mean)[0, 1]
            if not np.isfinite(r):
                r = 0.0
            seg_corr[s] = r

    var_z = safe_zscore(seg_var)
    kurt_z = safe_zscore(seg_kurt)

    # NEW: flat segment rule
    median_seg_std = np.nanmedian(seg_std)
    relative_flat_cutoff = SEGMENT_FLAT_RELATIVE_THRESH * median_seg_std if np.isfinite(median_seg_std) else np.nan

    flat_seg_mask = (
        (seg_std <= ABSOLUTE_FLAT_SD_THRESH) |
        (np.isfinite(relative_flat_cutoff) & (seg_std <= relative_flat_cutoff))
    )

    bad_mask = (
        (np.abs(var_z) > VAR_Z_THRESH) |
        (np.abs(kurt_z) > KURT_Z_THRESH) |
        (seg_corr < CORR_THRESH) |
        flat_seg_mask
    )

    return {
        "seg_var": seg_var,
        "seg_std": seg_std,
        "seg_kurt": seg_kurt,
        "seg_corr": seg_corr,
        "var_z": var_z,
        "kurt_z": kurt_z,
        "flat_seg_mask": flat_seg_mask,
        "bad_mask": bad_mask,
    }


def summarize_binary_mask(mask: np.ndarray) -> Tuple[int, int, float]:
    n_total = int(mask.size)
    n_bad = int(mask.sum())
    n_kept = n_total - n_bad
    pct_bad = (n_bad / n_total * 100.0) if n_total > 0 else np.nan
    return n_total, n_kept, pct_bad


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

    # Restrict to the requested channels and enforce order
    raw.pick(RAW_CHANNELS)
    # Reorder to match RAW_CHANNELS list
    raw.reorder_channels(RAW_CHANNELS)
    print(f"  Kept channels: {raw.ch_names}")

    # Epoch around trigger markers
    epochs = make_epochs(raw)
    print(f"  Number of valid epochs: {len(epochs)}")
    print(f"  Epoch shape: {epochs.get_data(copy=False).shape}")

    # Concatenate epochs back together
    raw_concat = concatenate_epochs_to_raw(epochs)
    print(f"  Concatenated task-only stream: shape={raw_concat.get_data().shape}, sfreq={raw_concat.info['sfreq']}")

    # Downsample
    print(f"  Resampling to {DOWNSAMPLE_HZ} Hz ...")
    raw_concat.resample(DOWNSAMPLE_HZ, npad="auto", verbose=False)
    print(f"  New sfreq after resample: {raw_concat.info['sfreq']}")

    # Filter
    print(f"  Filtering {HP_FREQ:.6f} - {LP_FREQ:.2f} Hz using {FILTER_METHOD}/{FIR_DESIGN} ...")
    raw_concat.filter(
        l_freq=HP_FREQ,
        h_freq=LP_FREQ,
        method=FILTER_METHOD,
        fir_design=FIR_DESIGN,
        verbose=False,
    )

    filt_data = raw_concat.get_data()  # (channels, time)
    sfreq = raw_concat.info["sfreq"]

    # Re-split into segments
    samples_per_segment = int(round(EPOCH_LEN_SEC * sfreq))
    n_segments = len(epochs)
    expected_total = n_segments * samples_per_segment

    if filt_data.shape[1] != expected_total:
        # Allow small tolerance (±30 samples) for resampling rounding
        tolerance = 30
        if abs(filt_data.shape[1] - expected_total) <= tolerance:
            print(f"  Note: minor resampling rounding ({filt_data.shape[1] - expected_total:+d} samples)")
            # Trim or pad to exact expected length
            if filt_data.shape[1] > expected_total:
                filt_data = filt_data[:, :expected_total]
            else:
                pad_width = ((0, 0), (0, expected_total - filt_data.shape[1]))
                filt_data = np.pad(filt_data, pad_width, mode='edge')
        else:
            raise ValueError(
                f"After resampling/filtering, data length mismatch for participant {participant_id}. "
                f"Expected {expected_total}, got {filt_data.shape[1]}"
            )

    segments = split_continuous_back_to_segments(filt_data, n_segments, samples_per_segment)
    print(f"  Re-split filtered task data into segments: {segments.shape}")

    channel_results: Dict[str, dict] = {}
    exclusion_log: List[dict] = []

    # NEW: immediate flat channel detection across full task-only filtered data
    overall_channel_stds = compute_overall_channel_stds(segments)
    median_channel_std = np.nanmedian(overall_channel_stds)

    print("  Overall filtered channel stds:")
    for ch_name, ch_std in zip(raw_concat.ch_names, overall_channel_stds):
        print(f"    {ch_name:<4} std={ch_std:.6g}")

    if not np.isfinite(median_channel_std):
        raise ValueError(f"Median channel std is not finite for participant {participant_id}")

    relative_channel_flat_cutoff = CHANNEL_FLAT_RELATIVE_THRESH * median_channel_std
    print(f"  Median channel std: {median_channel_std:.6g}")
    print(f"  Immediate flat-channel cutoff (relative): {relative_channel_flat_cutoff:.6g}")
    print(f"  Immediate flat-channel cutoff (absolute): {ABSOLUTE_FLAT_SD_THRESH:.6g}")

    for ch_idx, ch_name in enumerate(raw_concat.ch_names):
        overall_std = overall_channel_stds[ch_idx]

        # NEW: immediate flat channel exclusion
        is_flat_channel = (
            (overall_std <= ABSOLUTE_FLAT_SD_THRESH) or
            (overall_std <= relative_channel_flat_cutoff)
        )

        if is_flat_channel:
            print(
                f"    Channel {ch_name:<4} | overall_std={overall_std:.6g} "
                f"-> IMMEDIATE EXCLUSION as flat channel"
            )
            exclusion_log.append({
                "participant_id": participant_id,
                "raw_channel_name": ch_name,
                "n_segments_total": n_segments,
                "n_segments_kept": 0,
                "n_segments_removed": n_segments,
                "pct_segments_removed": 100.0,
                "reason": (
                    f"Immediate flat-channel exclusion "
                    f"(overall_std={overall_std:.6g}, "
                    f"median_channel_std={median_channel_std:.6g})"
                ),
            })
            channel_results[ch_name] = None
            continue

        qc = compute_channel_segment_qc_flags(segments, ch_idx)
        bad_mask = qc["bad_mask"]
        flat_seg_mask = qc["flat_seg_mask"]

        n_total, n_kept, pct_bad = summarize_binary_mask(bad_mask)
        n_removed = n_total - n_kept
        removal_prop = n_removed / n_total if n_total > 0 else np.nan
        n_flat_segments = int(flat_seg_mask.sum())

        print(
            f"    Channel {ch_name:<4} | total={n_total:>3} kept={n_kept:>3} "
            f"removed={n_removed:>3} ({pct_bad:5.1f}% removed) "
            f"| flat_segments={n_flat_segments:>3}"
        )

        if removal_prop > MAX_REMOVAL_PROP:
            print(f"      -> EXCLUDED for this participant-channel (>50% segments removed)")
            exclusion_log.append({
                "participant_id": participant_id,
                "raw_channel_name": ch_name,
                "n_segments_total": n_total,
                "n_segments_kept": n_kept,
                "n_segments_removed": n_removed,
                "pct_segments_removed": pct_bad,
                "reason": ">50% segments removed",
            })
            channel_results[ch_name] = None
            continue

        kept_segments = segments[~bad_mask, ch_idx, :]
        if kept_segments.size == 0:
            print("      -> EXCLUDED (no surviving data after segment rejection)")
            exclusion_log.append({
                "participant_id": participant_id,
                "raw_channel_name": ch_name,
                "n_segments_total": n_total,
                "n_segments_kept": n_kept,
                "n_segments_removed": n_removed,
                "pct_segments_removed": pct_bad,
                "reason": "No surviving data after QC",
            })
            channel_results[ch_name] = None
            continue

        concat_kept = kept_segments.reshape(-1)
        amplitude_variance = float(np.var(concat_kept, ddof=1))

        channel_results[ch_name] = {
            "participant_id": participant_id,
            "raw_channel_name": ch_name,
            "n_segments_total": n_total,
            "n_segments_kept": n_kept,
            "n_segments_removed": n_removed,
            "pct_segments_removed": pct_bad,
            "amplitude_variance": amplitude_variance,
        }

    return channel_results, exclusion_log


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vhdr_files = sorted(INPUT_DIR.rglob(FILE_GLOB))
    if not vhdr_files:
        raise FileNotFoundError(f"No BrainVision .vhdr files found in: {INPUT_DIR}")

    print(f"Found {len(vhdr_files)} BrainVision files.")

    # One list of row dicts per channel
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

    # Save metadata / config
    config = {
        "raw_channels": RAW_CHANNELS,
        "trigger_markers": sorted(TRIGGER_MARKERS),
        "tmin_sec": TMIN,
        "tmax_sec": TMAX,
        "epoch_len_sec": EPOCH_LEN_SEC,
        "downsample_hz": DOWNSAMPLE_HZ,
        "highpass_hz": HP_FREQ,
        "lowpass_hz": LP_FREQ,
        "filter_method": FILTER_METHOD,
        "fir_design": FIR_DESIGN,
        "var_z_thresh": VAR_Z_THRESH,
        "kurt_z_thresh": KURT_Z_THRESH,
        "corr_thresh": CORR_THRESH,
        "max_removal_prop": MAX_REMOVAL_PROP,
        "absolute_flat_sd_thresh": ABSOLUTE_FLAT_SD_THRESH,
        "channel_flat_relative_thresh": CHANNEL_FLAT_RELATIVE_THRESH,
        "segment_flat_relative_thresh": SEGMENT_FLAT_RELATIVE_THRESH,
    }

    config_path = OUTPUT_DIR / "processing_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved processing config: {config_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()