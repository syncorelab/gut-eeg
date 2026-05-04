import numpy as np
from scipy.io import savemat
from pathlib import Path


def generate_fake_concat_origin_mats(
    out_dir: str = "data/fake_windowed",
    seed: int = 99,
):
    """
    Generate fake already-windowed EEG/EGG-style data in the exact format
    expected by load_eeg_and_triggers().

    This creates 6 files:
        concat_s1.mat
        origin_s1.mat
        concat_s2.mat
        origin_s2.mat
        concat_s4.mat
        origin_s4.mat

    Each concat file contains:
        concat : ndarray, shape (n_windows, 19, win_len)

    Each origin file contains:
        origin : ndarray, shape (n_windows,)
                 participant ID for each window

    Design
    ------
    - 10 participants
    - 10 trials per class per participant
    - classes = 1, 2, 4
    - 1 whole-trial window per trial
    - win_len = 16 s * 512 Hz = 8192 samples

    This version is intentionally EASY:
    each class has a strong condition-specific signal pattern
    plus some noise/drift/spikes, so your model should be able
    to learn it extremely well.
    """

    rng = np.random.default_rng(seed)

    # -------------------------
    # Fixed experiment settings
    # -------------------------
    n_participants = 10
    participant_ids = np.arange(1, n_participants + 1, dtype=np.int32)

    class_codes = [1, 2, 4]
    trials_per_class_per_subject = 10

    n_channels = 19
    fs = 512
    trial_length_sec = 16
    win_len = fs * trial_length_sec  # 8192
    t = np.arange(win_len) / fs

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Strong class structure
    # -------------------------
    class_offsets = {
        1: 1.5,    # strong positive offset
        2: 0.0,    # centered
        4: -1.5,   # strong negative offset
    }

    class_freqs = {
        1: 22.0,
        2: 26.0,
        4: 30.0,
    }

    # Distinct spatial profiles across channels
    spatial_profiles = {
        1: np.linspace(0.8, 1.6, n_channels),
        2: np.concatenate([
            np.linspace(1.5, 0.7, n_channels // 2),
            np.linspace(0.7, 1.4, n_channels - n_channels // 2)
        ]),
        4: np.linspace(1.6, 0.8, n_channels),
    }

    # -------------------------
    # Helper: make one fake window for a specific condition
    # -------------------------
    def make_fake_window(cond: int, subj_id: int):
        """
        Returns one fake window of shape (19, win_len).

        Each class has:
        - a distinct DC offset
        - a distinct dominant frequency in the 20-30 Hz range
        - a distinct spatial profile across channels

        Still includes:
        - Gaussian noise
        - slow drift
        - sparse spikes
        - participant and channel variation
        """
        window = np.zeros((n_channels, win_len), dtype=np.float32)

        # participant-specific variation
        participant_gain = rng.normal(1.0, 0.08)
        participant_phase_shift = rng.normal(0.0, 0.15)

        cond_offset = class_offsets[cond]
        cond_freq = class_freqs[cond]
        cond_spatial = spatial_profiles[cond]

        # slight per-trial jitter to avoid exact copies
        freq_jitter = rng.normal(0.0, 0.25)
        amp_jitter = rng.normal(1.0, 0.05)

        for ch in range(n_channels):
            channel_gain = rng.normal(1.0, 0.05)
            phase = rng.uniform(0, 2 * np.pi) + participant_phase_shift

            # Strong class-defining oscillatory structure
            base_wave = np.sin(2 * np.pi * (cond_freq + freq_jitter) * t + phase)
            harmonic = 0.35 * np.sin(2 * np.pi * (cond_freq + 2.0 + freq_jitter) * t + 0.5 * phase)

            class_signal = (
                cond_offset
                + 2.5 * amp_jitter * cond_spatial[ch] * (base_wave + harmonic)
            )

            # Noise and nuisance components
            gaussian_noise = rng.normal(0.0, 0.35, size=win_len)
            drift = 0.03 * np.cumsum(rng.normal(0.0, 1.0, size=win_len)) / np.sqrt(win_len)
            spikes = (rng.random(win_len) < 0.0008) * rng.normal(0.0, 1.5, size=win_len)

            signal = (class_signal + gaussian_noise + drift + spikes)
            signal *= (participant_gain * channel_gain)

            window[ch, :] = signal.astype(np.float32)

        return window

    # -------------------------
    # Create one concat/origin pair per condition
    # -------------------------
    for cond in class_codes:
        concat_list = []
        origin_list = []

        for subj_id in participant_ids:
            for _ in range(trials_per_class_per_subject):
                x = make_fake_window(cond=cond, subj_id=subj_id)
                concat_list.append(x)
                origin_list.append(subj_id)

        # Stack to shape: (n_windows, 19, win_len)
        concat = np.stack(concat_list, axis=0).astype(np.float32)
        origin = np.asarray(origin_list, dtype=np.int32)

        # Shuffle windows within condition
        perm = rng.permutation(concat.shape[0])
        concat = concat[perm]
        origin = origin[perm]

        # Save files
        concat_path = out_dir / f"concat_s{cond}.mat"
        origin_path = out_dir / f"origin_s{cond}.mat"

        savemat(concat_path, {"concat": concat})
        savemat(origin_path, {"origin": origin})

        print(f"Saved {concat_path} with concat shape {concat.shape}")
        print(f"Saved {origin_path} with origin shape {origin.shape}")

    # Optional metadata file for debugging
    meta = {
        "participant_ids": participant_ids,
        "fs": np.array([[fs]], dtype=np.int32),
        "trial_length_sec": np.array([[trial_length_sec]], dtype=np.int32),
        "n_channels": np.array([[n_channels]], dtype=np.int32),
        "trials_per_class_per_subject": np.array([[trials_per_class_per_subject]], dtype=np.int32),
        "class_offsets_s1_s2_s4": np.array([[class_offsets[1], class_offsets[2], class_offsets[4]]], dtype=np.float32),
        "class_freqs_s1_s2_s4": np.array([[class_freqs[1], class_freqs[2], class_freqs[4]]], dtype=np.float32),
    }
    savemat(out_dir / "fake_data_meta.mat", meta)
    print(f"Saved metadata to {out_dir / 'fake_data_meta.mat'}")


if __name__ == "__main__":
    generate_fake_concat_origin_mats()