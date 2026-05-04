# gut-eeg

This repository contains the full analysis pipeline for a study that classifies three emotional states — **Fear**, **Hunger**, and **Nature (neutral)** — from multichannel abdominal EEG (EGG) recordings. The pipeline covers raw signal preprocessing in MATLAB, deep learning classification in Python/PyTorch, EEG amplitude regression analyses, and subjective video rating statistics.

---

## Table of contents

1. [Project overview](#project-overview)
2. [File overview](#file-overview)
3. [Using the scripts with your own data](#using-the-scripts-with-your-own-data)

---

## Project overview

### What the study does

Participants watched short video clips belonging to three conditions:

| Label | Condition code | Class index |
|-------|---------------|-------------|
| Fear  | `s1`          | `0`         |
| Hunger | `s2`         | `1`         |
| Nature (neutral) | `s4` | `2`      |

Multichannel abdominal EEG was recorded during viewing. The goal is to decode which condition a participant was experiencing from a 16-second trial of EEG signal.

### Pipeline at a glance

```
Raw BrainVision (.vhdr)
        │
        ▼
[MATLAB] egg_clean_multitrack   ← cleans, filters into multiple frequency tracks,
                                  runs ICA, exports per-subject .mat files
        │
        ▼
[MATLAB] Zscore_sort_withICA    ← normalises across subjects, concatenates trials
                                  by condition into concat_sX.mat / origin_sX.mat
        │
        ▼
[Python] run_CNN_classifier.py  ← random hyperparameter search over CNN or
      or run_CNN_LSTM_classifier.py   CNN-LSTM model, cross-validated by subject,
                                  final holdout evaluation
        │
        ▼
results/exp_YYYYMMDD_HHMMSS/    ← per-trial loss curves, confusion matrices,
                                  model weights, holdout summary JSON
```

Separate analysis tracks:

- **Amplitude regression** (`scripts/amplitude_reg_analyses/`) — extracts channel-level EEG amplitude variance from raw BrainVision files and runs hierarchical OLS regressions against participant metadata.
- **Gastric slow-wave power regression** (`amplitude_reg_analyses_gastricSW*.py`) — same framework but for the gastric frequency band (0.008–0.15 Hz) extracted from continuous data.
- **Video rating statistics** (`Subjective_video_ratings_check/`) — repeated-measures ANOVA on subjective stress and negativity ratings collected after viewing.

### Quick-start with fake data (no real data needed)

To verify that the deep learning pipeline runs end-to-end before using real data:

```bash
# 1. Generate synthetic data that mimics the real .mat format
python scripts/make_fake_data.py

# 2. Run the CNN classifier on the fake data
python scripts/run_CNN_on_fake_data.py

# 3. Or run the CNN-LSTM variant
python scripts/run_CNN_LSTM_on_fake_data.py
```

Results are written to `results/exp_*/`.

### Dependencies

**MATLAB preprocessing**
- EEGLAB (with `bva-io` and `cleanline` plugins)
- Signal Processing Toolbox

**Python deep learning and statistics**
- Python ≥ 3.10
- `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`
- `mne` (amplitude regression scripts only)
- `pandas`, `statsmodels`, `openpyxl` (regression and rating scripts)
- `h5py` (for MATLAB v7.3 `.mat` files)

---

## File overview

### Preprocessing — `preprocessing/pre-processing_pipeline/`

| File | What it does |
|------|-------------|
| `call_egg_clean.m` | **Entry point.** Loops over all `.vhdr` files in the raw data folder and calls `egg_clean_multitrack` on each one. Edit the `datapath` and `outpath` variables at the top before running. |
| `egg_clean_multitrack_UPDATED.m` | Core cleaning pipeline. Loads a BrainVision file into EEGLAB, maps channel names to canonical EGG1–EGG19 labels, runs CleanLine (50 Hz removal), detects/interpolates bad channels, runs ICA, then exports five frequency-band tracks (A = broadband; B1 = gastric 0.008–0.15 Hz; B2 = intestinal; B3 = neural burst .0625-30 Hz; B4 = autonomic 20-30 Hz) plus bipolar and average-reference variants. Each track is saved as a `.mat` via `export_for_dl`. |
| `export_for_dl.m` | Converts one EEGLAB dataset to a `.mat` (v7.3) with variables `data` (channels × timepoints × trials), `labels` (numeric trigger codes), and `meta` (sampling rate, channel names, epoch QC flags). Handles all common BrainVision trigger label formats (`s1`, `Stimulus/S 1`, `1`, etc.). |
| `make_bipolar.m` | Utility called inside the pipeline. Creates bipolar derivations (ch1 − ch2) from a list of electrode pairs and returns a new EEGLAB dataset. |
| `epoch_quality.m` | Computes per-epoch quality metrics (max amplitude, RMS, kurtosis, spectral flatness) and sets a flag on epochs above the 95th percentile. These values are stored in `meta.epoch_qc` and passed through to `Zscore_sort_withICA`. |
| `Zscore_sort_withICA.m` | Takes the per-subject `.mat` files produced by `export_for_dl`, normalises them across subjects (mode A = subject-level scaling; mode C = global per-channel standardisation), splits trials by trigger code, and saves three output files per condition: `concat_sX.mat` (time × channels × trials), `origin_sX.mat` (subject IDs per trial), and `quality_sX.mat` (per-trial quality metrics). These files are the direct input to the Python classifiers. |
| `run_Zscore_sort.m` | Thin wrapper that calls `Zscore_sort_withICA` with the paths and settings used in this project. Edit paths at the top if adapting. |
| `sort_files.py` | Small Python helper for organising raw files into a consistent folder structure before preprocessing. |

---

### Deep learning models — `src/models/`

| File | What it does |
|------|-------------|
| `cnn_model.py` | Shared CNN backbone (`cnnNet`). Applies 1–3 configurable 1D convolutional blocks (Conv → BatchNorm → Activation → MaxPool → Dropout) to a multichannel time series and returns feature maps. |
| `classifier_model.py` | Shared classifier head (`Classifier`). Takes a feature vector and maps it through one hidden linear layer to 3 output logits. |
| `cnn_classifier_model.py` | Full CNN model (`GUTNet`). Combines `cnnNet` with global average pooling over time and the classifier head. Used by `run_CNN_classifier.py`. |
| `cnn_lstm_model.py` | Full CNN-LSTM model (`GUTNet`). Extends the CNN backbone with an LSTM sequence model and an attention pooling layer before the classifier head. Used by `run_CNN_LSTM_classifier.py`. |
| `lstm_model.py` | LSTM and `AttentionPooling` components used by `cnn_lstm_model.py`. |

---

### Data loading — `src/dataloading/`

| File | What it does |
|------|-------------|
| `splits.py` | `load_eeg_and_triggers()` loads the `concat_sX.mat` and `origin_sX.mat` files and returns arrays `X` (N × 19 × win_len), `y` (N,), and `subjects` (N,). `make_stratified_group_folds()` holds out 3 random subjects for final evaluation and builds 5-fold stratified group cross-validation splits from the remainder. |
| `dataloader.py` | `make_dataloaders()` converts one fold of the split into shuffled PyTorch training and test `DataLoader` objects. |

---

### Training — `src/training/`

| File | What it does |
|------|-------------|
| `train_loop.py` | `train_GUTNet()` trains any model that accepts `(batch, channels, time)` input. Reads optimizer settings from a hyperparameter dict, runs the epoch loop, optionally applies early stopping, restores the best weights, saves the model state dict, and returns loss/accuracy histories and predictions. Used for the CNN model. |
| `train_loop_for_lstm.py` | Identical in structure to `train_loop.py` but adapted for the CNN-LSTM model. |
| `early_stopping.py` | `EarlyStopping` helper. Monitors a metric, saves the best model weights in memory, and sets a `should_stop` flag after a configurable patience window. |

---

### Evaluation — `src/evaluation/`

| File | What it does |
|------|-------------|
| `evaluation.py` | Plotting and metric utilities: `plot_confusion_matrices()` (train and test), `plot_train_test_accuracy()`, `plot_train_test_loss()`, `compute_accuracy()`, `compute_confusion_matrix()`, `compute_per_class_metrics()`, and `evaluate_on_new_subject()`. All plot functions save to disk when a `save_dir` is passed. |

---

### Hyperparameter configurations — `configs/`

| File | What it does |
|------|-------------|
| `CNN_Hparameters_distributions.py` | Search space (`HPARAM_DISTS`) for the CNN classifier. Defines categorical and continuous distributions for batch size, convolutional architecture, dropout rates, learning rate, and training schedule. |
| `CNN_LSTM_Hparameters_distributions.py` | Same structure as above but extended with LSTM-specific parameters (hidden size, number of layers, bidirectionality, attention dimensions). |

---

### Utilities — `src/utils/`

| File | What it does |
|------|-------------|
| `randomsearch.py` | `sample_hparams()` samples one configuration from a distribution spec dict. `is_valid_config()` enforces constraints (e.g. monotonically increasing channel counts across CNN blocks). |

---

### Main scripts — `scripts/`

| File | What it does |
|------|-------------|
| `run_CNN_classifier.py` | **Main entry point for CNN training.** Runs a random hyperparameter search (default 10 trials). For each trial: loads real data, builds stratified folds, trains the CNN, saves weights and plots. After all trials, reloads the best model and evaluates it once on the held-out subjects. Results go to `results/exp_YYYYMMDD_HHMMSS/`. |
| `run_CNN_LSTM_classifier.py` | Same pipeline as above but uses the CNN-LSTM architecture. |
| `run_CNN_on_fake_data.py` | Same as `run_CNN_classifier.py` but points to the synthetic data generated by `make_fake_data.py`. Use this to validate the pipeline without real data. |
| `run_CNN_LSTM_on_fake_data.py` | Same but for the CNN-LSTM model on fake data. |
| `make_fake_data.py` | Generates six synthetic `.mat` files (`concat_s1/s2/s4.mat`, `origin_s1/s2/s4.mat`) under `data/fake_windowed/`. Each condition has a distinct DC offset, dominant frequency, and spatial profile across 19 channels, making the classification task intentionally easy for pipeline testing. |
| `open_subject_file.py` | Interactive inspection utility. Call `inspect_subject_mat(path)` to pretty-print the contents of any `.mat` file (supports both standard and HDF5/v7.3 formats). Useful for checking that preprocessing produced the expected variable names and shapes. |

---

### Amplitude regression — `scripts/amplitude_reg_analyses/`

| File | What it does |
|------|-------------|
| `amplitude_calculation.py` | Reads raw BrainVision `.vhdr` files, epochs around the three condition triggers, filters (0.06–30 Hz, 512 Hz), rejects flat channels and noisy segments, and saves one CSV per channel containing per-participant amplitude variance. |
| `amplitude_reg_analyses_gastricSW.py` | Same framework but for gastric slow-wave power. Downsamples to 10 Hz, applies a 0.008–0.15 Hz Butterworth filter, computes Welch PSD in the gastric band, and saves per-channel gastric power estimates. |
| `amplitude_reg_analyses_gastricSW1.py` | Variant of the gastric analysis with slightly different QC or output settings. |
| `channelwise_hierarchical_regression.py` | Loads the per-channel CSVs, merges them with participant metadata (gender, time since last meal, test time, pinch test) from an Excel sheet, and fits three-block hierarchical OLS regressions. Saves coefficient tables, fit statistics, VIF tables, and applies FDR correction across channels. |

---

### Subjective ratings — `Subjective_video_ratings_check/`

| File | What it does |
|------|-------------|
| `ANOVA_videoratings.py` | Loads per-participant `.xlsx` rating files, runs a repeated-measures ANOVA on stress and negativity ratings across the three conditions (Fear / Hunger / Nature), and performs post-hoc pairwise tests (paired t-test or Wilcoxon) with FDR correction. |

---

## Using the scripts with your own data

### Step 1 — Preprocessing (MATLAB)

Your raw data must be BrainVision files (`.vhdr` / `.eeg` / `.vmrk`).

1. Open `preprocessing/pre-processing_pipeline/call_egg_clean.m`.
2. Set `datapath` to the folder containing your `.vhdr` files.
3. Set `outpath` to where you want the cleaned output.
4. Set `eeglabdir` to your EEGLAB installation.
5. Run the script. It processes every `.vhdr` file in the folder and produces a per-subject output folder with one `.mat` file per frequency track.

If your trigger codes differ from `s1 / s2 / s4`, edit the `parse_trigger_label` function inside `export_for_dl.m` to handle your codes, or pass your codes explicitly to `Zscore_sort_withICA`.

**Channel names** — the pipeline expects 19 channels. If your montage is different, update the `RAW_CHANNELS` list in the amplitude scripts and the channel mapping section inside `egg_clean_multitrack_UPDATED.m`.

### Step 2 — Concatenation and normalisation (MATLAB)

1. Open `preprocessing/pre-processing_pipeline/run_Zscore_sort.m`.
2. Set `in_dir` to the folder produced in Step 1 (one `.mat` per subject, pattern `*_B3_c1ref.mat`).
3. Set `out_dir` to where you want the concatenated outputs.
4. Choose normalisation mode: `"A"` (subject-level scaling) or `"C"` (global per-channel standardisation).
5. Run the script. It produces `concat_s1.mat`, `origin_s1.mat`, etc. for each condition.

### Step 3 — Deep learning (Python)

**Adapt the data paths.** In `run_CNN_classifier.py` (and the LSTM variant), find the `concat_files` and `origin_files` lists near the bottom of `main()` and replace the hardcoded absolute paths with paths to your own files:

```python
concat_files = [
    "path/to/your/data/concat_s1.mat",
    "path/to/your/data/concat_s2.mat",
    "path/to/your/data/concat_s4.mat",
]
origin_files = [
    "path/to/your/data/origin_s1.mat",
    "path/to/your/data/origin_s2.mat",
    "path/to/your/data/origin_s4.mat",
]
```

If you have different condition codes, also update the `cond_nums` argument passed to `load_eeg_and_triggers()` and the `label_map` dict inside `splits.py`.

**Adapt class names.** Change the `class_names` list (default `["fear", "hunger", "nature"]`) in the script to match your conditions.

**Change the number of trials.** Edit `N_TRIALS` at the top of `main()` to run more or fewer hyperparameter search trials.

**Run the classifier:**

```bash
python scripts/run_CNN_classifier.py
# or
python scripts/run_CNN_LSTM_classifier.py
```

Results are saved to `results/exp_YYYYMMDD_HHMMSS/`, with one subfolder per trial containing:
- `hparams_sampled.json` — the sampled hyperparameters
- `trained_model_state_dict.pt` — model weights
- `accuracy.png`, `loss.png` — learning curves
- `confusion_matrix_train.png`, `confusion_matrix_test.png`
- `trial_meta.json` — peak accuracy and epoch summary

After all trials, the best model is re-evaluated on the holdout subjects and results are saved in `best_trial_holdout_eval/holdout_eval_summary.json`.

**Inspect a .mat file before running:**

```python
from scripts.open_subject_file import inspect_subject_mat
inspect_subject_mat("path/to/your/file.mat")
```

### Step 4 — Amplitude regression analyses (Python)

1. Set `INPUT_DIR` in `amplitude_calculation.py` to your folder of `.vhdr` files.
2. Set `OUTPUT_DIR` to where CSVs should be saved.
3. Adjust `RAW_CHANNELS` and `TRIGGER_MARKERS` to match your data.
4. Run the script; it saves one CSV per channel and a processing config JSON.
5. Open `channelwise_hierarchical_regression.py`, point `CHANNEL_CSV_DIR` to the CSV output folder and `XLSX_PATH` to your participant metadata spreadsheet.
6. Update the column name constants at the top to match your spreadsheet headers.
7. Run to obtain per-channel regression summaries and FDR-corrected p-values.

### Step 5 — Subjective ratings (Python)

1. Set `input_folder` in `ANOVA_videoratings.py` to the folder containing your per-participant `.xlsx` rating files.
2. Set `output_file` to where the results text file should be saved.
3. Ensure your spreadsheets have columns `participant_id`, `folder` (condition name), `stress_rating`, and `negativity_rating`.
4. Run the script. It prints and saves descriptive statistics, the repeated-measures ANOVA result, and post-hoc pairwise comparisons.

---

### Data format reference

The Python classifiers expect `.mat` files in this exact format:

**`concat_sX.mat`** — contains variable `concat` with shape `(win_len, n_channels, n_trials)` before loading, which is transposed internally to `(n_trials, n_channels, win_len)`.

**`origin_sX.mat`** — contains variable `origin` with shape `(n_trials,)` holding the integer participant ID for each trial.

The fake data generator (`make_fake_data.py`) produces files in this format and is the easiest way to verify that your environment and paths are set up correctly before running on real data.
