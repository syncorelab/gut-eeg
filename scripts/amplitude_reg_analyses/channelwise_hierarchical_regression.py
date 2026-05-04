from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import shapiro
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant


"""
hierarchical_channel_regression

Run hierarchical OLS regressions separately for each channel-level CSV file.

The script:
1. Loads participant-level predictor data from an Excel sheet
2. Cleans and encodes predictors
3. Merges predictors with each channel's dependent-variable data
4. Fits three regression blocks for each channel
5. Saves model summaries, coefficient tables, fit statistics, and VIF tables
6. Applies FDR correction across channels for final-block model and coefficient p-values
"""


# Paths and column names used throughout the analysis.
CHANNEL_CSV_DIR = Path(r"/home/gutproject/Desktop/guteeg/gut-eeg/scripts/amplitude_reg_analyses/amp_gastric_data")
XLSX_PATH = Path(r"/home/gutproject/Desktop/guteeg/gut-eeg/data/Participant_diet_info_sheet.xlsx")
OUTPUT_DIR = Path(r"/home/gutproject/Desktop/guteeg/gut-eeg/results/amp_reg_gestric")

CHANNEL_GLOB = "*.csv"

PARTICIPANT_COL = "Participant ID"
GENDER_COL = "Gender"
MEAL_COL = "Time Since Last Meal (minutes)"
TIME_COL = "Test Conducted Before or After 14:00"
PINCH_COL = "Pinch Test Result (mm)"
DIET_NOTES_COL = "Other Diet Notes"

DV_COL = "gastric_power"

# Predictors are added in blocks to test incremental model fit.
BLOCK1 = ["meal_minutes"]
BLOCK2 = ["pinch_mm"]
BLOCK3 = ["coffee_binary", "after_14_binary", "gender_female"]


def normalize_participant_id(x) -> str | None:
    """Extract a 1- or 2-digit participant ID from mixed text values."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    match = re.search(r"(\d{1,2})", s)
    return match.group(1) if match else None


def contains_coffee(text) -> int:
    """Return 1 if the diet notes mention coffee, otherwise 0."""
    if pd.isna(text):
        return 0
    return int(re.search(r"coffee", str(text), flags=re.IGNORECASE) is not None)


def parse_before_after_14(x) -> float:
    """Encode 'before' as 0 and 'after' as 1 for the 14:00 test-time variable."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == "before":
        return 0.0
    if s == "after":
        return 1.0
    return np.nan


def parse_gender(x) -> float:
    """Encode male as 0 and female as 1. Other or unrecognized values become missing."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s == "m":
        return 0.0
    if s == "k":
        return 1.0
    return np.nan


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas series to numeric values, setting invalid entries to NaN."""
    return pd.to_numeric(series, errors="coerce")


def compute_vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Compute variance inflation factors for each predictor."""
    if "const" not in X.columns:
        X = add_constant(X, has_constant="add")

    vif_rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        try:
            vif_val = variance_inflation_factor(X.values, i)
        except Exception:
            vif_val = np.nan
        vif_rows.append({"predictor": col, "VIF": vif_val})

    return pd.DataFrame(vif_rows)


def fit_ols(y: pd.Series, X: pd.DataFrame):
    """Fit an ordinary least squares model and an HC3 robust version."""
    X_const = add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const, missing="drop").fit()
    robust = model.get_robustcov_results(cov_type="HC3")
    return model, robust, X_const


def model_summary_dict(model, robust_model, block_name: str, predictors: List[str]) -> Dict:
    """Collect model fit and coefficient results into a JSON-friendly dictionary."""
    out = {
        "block": block_name,
        "n_obs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue) if model.fvalue is not None else np.nan,
        "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
        "aic": float(model.aic),
        "bic": float(model.bic),
        "predictors": predictors,
    }

    coef_rows = []
    conf = model.conf_int()
    robust_conf = robust_model.conf_int()

    # Robust outputs may not preserve parameter names, so they are aligned
    # manually to the original model parameter order.
    param_names = list(model.params.index)
    robust_params = np.asarray(robust_model.params)
    robust_pvals = np.asarray(robust_model.pvalues)
    robust_ci = np.asarray(robust_conf)

    for idx, name in enumerate(param_names):
        coef_rows.append({
            "term": name,
            "coef": float(model.params[name]),
            "se": float(model.bse[name]),
            "t": float(model.tvalues[name]),
            "p_raw": float(model.pvalues[name]),
            "ci_low": float(conf.loc[name, 0]),
            "ci_high": float(conf.loc[name, 1]),
            "coef_hc3": float(robust_params[idx]),
            "p_hc3": float(robust_pvals[idx]),
            "ci_low_hc3": float(robust_ci[idx, 0]),
            "ci_high_hc3": float(robust_ci[idx, 1]),
        })

    out["coefficients"] = coef_rows
    return out


def delta_r2(prev_model, curr_model) -> Dict[str, float]:
    """Compute the increase in R² from the previous block to the current block."""
    if prev_model is None:
        return {"delta_r2": np.nan}
    return {"delta_r2": float(curr_model.rsquared - prev_model.rsquared)}


def run_assumption_checks(model, X_const: pd.DataFrame) -> Dict:
    """Run a small set of standard regression assumption checks."""
    residuals = model.resid

    # Shapiro-Wilk is only run for sample sizes supported by the test.
    if len(residuals) >= 3 and len(residuals) <= 5000:
        shapiro_stat, shapiro_p = shapiro(residuals)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    try:
        bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, X_const)
    except Exception:
        bp_stat, bp_p, bp_f, bp_f_p = np.nan, np.nan, np.nan, np.nan

    try:
        dw = durbin_watson(residuals)
    except Exception:
        dw = np.nan

    vif_df = compute_vif_table(X_const.drop(columns=["const"], errors="ignore"))

    return {
        "shapiro_stat": float(shapiro_stat) if np.isfinite(shapiro_stat) else np.nan,
        "shapiro_p": float(shapiro_p) if np.isfinite(shapiro_p) else np.nan,
        "breusch_pagan_stat": float(bp_stat) if np.isfinite(bp_stat) else np.nan,
        "breusch_pagan_p": float(bp_p) if np.isfinite(bp_p) else np.nan,
        "breusch_pagan_f": float(bp_f) if np.isfinite(bp_f) else np.nan,
        "breusch_pagan_f_p": float(bp_f_p) if np.isfinite(bp_f_p) else np.nan,
        "durbin_watson": float(dw) if np.isfinite(dw) else np.nan,
        "vif_table": vif_df.to_dict(orient="records"),
    }


def load_predictor_data(xlsx_path: Path) -> pd.DataFrame:
    """Load, clean, and encode participant-level predictors from the Excel sheet."""
    df = pd.read_excel(xlsx_path, sheet_name="AnalysisReady", header=1)

    # Remove empty rows and columns before further processing.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").copy()

    # Trim column names to reduce matching errors caused by whitespace.
    df.columns = [str(c).strip() for c in df.columns]

    # Map spreadsheet column names to the internal names expected below.
    column_aliases = {
        "Participant ID (3 digits)": PARTICIPANT_COL,
        "Gender (Male / Female / Other)": GENDER_COL,
        "Time Since Last Meal (minutes, no decimals)": MEAL_COL,
        "Test Conducted Before or After 14:00": TIME_COL,
        "Pinch Test Result (mm, no decimals)": PINCH_COL,
        "Other Diet Notes": DIET_NOTES_COL,
    }

    rename_dict = {k: v for k, v in column_aliases.items() if k in df.columns}
    df = df.rename(columns=rename_dict)

    expected_cols = [
        PARTICIPANT_COL,
        GENDER_COL,
        MEAL_COL,
        TIME_COL,
        PINCH_COL,
        DIET_NOTES_COL,
    ]

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        print("Available XLSX columns:")
        print(df.columns.tolist())
        raise ValueError(f"Missing expected XLSX columns: {missing}")

    out = pd.DataFrame()
    out["participant_id"] = df[PARTICIPANT_COL].apply(normalize_participant_id)

    out["meal_minutes"] = safe_numeric(df[MEAL_COL])
    out["pinch_mm"] = safe_numeric(df[PINCH_COL])
    out["coffee_binary"] = df[DIET_NOTES_COL].apply(contains_coffee).astype(float)
    out["after_14_binary"] = df[TIME_COL].apply(parse_before_after_14)
    out["gender_female"] = df[GENDER_COL].apply(parse_gender)

    # Keep only rows with usable participant IDs for merging.
    out = out.dropna(subset=["participant_id"]).copy()

    # If the same participant appears multiple times, keep the first row.
    dupes = out["participant_id"].duplicated(keep=False)
    if dupes.any():
        print("WARNING: Duplicate participant IDs found in XLSX. Keeping first occurrence only:")
        print(out.loc[dupes, ["participant_id"]].sort_values("participant_id"))
        out = out.drop_duplicates(subset=["participant_id"], keep="first").copy()

    return out


def run_channel_regression(channel_csv: Path, predictors_df: pd.DataFrame, output_dir: Path) -> Dict:
    """Run the full hierarchical regression workflow for one channel CSV file."""
    channel_name = channel_csv.stem
    print("=" * 80)
    print(f"Running regression for channel: {channel_name}")

    ch_df = pd.read_csv(channel_csv)

    required_cols = ["participant_id", "raw_channel_name", DV_COL]
    missing = [c for c in required_cols if c not in ch_df.columns]
    if missing:
        raise ValueError(f"{channel_csv.name} is missing required columns: {missing}")

    # Skip bookkeeping files that may live in the same folder.
    if channel_name in {"channel_participant_exclusions", "failed_files"}:
        raise ValueError(f"Skipping non-channel CSV: {channel_csv.name}")

    ch_df["participant_id"] = ch_df["participant_id"].apply(normalize_participant_id)

    # Merge participant predictors with the channel-specific dependent variable.
    merged = pd.merge(ch_df, predictors_df, on="participant_id", how="inner")
    print(f"  Rows after merge: {len(merged)}")

    # Remove rows with missing values in the dependent variable or any predictor.
    needed_all = [DV_COL] + BLOCK1 + BLOCK2 + BLOCK3
    before_drop = len(merged)
    merged = merged.dropna(subset=needed_all).copy()
    dropped = before_drop - len(merged)

    print(f"  Rows after dropping missing / 'Other' gender / parse failures: {len(merged)}")
    print(f"  Dropped rows for this channel: {dropped}")

    if len(merged) < 10:
        raise ValueError(
            f"Too few rows for a stable regression in channel {channel_name}: n={len(merged)}"
        )

    # Print a compact summary of predictor distributions for quick checking.
    print("  Predictor sanity check:")
    print(f"    meal_minutes: min={merged['meal_minutes'].min()}, max={merged['meal_minutes'].max()}, mean={merged['meal_minutes'].mean():.2f}")
    print(f"    pinch_mm:     min={merged['pinch_mm'].min()}, max={merged['pinch_mm'].max()}, mean={merged['pinch_mm'].mean():.2f}")
    print(f"    coffee_binary counts:\n{merged['coffee_binary'].value_counts(dropna=False).sort_index()}")
    print(f"    after_14_binary counts:\n{merged['after_14_binary'].value_counts(dropna=False).sort_index()}")
    print(f"    gender_female counts:\n{merged['gender_female'].value_counts(dropna=False).sort_index()}")
    print(f"    DV variance summary: min={merged[DV_COL].min():.6g}, max={merged[DV_COL].max():.6g}, mean={merged[DV_COL].mean():.6g}")

    y = merged[DV_COL].astype(float)

    # Each block adds a new set of predictors to test incremental explanatory value.
    block_defs = [
        ("block1_meal_only", BLOCK1),
        ("block2_add_pinch", BLOCK1 + BLOCK2),
        ("block3_add_categoricals", BLOCK1 + BLOCK2 + BLOCK3),
    ]

    model_results = []
    fitted_models = []
    prev_model = None

    for block_name, predictors in block_defs:
        X = merged[predictors].astype(float)
        model, robust_model, X_const = fit_ols(y, X)

        checks = run_assumption_checks(model, X_const)
        summary = model_summary_dict(model, robust_model, block_name, predictors)
        summary.update(delta_r2(prev_model, model))
        summary["assumption_checks"] = checks

        model_results.append(summary)
        fitted_models.append((block_name, model, robust_model))

        prev_model = model

        print(f"  {block_name}: R²={model.rsquared:.4f}, adjR²={model.rsquared_adj:.4f}, model p={model.f_pvalue:.6g}")

    # Create one output folder per channel.
    channel_out_dir = output_dir / channel_name
    channel_out_dir.mkdir(parents=True, exist_ok=True)

    # Save the merged analysis table used for regression.
    merged_out = channel_out_dir / f"{channel_name}_analysis_dataset.csv"
    merged.to_csv(merged_out, index=False)

    # Save the full model summaries as JSON.
    json_out = channel_out_dir / f"{channel_name}_regression_summary.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(model_results, f, indent=2)

    # Save all coefficient estimates from all blocks in one table.
    all_coef_rows = []
    for block in model_results:
        for row in block["coefficients"]:
            all_coef_rows.append({
                "channel": channel_name,
                "block": block["block"],
                "term": row["term"],
                "coef": row["coef"],
                "se": row["se"],
                "t": row["t"],
                "p_raw": row["p_raw"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "coef_hc3": row["coef_hc3"],
                "p_hc3": row["p_hc3"],
                "ci_low_hc3": row["ci_low_hc3"],
                "ci_high_hc3": row["ci_high_hc3"],
            })

    coef_df = pd.DataFrame(all_coef_rows)
    coef_out = channel_out_dir / f"{channel_name}_coefficients.csv"
    coef_df.to_csv(coef_out, index=False)

    # Save one row per block with overall model fit statistics.
    fit_rows = []
    for block in model_results:
        fit_rows.append({
            "channel": channel_name,
            "block": block["block"],
            "n_obs": block["n_obs"],
            "r_squared": block["r_squared"],
            "adj_r_squared": block["adj_r_squared"],
            "delta_r2": block["delta_r2"],
            "f_statistic": block["f_statistic"],
            "f_pvalue": block["f_pvalue"],
            "aic": block["aic"],
            "bic": block["bic"],
            "shapiro_p": block["assumption_checks"]["shapiro_p"],
            "breusch_pagan_p": block["assumption_checks"]["breusch_pagan_p"],
            "durbin_watson": block["assumption_checks"]["durbin_watson"],
        })

    fit_df = pd.DataFrame(fit_rows)
    fit_out = channel_out_dir / f"{channel_name}_model_fits.csv"
    fit_df.to_csv(fit_out, index=False)

    # Save multicollinearity statistics for each block.
    vif_rows = []
    for block in model_results:
        for row in block["assumption_checks"]["vif_table"]:
            vif_rows.append({
                "channel": channel_name,
                "block": block["block"],
                "predictor": row["predictor"],
                "VIF": row["VIF"],
            })

    vif_df = pd.DataFrame(vif_rows)
    vif_out = channel_out_dir / f"{channel_name}_vif.csv"
    vif_df.to_csv(vif_out, index=False)

    # Return the final block so FDR correction can be done across channels later.
    final_block = model_results[-1]
    coef_final = pd.DataFrame(final_block["coefficients"])

    return {
        "channel": channel_name,
        "n_obs": final_block["n_obs"],
        "model_p_raw_final_block": final_block["f_pvalue"],
        "coef_table_final_block": coef_final,
        "channel_dir": str(channel_out_dir),
    }


def main():
    """Run the regression workflow across all channel files and save pooled summaries."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictors_df = load_predictor_data(XLSX_PATH)
    print("Loaded predictor table.")
    print(f"Predictor rows after initial cleaning: {len(predictors_df)}")
    print("Predictor preview:")
    print(predictors_df.head())

    # Collect only actual channel files from the folder.
    channel_csvs = sorted(
        [
            p for p in CHANNEL_CSV_DIR.glob(CHANNEL_GLOB)
            if p.stem not in {"channel_participant_exclusions", "failed_files"}
        ]
    )

    if not channel_csvs:
        raise FileNotFoundError(f"No channel CSV files found in {CHANNEL_CSV_DIR}")

    print(f"Found {len(channel_csvs)} channel CSV files.")

    channel_summaries = []
    failed = []

    for csv_path in channel_csvs:
        try:
            result = run_channel_regression(csv_path, predictors_df, OUTPUT_DIR)
            channel_summaries.append(result)
        except Exception as e:
            print(f"  !!! FAILED channel {csv_path.stem}: {e}")
            failed.append({
                "channel": csv_path.stem,
                "file": str(csv_path),
                "error": str(e),
            })

    # Apply FDR correction across channels for the final regression block.
    print("\nApplying FDR correction across channels ...")

    # Correct the final-block overall model p-value once per channel.
    overall_df = pd.DataFrame([
        {
            "channel": s["channel"],
            "n_obs": s["n_obs"],
            "model_p_raw_final_block": s["model_p_raw_final_block"],
        }
        for s in channel_summaries
    ])

    if not overall_df.empty:
        mask = overall_df["model_p_raw_final_block"].notna()
        corrected = np.full(len(overall_df), np.nan)

        if mask.sum() > 0:
            _, p_fdr, _, _ = multipletests(
                overall_df.loc[mask, "model_p_raw_final_block"].values,
                alpha=0.05,
                method="fdr_bh"
            )
            corrected[mask.values] = p_fdr

        overall_df["model_p_fdr_final_block"] = corrected

    overall_out = OUTPUT_DIR / "all_channels_final_block_model_pvalues.csv"
    overall_df.to_csv(overall_out, index=False)

    # Correct each predictor separately across channels in the final block.
    coef_rows = []
    for s in channel_summaries:
        coef_df = s["coef_table_final_block"].copy()
        coef_df["channel"] = s["channel"]
        coef_rows.append(coef_df)

    if coef_rows:
        coef_all = pd.concat(coef_rows, ignore_index=True)

        # The intercept is excluded from the cross-channel FDR step.
        coef_all = coef_all.rename(columns={"p_raw": "p_raw_final_block"})
        coef_all["p_fdr_final_block"] = np.nan

        for term in coef_all["term"].unique():
            if term == "const":
                continue

            idx = coef_all["term"] == term
            pvals = coef_all.loc[idx, "p_raw_final_block"].values

            valid = np.isfinite(pvals)
            corrected = np.full(len(pvals), np.nan)

            if valid.sum() > 0:
                _, p_fdr, _, _ = multipletests(
                    pvals[valid],
                    alpha=0.05,
                    method="fdr_bh"
                )
                corrected[np.where(valid)[0]] = p_fdr

            coef_all.loc[idx, "p_fdr_final_block"] = corrected

        coef_fdr_out = OUTPUT_DIR / "all_channels_final_block_coefficients_with_fdr.csv"
        coef_all.to_csv(coef_fdr_out, index=False)
    else:
        coef_all = pd.DataFrame()

    # Save channels that failed so they can be inspected later.
    failed_df = pd.DataFrame(failed)
    failed_out = OUTPUT_DIR / "failed_channels.csv"
    failed_df.to_csv(failed_out, index=False)

    print("\nSaved outputs:")
    print(f"  {overall_out}")
    if not coef_all.empty:
        print(f"  {OUTPUT_DIR / 'all_channels_final_block_coefficients_with_fdr.csv'}")
    print(f"  {failed_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()