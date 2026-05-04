import os
import glob
import pandas as pd
from scipy.stats import shapiro, friedmanchisquare, ttest_rel, wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests


# ----------------------------
# SETTINGS
# ----------------------------
input_folder = r"C:\Users\asbjorht\OneDrive - NTNU\Desktop\guteeg\gut-eeg\data\post-hoc_data\ratings"
output_file = r"C:\Users\asbjorht\OneDrive - NTNU\Desktop\guteeg\gut-eeg\results\anova_results.txt"


# ----------------------------
# LOAD ALL .XLSX FILES
# ----------------------------
files = glob.glob(os.path.join(input_folder, "*.xlsx"))
if not files:
    raise ValueError("No .xlsx files found in the folder.")

df_list = [pd.read_excel(file) for file in files]
data = pd.concat(df_list, ignore_index=True)

data = data[["participant_id", "folder", "stress_rating", "negativity_rating"]].copy()
data["folder"] = data["folder"].astype(str).str.strip()

expected_folders = {"Fear", "Hunger", "Nature"}
found_folders = set(data["folder"].unique())
if found_folders != expected_folders:
    raise ValueError(f"Expected folders {expected_folders}, found {found_folders}")


# ----------------------------
# DESCRIPTIVE STATISTICS
# ----------------------------
desc = (
    data.groupby(["participant_id", "folder"])
    .agg(
        stress_mean=("stress_rating", "mean"),
        stress_sd=("stress_rating", "std"),
        negativity_mean=("negativity_rating", "mean"),
        negativity_sd=("negativity_rating", "std"),
    )
    .reset_index()
)

desc_wide = desc.pivot(index="participant_id", columns="folder")
desc_wide.columns = [f"{var}_{folder}" for var, folder in desc_wide.columns]
desc_wide = desc_wide.reset_index()


# ----------------------------
# HELPERS
# ----------------------------
def get_subject_means(df, dv):
    means = (
        df.groupby(["participant_id", "folder"])[dv]
        .mean()
        .unstack()
        .dropna()
    )

    # enforce order
    means = means[["Fear", "Hunger", "Nature"]]
    return means


def shapiro_tests_on_means(df, dv):
    means = get_subject_means(df, dv)
    results = []
    violated = False

    for folder in ["Fear", "Hunger", "Nature"]:
        stat, p = shapiro(means[folder])
        results.append((folder, stat, p))
        if p < 0.05:
            violated = True

    return results, violated


def run_rm_anova(df, dv):
    means_long = (
        df.groupby(["participant_id", "folder"])[dv]
        .mean()
        .reset_index()
    )
    model = AnovaRM(means_long, depvar=dv, subject="participant_id", within=["folder"])
    result = model.fit()
    return result


def paired_ttests_bonf(df, dv):
    means = get_subject_means(df, dv)
    pairs = [("Fear", "Hunger"), ("Fear", "Nature"), ("Hunger", "Nature")]
    raw = []

    for a, b in pairs:
        t, p = ttest_rel(means[a], means[b], nan_policy="omit")
        raw.append((a, b, t, p))

    pvals = [x[3] for x in raw]
    _, p_corr, _, _ = multipletests(pvals, method="bonferroni")

    return [(a, b, t, p, pc) for (a, b, t, p), pc in zip(raw, p_corr)]


def run_friedman(df, dv):
    means = get_subject_means(df, dv)
    stat, p = friedmanchisquare(means["Fear"], means["Hunger"], means["Nature"])
    return stat, p


def wilcoxon_posthoc_bonf(df, dv):
    means = get_subject_means(df, dv)
    pairs = [("Fear", "Hunger"), ("Fear", "Nature"), ("Hunger", "Nature")]
    raw = []

    for a, b in pairs:
        stat, p = wilcoxon(means[a], means[b])
        raw.append((a, b, stat, p))

    pvals = [x[3] for x in raw]
    _, p_corr, _, _ = multipletests(pvals, method="bonferroni")

    return [(a, b, stat, p, pc) for (a, b, stat, p), pc in zip(raw, p_corr)]


# ----------------------------
# RUN ANALYSES
# ----------------------------
with open(output_file, "w", encoding="utf-8") as f:
    f.write("VIDEO RATING ANALYSES\n")
    f.write("=" * 60 + "\n\n")

    f.write("DESCRIPTIVE STATISTICS PER PARTICIPANT\n")
    f.write(desc_wide.to_string(index=False))
    f.write("\n\n")

    for dv in ["stress_rating", "negativity_rating"]:
        f.write(f"{dv.upper()}\n")
        f.write("-" * 60 + "\n")

        # Normality check on participant means per condition
        shapiro_results, normality_violated = shapiro_tests_on_means(data, dv)

        f.write("Shapiro-Wilk tests on participant means:\n")
        for folder, stat, p in shapiro_results:
            f.write(f"{folder}: W = {stat:.4f}, p = {p:.6f}\n")
        f.write("\n")

        # Always run standard repeated-measures ANOVA
        aov = run_rm_anova(data, dv)
        table = aov.anova_table

        f.write("Repeated-measures ANOVA (non-corrected version):\n")
        f.write(table.to_string())
        f.write("\n\n")

        f_value = table.loc["folder", "F Value"]
        p_value = table.loc["folder", "Pr > F"]
        df_num = table.loc["folder", "Num DF"]
        df_den = table.loc["folder", "Den DF"]

        f.write("ANOVA summary:\n")
        f.write(f"F = {f_value:.4f}\n")
        f.write(f"p = {p_value:.6f}\n")
        f.write(f"df_num = {df_num}\n")
        f.write(f"df_den = {df_den}\n")
        f.write("Sum of squares: not directly provided by AnovaRM\n\n")

        posthoc_t = paired_ttests_bonf(data, dv)
        f.write("Post hoc paired t-tests with Bonferroni correction:\n")
        for a, b, t_stat, p_raw, p_corr in posthoc_t:
            f.write(
                f"{a} vs {b}: t = {t_stat:.4f}, raw p = {p_raw:.6f}, Bonferroni p = {p_corr:.6f}\n"
            )
        f.write("\n")

        # If normality violated, also run Friedman + Wilcoxon
        if normality_violated:
            f.write("Normality violated in at least one condition.\n")
            f.write("Running nonparametric within-subject alternative: Friedman test.\n\n")

            friedman_stat, friedman_p = run_friedman(data, dv)
            f.write(f"Friedman test: chi2 = {friedman_stat:.4f}, p = {friedman_p:.6f}\n\n")

            posthoc_w = wilcoxon_posthoc_bonf(data, dv)
            f.write("Post hoc Wilcoxon signed-rank tests with Bonferroni correction:\n")
            for a, b, stat, p_raw, p_corr in posthoc_w:
                f.write(
                    f"{a} vs {b}: W = {stat:.4f}, raw p = {p_raw:.6f}, Bonferroni p = {p_corr:.6f}\n"
                )
            f.write("\n")

        f.write("\n" + "=" * 60 + "\n\n")

print(f"Done. Results saved to: {output_file}")