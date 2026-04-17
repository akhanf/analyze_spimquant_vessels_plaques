"""Treatment-effect and genotype-modulation analysis figures.

Reads  : data_{cohort}.parquet
Writes (one file per figure):
  fig_{cohort}_boxplots_treatment_genotype.png   2×2 boxplots of burden metrics
  fig_{cohort}_violin_plaque_size.png            violin plots of per-plaque diameter
  fig_{cohort}_interaction_plots.png             mean±SEM interaction plots (treatment × genotype)
  fig_{cohort}_stratified_by_genotype.png        treatment boxplots stratified by genotype
  fig_{cohort}_ecdf_size_by_genotype.png         ECDFs of plaque size per genotype
  fig_{cohort}_size_distribution_by_genotype.png histograms of plaque size per genotype
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.formula.api import ols  # noqa: E402
from statsmodels.stats.anova import anova_lm  # noqa: E402
from statsmodels.stats.multicomp import pairwise_tukeyhsd  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Default plot config (used when running outside Snakemake) ─────────────────
DEFAULT_PLOT_CONFIG = {
    "factors": {
        "treatment": {
            "column": "treatment",
            "order": ["PBS", "Lecanemab"],
            "palette": {"PBS": "#4C72B0", "Lecanemab": "#DD8452"},
        },
        "genotype": {
            "column": "genotype",
            "order": ["ApoE3", "ApoE4"],
            "palette": {"ApoE3": "#55A868", "ApoE4": "#C44E52"},
        },
        "sex": {
            "column": "sex",
            "order": ["M", "F"],
            "palette": {"M": "#8172B2", "F": "#CCB974"},
        },
    },
    "primary_metric": "plaque_density",
    "volume_threshold_ml": 1e-4,
}

# ── Snakemake integration ────────────────────────────────────────────────────
if "snakemake" in dir():
    input_parquet = str(snakemake.input.parquet)  # noqa: F821
    output_figs = dict(snakemake.output)  # noqa: F821
    cohort = snakemake.wildcards.cohort  # noqa: F821
    cfg = snakemake.params.plot_config  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()
    input_parquet = args.parquet
    cohort = args.cohort
    cfg = DEFAULT_PLOT_CONFIG
    output_figs = {
        fig: f"{args.output_dir}/fig_{cohort}_{fig}.png"
        for fig in [
            "boxplots_treatment_genotype", "violin_plaque_size", "interaction_plots",
            "stratified_by_genotype", "ecdf_size_by_genotype",
            "size_distribution_by_genotype",
        ]
    }

# ── Extract plot config ───────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)

treat_cfg = cfg["factors"]["treatment"]
TREAT_COL = treat_cfg["column"]
TREAT_ORDER = treat_cfg["order"]
TREAT_PALETTE = treat_cfg["palette"]

geno_cfg = cfg["factors"]["genotype"]
GENO_COL = geno_cfg["column"]
GENO_ORDER = geno_cfg["order"]
GENO_PALETTE = geno_cfg["palette"]

sex_cfg = cfg["factors"]["sex"]
SEX_COL = sex_cfg["column"]

VOL_THRESH_ML = cfg.get("volume_threshold_ml", 1e-4)


# ── Helper functions ─────────────────────────────────────────────────────────
def compute_subject_metrics(plaque_df):
    """Aggregate plaque-level data to one row per subject."""
    grp = plaque_df.groupby(
        ["subject", TREAT_COL, GENO_COL, SEX_COL], observed=True
    )
    metrics = grp.agg(
        plaque_count=("nvoxels", "size"),
        total_vol_ml=("plaque_vol_ml", "sum"),
        mean_vol_ml=("plaque_vol_ml", "mean"),
        median_vol_ml=("plaque_vol_ml", "median"),
        mean_diam_um=("equiv_diam_um", "mean"),
        median_diam_um=("equiv_diam_um", "median"),
        total_vol_um3=("plaque_vol_um3", "sum"),
    ).reset_index()
    for col in [
        "plaque_count", "total_vol_ml", "mean_vol_ml", "median_vol_ml",
        "mean_diam_um", "median_diam_um", "total_vol_um3",
    ]:
        metrics[f"log_{col}"] = np.log(metrics[col])
    return metrics


def boxstrip(ax, data, x, y, hue=None, order=None, hue_order=None, palette=None,
             ylabel=None, yscale="linear"):
    """Paired boxplot + stripplot helper."""
    sns.boxplot(data=data, x=x, y=y, hue=hue,
                order=order, hue_order=hue_order, palette=palette,
                fill=False, linewidth=1.2, fliersize=0, ax=ax)
    sns.stripplot(data=data, x=x, y=y, hue=hue,
                  order=order, hue_order=hue_order, palette=palette,
                  dodge=True, alpha=0.7, size=6, jitter=True, ax=ax)
    if yscale != "linear":
        ax.set_yscale(yscale)
    if ylabel:
        ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    n = len(hue_order) if hue_order else 2
    ax.legend(handles[:n], labels[:n], title=hue, fontsize=9)


def run_twoway_anova(data, outcome):
    """Two-way OLS ANOVA (or one-way when only one group present)."""
    n_geno = data[GENO_COL].nunique()
    if n_geno > 1:
        formula = f"{outcome} ~ C({TREAT_COL}) * C({GENO_COL})"
    else:
        formula = f"{outcome} ~ C({TREAT_COL})"
    model = ols(formula, data=data).fit()
    return anova_lm(model, typ=2)


def interaction_plot_ax(ax, data, outcome, ylabel, yscale="linear"):
    """Mean ± SEM interaction plot (treatment on x, genotype as lines)."""
    agg = (
        data.groupby([TREAT_COL, GENO_COL], observed=True)[outcome]
        .agg(["mean", "sem"])
        .reset_index()
    )
    for geno, grp in agg.groupby(GENO_COL, observed=True):
        color = GENO_PALETTE.get(geno, "#888888")
        ax.plot(grp[TREAT_COL].astype(str), grp["mean"],
                marker="o", linewidth=2, color=color, label=geno)
        ax.errorbar(grp[TREAT_COL].astype(str), grp["mean"],
                    yerr=grp["sem"], fmt="none", color=color, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(TREAT_COL.replace("_", " ").title())
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.legend(title=GENO_COL.replace("_", " ").title(), fontsize=9)


def tukey_posthoc(data, outcome):
    """Run Tukey HSD across all treatment × genotype combinations."""
    d = data.copy()
    d["group"] = d[TREAT_COL].astype(str) + " | " + d[GENO_COL].astype(str)
    return pairwise_tukeyhsd(d[outcome], d["group"], alpha=0.05)


# ── Load and prepare data ────────────────────────────────────────────────────
df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()
df[TREAT_COL] = pd.Categorical(df[TREAT_COL], categories=TREAT_ORDER, ordered=True)
geno_present = [g for g in GENO_ORDER if (df[GENO_COL] == g).any()]
df[GENO_COL] = pd.Categorical(df[GENO_COL], categories=geno_present, ordered=True)
geno_palette = {g: GENO_PALETTE[g] for g in geno_present}
n_geno = len(geno_present)

subj = compute_subject_metrics(df)
subj[GENO_COL] = pd.Categorical(subj[GENO_COL], categories=geno_present, ordered=True)

treat_label = TREAT_COL.replace("_", " ").title()
geno_label = GENO_COL.replace("_", " ").title()

# ── Figure dispatch ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
metrics_to_plot = [
    ("plaque_count", "Total plaque count", "linear"),
    ("total_vol_ml", "Total plaque burden as volume (mL)", "linear"),
    ("median_vol_ml", "Median plaque size (mL)", "linear"),
    ("median_diam_um", "Median plaque equiv. diameter (\u00b5m)", "linear"),
]
for ax, (col, label, yscale) in zip(axes.flat, metrics_to_plot):
    if n_geno > 1:
        boxstrip(ax, subj, x=TREAT_COL, y=col, hue=GENO_COL,
                 order=TREAT_ORDER, hue_order=geno_present, palette=geno_palette,
                 ylabel=label, yscale=yscale)
    else:
        sns.boxplot(data=subj, x=TREAT_COL, y=col, order=TREAT_ORDER,
                    palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                    ax=ax)
        sns.stripplot(data=subj, x=TREAT_COL, y=col, order=TREAT_ORDER,
                      palette=TREAT_PALETTE, alpha=0.7, size=6, jitter=True, ax=ax)
        ax.set_ylabel(label)
    ax.set_xlabel("")
fig.suptitle(
    f"Plaque burden metrics by {treat_label} and {geno_label} \u2014 {cohort}",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(output_figs["boxplots_treatment_genotype"], dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (hue_col, hue_order, palette) in zip(
    axes,
    [
        (TREAT_COL, TREAT_ORDER, TREAT_PALETTE),
        (GENO_COL, geno_present, geno_palette),
    ],
):
    sns.violinplot(
        data=df, x=hue_col, y="equiv_diam_um", order=hue_order, palette=palette,
        inner="quartile", cut=0, linewidth=1.2, ax=ax,
    )
    ax.set_ylabel("Equivalent diameter (\u00b5m)")
    ax.set_xlabel("")
    ax.set_ylim(0, 80)
axes[0].set_title(f"Plaque size by {treat_label}")
axes[1].set_title(f"Plaque size by {geno_label}")
fig.suptitle(
    f"Per-plaque equivalent diameter distributions \u2014 {cohort}", fontsize=13
)
plt.tight_layout()
plt.savefig(output_figs["violin_plaque_size"], dpi=150, bbox_inches="tight")
plt.close(fig)

int_metrics = [
    ("plaque_count", "Plaque count", "log"),
    ("total_vol_ml", "Total plaque volume (mL)", "log"),
    ("mean_diam_um", "Mean equiv. diameter (\u00b5m)", "linear"),
]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, (col, label, yscale) in zip(axes, int_metrics):
    interaction_plot_ax(ax, subj, col, label, yscale)
fig.suptitle(
    f"Interaction plots: {treat_label} \u00d7 {geno_label} (mean \u00b1 SEM) \u2014 {cohort}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["interaction_plots"], dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(n_geno, 3, figsize=(14, 4.5 * n_geno), squeeze=False)
for row_idx, geno in enumerate(geno_present):
    g = subj[subj[GENO_COL] == geno]
    for ax, (col, label, yscale) in zip(
        axes[row_idx],
        [
            ("plaque_count", "Plaque count", "linear"),
            ("total_vol_ml", "Total plaque volume (mL)", "linear"),
            ("median_diam_um", "Median diameter (\u00b5m)", "linear"),
        ],
    ):
        sns.boxplot(data=g, x=TREAT_COL, y=col, order=TREAT_ORDER,
                    palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                    ax=ax)
        sns.stripplot(data=g, x=TREAT_COL, y=col, order=TREAT_ORDER,
                      palette=TREAT_PALETTE, alpha=0.75, size=7, jitter=True, ax=ax)
        if yscale != "linear":
            ax.set_yscale(yscale)
        ax.set_ylabel(label)
        ax.set_xlabel("")
        ax.set_title(geno)
fig.suptitle(
    f"{treat_label} effect stratified by {geno_label} \u2014 {cohort}", fontsize=13
)
plt.tight_layout()
plt.savefig(output_figs["stratified_by_genotype"], dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(
    1, n_geno, figsize=(6 * n_geno, 5), sharey=True, squeeze=False
)
for ax, geno in zip(axes[0], geno_present):
    g = df[df[GENO_COL] == geno]
    sns.ecdfplot(
        data=g, x="equiv_diam_um", hue=TREAT_COL,
        hue_order=TREAT_ORDER, palette=TREAT_PALETTE, ax=ax,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Equivalent diameter (\u00b5m)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(geno)
fig.suptitle(
    f"ECDF of plaque size by {treat_label}, stratified by {geno_label} \u2014 {cohort}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["ecdf_size_by_genotype"], dpi=150, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(
    1, n_geno, figsize=(6 * n_geno, 5), sharey=True, squeeze=False
)
for ax, geno in zip(axes[0], geno_present):
    g = df[df[GENO_COL] == geno]
    sns.histplot(
        data=g, x="equiv_diam_um", hue=TREAT_COL,
        kde=True, kde_kws={"clip": (14, 100), "cut": 0, "bw_adjust": 0.1},
        hue_order=TREAT_ORDER, palette=TREAT_PALETTE, ax=ax,
    )
    ax.set_xlabel("Equivalent diameter (\u00b5m)")
    ax.set_ylabel("Count")
    ax.set_title(geno)
fig.suptitle(
    f"Distribution of plaque size by {treat_label}, stratified by {geno_label}"
    f" \u2014 {cohort}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["size_distribution_by_genotype"], dpi=150, bbox_inches="tight")
plt.close(fig)
