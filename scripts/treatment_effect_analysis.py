"""Treatment-effect and genotype-modulation analysis figures.

Reads  : data_{cohort}.parquet
Writes : fig_{cohort}_{fig}.png

Figure types (passed via Snakemake wildcard `fig` or --fig CLI arg):
  boxplots_treatment_genotype   2×2 boxplots of burden metrics
  violin_plaque_size            violin plots of per-plaque diameter
  interaction_plots             mean±SEM interaction plots (treatment × genotype)
  stratified_by_genotype        treatment boxplots stratified by genotype
  ecdf_size_by_genotype         ECDFs of plaque size per genotype
  size_distribution_by_genotype histograms of plaque size per genotype
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

# ── Snakemake integration ────────────────────────────────────────────────────
if "snakemake" in dir():
    input_parquet = str(snakemake.input.parquet)  # noqa: F821
    output_fig = str(snakemake.output[0])  # noqa: F821
    cohort = snakemake.wildcards.cohort  # noqa: F821
    fig_type = snakemake.wildcards.fig  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--fig", required=True)
    args = parser.parse_args()
    input_parquet = args.parquet
    output_fig = args.output
    cohort = args.cohort
    fig_type = args.fig

# ── Constants ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
TREAT_ORDER = ["PBS", "Lecanemab"]
TREAT_PALETTE = {"PBS": "#4C72B0", "Lecanemab": "#DD8452"}
GENO_ORDER = ["ApoE3", "ApoE4"]
GENO_PALETTE = {"ApoE3": "#55A868", "ApoE4": "#C44E52"}
VOL_THRESH_ML = 1e-4


# ── Helper functions ─────────────────────────────────────────────────────────
def compute_subject_metrics(plaque_df):
    """Aggregate plaque-level data to one row per subject."""
    grp = plaque_df.groupby(
        ["subject", "treatment", "genotype", "sex"], observed=True
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
    """Two-way OLS ANOVA (or one-way when only one genotype present)."""
    n_geno = data["genotype"].nunique()
    if n_geno > 1:
        formula = f"{outcome} ~ C(treatment) * C(genotype)"
    else:
        formula = f"{outcome} ~ C(treatment)"
    model = ols(formula, data=data).fit()
    return anova_lm(model, typ=2)


def interaction_plot_ax(ax, data, outcome, ylabel, yscale="linear"):
    """Mean ± SEM interaction plot (treatment on x, genotype as lines)."""
    agg = (
        data.groupby(["treatment", "genotype"], observed=True)[outcome]
        .agg(["mean", "sem"])
        .reset_index()
    )
    for geno, grp in agg.groupby("genotype", observed=True):
        color = GENO_PALETTE[geno]
        ax.plot(grp["treatment"].astype(str), grp["mean"],
                marker="o", linewidth=2, color=color, label=geno)
        ax.errorbar(grp["treatment"].astype(str), grp["mean"],
                    yerr=grp["sem"], fmt="none", color=color, capsize=4)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Treatment")
    if yscale != "linear":
        ax.set_yscale(yscale)
    ax.legend(title="Genotype", fontsize=9)


def tukey_posthoc(data, outcome):
    """Run Tukey HSD across all treatment × genotype combinations."""
    d = data.copy()
    d["group"] = d["treatment"].astype(str) + " | " + d["genotype"].astype(str)
    return pairwise_tukeyhsd(d[outcome], d["group"], alpha=0.05)


# ── Load and prepare data ────────────────────────────────────────────────────
df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()
df["treatment"] = pd.Categorical(df["treatment"], categories=TREAT_ORDER, ordered=True)
geno_present = [g for g in GENO_ORDER if (df["genotype"] == g).any()]
df["genotype"] = pd.Categorical(df["genotype"], categories=geno_present, ordered=True)
geno_palette = {g: GENO_PALETTE[g] for g in geno_present}
n_geno = len(geno_present)

subj = compute_subject_metrics(df)
subj["genotype"] = pd.Categorical(subj["genotype"], categories=geno_present, ordered=True)

# ── Figure dispatch ───────────────────────────────────────────────────────────
if fig_type == "boxplots_treatment_genotype":
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics_to_plot = [
        ("plaque_count", "Total plaque count", "linear"),
        ("total_vol_ml", "Total plaque burden as volume (mL)", "linear"),
        ("median_vol_ml", "Median plaque size (mL)", "linear"),
        ("median_diam_um", "Median plaque equiv. diameter (\u00b5m)", "linear"),
    ]
    for ax, (col, label, yscale) in zip(axes.flat, metrics_to_plot):
        if n_geno > 1:
            boxstrip(ax, subj, x="treatment", y=col, hue="genotype",
                     order=TREAT_ORDER, hue_order=geno_present, palette=geno_palette,
                     ylabel=label, yscale=yscale)
        else:
            sns.boxplot(data=subj, x="treatment", y=col, order=TREAT_ORDER,
                        palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                        ax=ax)
            sns.stripplot(data=subj, x="treatment", y=col, order=TREAT_ORDER,
                          palette=TREAT_PALETTE, alpha=0.7, size=6, jitter=True, ax=ax)
            ax.set_ylabel(label)
        ax.set_xlabel("")
    fig.suptitle(
        f"Plaque burden metrics by treatment and genotype \u2014 {cohort}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

elif fig_type == "violin_plaque_size":
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (hue_col, hue_order, palette) in zip(
        axes,
        [
            ("treatment", TREAT_ORDER, TREAT_PALETTE),
            ("genotype", geno_present, geno_palette),
        ],
    ):
        sns.violinplot(
            data=df, x=hue_col, y="equiv_diam_um", order=hue_order, palette=palette,
            inner="quartile", cut=0, linewidth=1.2, ax=ax,
        )
        ax.set_ylabel("Equivalent diameter (\u00b5m)")
        ax.set_xlabel("")
        ax.set_ylim(0, 80)
    axes[0].set_title("Plaque size by treatment")
    axes[1].set_title("Plaque size by genotype")
    fig.suptitle(
        f"Per-plaque equivalent diameter distributions \u2014 {cohort}", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

elif fig_type == "interaction_plots":
    int_metrics = [
        ("plaque_count", "Plaque count", "log"),
        ("total_vol_ml", "Total plaque volume (mL)", "log"),
        ("mean_diam_um", "Mean equiv. diameter (\u00b5m)", "linear"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (col, label, yscale) in zip(axes, int_metrics):
        interaction_plot_ax(ax, subj, col, label, yscale)
    fig.suptitle(
        f"Interaction plots: treatment \u00d7 genotype (mean \u00b1 SEM) \u2014 {cohort}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

elif fig_type == "stratified_by_genotype":
    fig, axes = plt.subplots(n_geno, 3, figsize=(14, 4.5 * n_geno), squeeze=False)
    for row_idx, geno in enumerate(geno_present):
        g = subj[subj["genotype"] == geno]
        for ax, (col, label, yscale) in zip(
            axes[row_idx],
            [
                ("plaque_count", "Plaque count", "linear"),
                ("total_vol_ml", "Total plaque volume (mL)", "linear"),
                ("median_diam_um", "Median diameter (\u00b5m)", "linear"),
            ],
        ):
            sns.boxplot(data=g, x="treatment", y=col, order=TREAT_ORDER,
                        palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                        ax=ax)
            sns.stripplot(data=g, x="treatment", y=col, order=TREAT_ORDER,
                          palette=TREAT_PALETTE, alpha=0.75, size=7, jitter=True, ax=ax)
            if yscale != "linear":
                ax.set_yscale(yscale)
            ax.set_ylabel(label)
            ax.set_xlabel("")
            ax.set_title(geno)
    fig.suptitle(
        f"Treatment effect stratified by genotype \u2014 {cohort}", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

elif fig_type == "ecdf_size_by_genotype":
    fig, axes = plt.subplots(
        1, n_geno, figsize=(6 * n_geno, 5), sharey=True, squeeze=False
    )
    for ax, geno in zip(axes[0], geno_present):
        g = df[df["genotype"] == geno]
        sns.ecdfplot(
            data=g, x="equiv_diam_um", hue="treatment",
            hue_order=TREAT_ORDER, palette=TREAT_PALETTE, ax=ax,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Equivalent diameter (\u00b5m)")
        ax.set_ylabel("Cumulative fraction")
        ax.set_title(geno)
    fig.suptitle(
        f"ECDF of plaque size by treatment, stratified by genotype \u2014 {cohort}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

elif fig_type == "size_distribution_by_genotype":
    fig, axes = plt.subplots(
        1, n_geno, figsize=(6 * n_geno, 5), sharey=True, squeeze=False
    )
    for ax, geno in zip(axes[0], geno_present):
        g = df[df["genotype"] == geno]
        sns.histplot(
            data=g, x="equiv_diam_um", hue="treatment",
            kde=True, kde_kws={"clip": (14, 100), "cut": 0, "bw_adjust": 0.1},
            hue_order=TREAT_ORDER, palette=TREAT_PALETTE, ax=ax,
        )
        ax.set_xlabel("Equivalent diameter (\u00b5m)")
        ax.set_ylabel("Count")
        ax.set_title(geno)
    fig.suptitle(
        f"Distribution of plaque size by treatment, stratified by genotype"
        f" \u2014 {cohort}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

else:
    raise ValueError(f"Unknown fig_type: '{fig_type}'")
