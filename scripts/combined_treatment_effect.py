"""Combined multipanel treatment-effect strip+box plot across batches and genotypes.

Reads  : data_{cohort}.parquet for each cohort (batch)
Writes :
  figures/fig_combined_treatment_effect.png
    Rows = burden metrics (plaque count, total volume, median volume, median diameter)
    Columns = batch × genotype combinations present in the data
    Same y-axis across all columns within each row (sharey per metric)
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Snakemake integration ────────────────────────────────────────────────────
if "snakemake" in dir():
    input_parquets = {k: str(v) for k, v in snakemake.input.items()}  # noqa: F821
    output_fig = str(snakemake.output[0])  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquets", nargs="+", metavar="COHORT=PATH",
        help="One or more cohort=path pairs, e.g. early=data_early.parquet",
        required=True,
    )
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()
    input_parquets = dict(pair.split("=", 1) for pair in args.parquets)
    output_fig = args.output

# ── Constants ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
TREAT_ORDER = ["PBS", "Lecanemab"]
TREAT_PALETTE = {"PBS": "#4C72B0", "Lecanemab": "#DD8452"}
GENO_ORDER = ["ApoE3", "ApoE4"]
VOL_THRESH_ML = 1e-4

METRICS = [
    ("plaque_count",   "Total plaque count",                  "linear"),
    ("total_vol_ml",   "Total plaque burden (mL)",            "linear"),
    ("median_vol_ml",  "Median plaque size (mL)",             "linear"),
    ("median_diam_um", "Median equiv. diameter (\u00b5m)",    "linear"),
]


# ── Helper functions ─────────────────────────────────────────────────────────
def compute_subject_metrics(plaque_df):
    """Aggregate plaque-level data to one row per subject."""
    grp = plaque_df.groupby(
        ["subject", "treatment", "genotype", "sex", "batch"], observed=True
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


def boxstrip(ax, data, x, y, order=None, palette=None, ylabel=None, yscale="linear"):
    """Paired boxplot + stripplot on a single axis."""
    sns.boxplot(
        data=data, x=x, y=y,
        order=order, palette=palette,
        fill=False, linewidth=1.2, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=data, x=x, y=y,
        order=order, palette=palette,
        dodge=False, alpha=0.7, size=6, jitter=True, ax=ax,
    )
    if yscale != "linear":
        ax.set_yscale(yscale)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    # Remove per-panel legend (treatment axis labels are sufficient)
    if ax.get_legend():
        ax.get_legend().remove()


# ── Load and combine data ────────────────────────────────────────────────────
frames = []
for cohort, path in sorted(input_parquets.items()):
    df_c = pd.read_parquet(path)
    df_c = df_c.loc[df_c["plaque_vol_ml"] <= VOL_THRESH_ML].copy()
    df_c["batch"] = cohort
    frames.append(df_c)

df_all = pd.concat(frames, ignore_index=True)
df_all["treatment"] = pd.Categorical(
    df_all["treatment"], categories=TREAT_ORDER, ordered=True
)
geno_present = [g for g in GENO_ORDER if (df_all["genotype"] == g).any()]
df_all["genotype"] = pd.Categorical(
    df_all["genotype"], categories=geno_present, ordered=True
)

subj = compute_subject_metrics(df_all)
subj["treatment"] = pd.Categorical(
    subj["treatment"], categories=TREAT_ORDER, ordered=True
)
subj["genotype"] = pd.Categorical(
    subj["genotype"], categories=geno_present, ordered=True
)

# Build ordered list of (batch, genotype) columns present in the data
batch_order = sorted(input_parquets.keys())
columns = [
    (batch, geno)
    for batch in batch_order
    for geno in geno_present
    if not subj[(subj["batch"] == batch) & (subj["genotype"] == geno)].empty
]
n_cols = len(columns)
n_rows = len(METRICS)

# ── Build figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4 * n_cols, 4 * n_rows),
    sharey="row",
    squeeze=False,
)

for row_idx, (metric, ylabel, yscale) in enumerate(METRICS):
    for col_idx, (batch, geno) in enumerate(columns):
        ax = axes[row_idx][col_idx]
        subset = subj[(subj["batch"] == batch) & (subj["genotype"] == geno)]
        boxstrip(
            ax, subset,
            x="treatment", y=metric,
            order=TREAT_ORDER, palette=TREAT_PALETTE,
            ylabel=ylabel if col_idx == 0 else None,
            yscale=yscale,
        )
        if row_idx == 0:
            ax.set_title(f"{batch}\n{geno}", fontsize=11)

fig.suptitle(
    "Treatment effect (PBS vs Lecanemab) by batch and genotype",
    fontsize=14,
    y=1.01,
)
plt.tight_layout()
plt.savefig(output_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
