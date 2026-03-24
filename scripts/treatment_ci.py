"""Confidence-interval analysis for subject-level treatment effects.

Appropriate for small-N (2-3 subjects per group) experiments where
p-value-based inference is underpowered.  Every plot shows all individual
data points so readers can judge the raw data directly.

Reads  : data_{cohort}.parquet
Writes:
  fig_{cohort}_treatment_ci_plots.png
      Mean ± 95 % CI point plots (t-distribution) with individual data
      overlay, one panel per metric × genotype combination.
  fig_{cohort}_treatment_ci_forest.png
      Forest plot of log₂ fold-change (Lecanemab / PBS) with 95 %
      parametric-bootstrap CI, one row per metric × genotype.
  {cohort}_treatment_ci_table.csv
      Per-metric × genotype summary: group means, 95 % CI bounds,
      log₂ FC, and bootstrap CI bounds.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Snakemake integration ────────────────────────────────────────────────────
if "snakemake" in dir():
    input_parquet = str(snakemake.input.parquet)  # noqa: F821
    output_figs = dict(snakemake.output)  # noqa: F821
    cohort = snakemake.wildcards.cohort  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()
    input_parquet = args.parquet
    cohort = args.cohort
    output_figs = {
        "treatment_ci_plots": f"{args.output_dir}/fig_{cohort}_treatment_ci_plots.png",
        "treatment_ci_forest": f"{args.output_dir}/fig_{cohort}_treatment_ci_forest.png",
        "treatment_ci_table": f"{args.output_dir}/{cohort}_treatment_ci_table.csv",
    }

# ── Constants ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
TREAT_ORDER = ["PBS", "Lecanemab"]
TREAT_PALETTE = {"PBS": "#4C72B0", "Lecanemab": "#DD8452"}
GENO_ORDER = ["ApoE3", "ApoE4"]
GENO_PALETTE = {"ApoE3": "#55A868", "ApoE4": "#C44E52"}
VOL_THRESH_ML = 1e-4
CI_LEVEL = 0.95          # confidence level
N_BOOT = 5_000          # bootstrap iterations for log2FC CI
RNG_SEED = 42

METRICS = [
    ("plaque_count", "Total plaque count"),
    ("total_vol_ml", "Total plaque burden (mL)"),
    ("median_vol_ml", "Median plaque size (mL)"),
    ("median_diam_um", "Median plaque diameter (\u00b5m)"),
]

PSEUDOCOUNT = 1e-9   # added to numerator and denominator before log ratios to avoid log(0)


# ── Helper functions ─────────────────────────────────────────────────────────
def t_ci(values, level=CI_LEVEL):
    """Return (mean, lo, hi) using the t-distribution (exact for any n ≥ 2)."""
    n = len(values)
    if n < 2:
        m = float(np.mean(values)) if n == 1 else np.nan
        return m, np.nan, np.nan
    m = float(np.mean(values))
    se = float(stats.sem(values))
    lo, hi = stats.t.interval(level, df=n - 1, loc=m, scale=se)
    return m, float(lo), float(hi)


def bootstrap_log2fc_ci(pbs_vals, lec_vals, n_boot=N_BOOT, level=CI_LEVEL,
                        rng=None):
    """Parametric bootstrap CI for log₂(mean_lec / mean_pbs).

    Resamples each group independently (with replacement) and computes
    the log₂ ratio of resampled means.  Returns (point_estimate, lo, hi).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n_pbs = len(pbs_vals)
    n_lec = len(lec_vals)
    if n_pbs < 1 or n_lec < 1:
        return np.nan, np.nan, np.nan

    mean_pbs = float(np.mean(pbs_vals))
    mean_lec = float(np.mean(lec_vals))
    point = np.log2((mean_lec + PSEUDOCOUNT) / (mean_pbs + PSEUDOCOUNT))

    if n_pbs == 1 and n_lec == 1:
        # Cannot resample a single observation; return point estimate only
        return point, np.nan, np.nan

    boot_ratios = np.empty(n_boot)
    for i in range(n_boot):
        b_pbs = rng.choice(pbs_vals, size=n_pbs, replace=True)
        b_lec = rng.choice(lec_vals, size=n_lec, replace=True)
        boot_ratios[i] = np.log2(
            (np.mean(b_lec) + PSEUDOCOUNT) / (np.mean(b_pbs) + PSEUDOCOUNT)
        )

    alpha = 1.0 - level
    lo = float(np.nanpercentile(boot_ratios, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot_ratios, 100 * (1 - alpha / 2)))
    return point, lo, hi


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
    return metrics


# ── Load and prepare data ────────────────────────────────────────────────────
df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()
df["treatment"] = pd.Categorical(df["treatment"], categories=TREAT_ORDER, ordered=True)
geno_present = [g for g in GENO_ORDER if (df["genotype"] == g).any()]
df["genotype"] = pd.Categorical(df["genotype"], categories=geno_present, ordered=True)
n_geno = len(geno_present)

subj = compute_subject_metrics(df)
subj["genotype"] = pd.Categorical(subj["genotype"], categories=geno_present, ordered=True)
subj["treatment"] = pd.Categorical(subj["treatment"], categories=TREAT_ORDER, ordered=True)

rng = np.random.default_rng(RNG_SEED)

# ── Build CI summary table ────────────────────────────────────────────────────
ci_rows = []
for geno in geno_present:
    g = subj[subj["genotype"] == geno]
    for col, label in METRICS:
        pbs_vals = g.loc[g["treatment"] == "PBS", col].dropna().values
        lec_vals = g.loc[g["treatment"] == "Lecanemab", col].dropna().values

        mean_pbs, pbs_lo, pbs_hi = t_ci(pbs_vals)
        mean_lec, lec_lo, lec_hi = t_ci(lec_vals)
        log2fc, fc_lo, fc_hi = bootstrap_log2fc_ci(pbs_vals, lec_vals, rng=rng)

        ci_rows.append({
            "genotype": geno,
            "metric": col,
            "metric_label": label,
            "n_PBS": len(pbs_vals),
            "n_Lecanemab": len(lec_vals),
            "mean_PBS": mean_pbs,
            "ci_lo_PBS": pbs_lo,
            "ci_hi_PBS": pbs_hi,
            "mean_Lecanemab": mean_lec,
            "ci_lo_Lecanemab": lec_lo,
            "ci_hi_Lecanemab": lec_hi,
            "log2fc_Lec_over_PBS": log2fc,
            "ci_lo_log2fc": fc_lo,
            "ci_hi_log2fc": fc_hi,
            "ci_excludes_zero": (
                bool(fc_lo > 0 or fc_hi < 0)
                if (np.isfinite(fc_lo) and np.isfinite(fc_hi))
                else False
            ),
        })

ci_df = pd.DataFrame(ci_rows)
ci_df.to_csv(output_figs["treatment_ci_table"], index=False)

# ── Figure 1: Mean ± 95 % CI point plots ────────────────────────────────────
# Layout: n_geno rows × n_metrics cols.
# Each panel shows PBS and Lecanemab with mean ± CI error bars and
# individual data points (always visible given the small N).
n_metrics = len(METRICS)
fig, axes = plt.subplots(
    n_geno, n_metrics,
    figsize=(4.5 * n_metrics, 5.0 * n_geno),
    squeeze=False,
)

x_pos = {t: i for i, t in enumerate(TREAT_ORDER)}
x_coords = [x_pos[t] for t in TREAT_ORDER]

for row_idx, geno in enumerate(geno_present):
    g = subj[subj["genotype"] == geno]
    geno_ci = ci_df[ci_df["genotype"] == geno]

    for col_idx, (col, label) in enumerate(METRICS):
        ax = axes[row_idx, col_idx]
        row_ci = geno_ci[geno_ci["metric"] == col].iloc[0]

        for treat in TREAT_ORDER:
            vals = g.loc[g["treatment"] == treat, col].dropna().values
            xc = x_pos[treat]
            color = TREAT_PALETTE[treat]

            # Individual points
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                xc + jitter, vals,
                color=color, alpha=0.7, s=60, zorder=3,
                edgecolors="white", linewidths=0.5,
            )

            # Mean ± CI
            mean_key = f"mean_{treat}"
            lo_key = f"ci_lo_{treat}"
            hi_key = f"ci_hi_{treat}"
            m = row_ci[mean_key]
            lo = row_ci[lo_key]
            hi = row_ci[hi_key]
            if np.isfinite(m):
                ax.plot([xc - 0.18, xc + 0.18], [m, m],
                        color=color, lw=2.5, zorder=4, solid_capstyle="round")
            if np.isfinite(lo) and np.isfinite(hi):
                ax.plot([xc, xc], [lo, hi],
                        color=color, lw=1.8, zorder=4)
                ax.plot([xc - 0.09, xc + 0.09], [lo, lo],
                        color=color, lw=1.8, zorder=4)
                ax.plot([xc - 0.09, xc + 0.09], [hi, hi],
                        color=color, lw=1.8, zorder=4)

        ax.set_xticks(x_coords)
        ax.set_xticklabels(TREAT_ORDER)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(label if col_idx == 0 else "")
        ax.set_xlabel("")
        if row_idx == 0:
            ax.set_title(label, fontsize=11, pad=8)

        n_pbs = int(row_ci["n_PBS"])
        n_lec = int(row_ci["n_Lecanemab"])
        ax.text(
            0.98, 0.02,
            f"n={n_pbs} PBS / {n_lec} Lec",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray",
        )

    axes[row_idx, 0].set_ylabel(
        f"{geno}\n{axes[row_idx, 0].get_ylabel()}", fontsize=11
    )

fig.suptitle(
    f"Treatment effect (PBS vs Lecanemab) by genotype \u2014 {cohort}\n"
    f"Bars = mean \u00b1 {int(CI_LEVEL * 100)}\u2009% CI (t-distribution); "
    "dots = individual subjects",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig(output_figs["treatment_ci_plots"], dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Log₂ FC forest plot ────────────────────────────────────────────
# One row per metric × genotype combination.
forest_rows = ci_df[ci_df["log2fc_Lec_over_PBS"].notna()].copy()
n_rows = len(forest_rows)

if n_rows == 0:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, "Insufficient data for forest plot",
            ha="center", va="center", fontsize=13, color="gray",
            transform=ax.transAxes)
    ax.axis("off")
else:
    fig_h = max(3.5, 0.6 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))

    y_labels = [
        f"{r.metric_label}\n({r.genotype})" for _, r in forest_rows.iterrows()
    ]
    y_pos = np.arange(n_rows)

    for i, (_, row) in enumerate(forest_rows.iterrows()):
        has_ci = np.isfinite(row["ci_lo_log2fc"]) and np.isfinite(row["ci_hi_log2fc"])
        ci_excl = bool(row["ci_excludes_zero"]) if has_ci else False
        color = "#DD8452" if ci_excl else "#4C72B0"
        marker_size = 90

        # CI line
        if has_ci:
            ax.plot(
                [row["ci_lo_log2fc"], row["ci_hi_log2fc"]],
                [i, i],
                color=color, lw=2.0, zorder=2,
            )
            # CI caps
            ax.plot([row["ci_lo_log2fc"]] * 2, [i - 0.1, i + 0.1],
                    color=color, lw=2.0, zorder=2)
            ax.plot([row["ci_hi_log2fc"]] * 2, [i - 0.1, i + 0.1],
                    color=color, lw=2.0, zorder=2)

        # Point estimate
        ax.scatter(
            [row["log2fc_Lec_over_PBS"]], [i],
            color=color, s=marker_size, zorder=3, edgecolors="white",
            linewidths=0.8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("log\u2082 fold-change (Lecanemab / PBS)", fontsize=11)
    ax.set_title(
        f"Treatment effect forest plot \u2014 {cohort}\n"
        f"{int(CI_LEVEL * 100)}\u2009% bootstrap CI  "
        "\u25cf orange = CI excludes 0  \u25cf blue = CI includes 0",
        fontsize=12,
    )
    ax.grid(axis="x", alpha=0.4)

    # Annotate n per group after axes limits are finalised
    x_right = ax.get_xlim()[1]
    for i, (_, row) in enumerate(forest_rows.iterrows()):
        n_txt = f"  n={int(row['n_PBS'])}/{int(row['n_Lecanemab'])}"
        ax.text(x_right, i, n_txt, va="center", ha="left",
                fontsize=8, color="gray", clip_on=False)

plt.tight_layout()
plt.savefig(output_figs["treatment_ci_forest"], dpi=150, bbox_inches="tight")
plt.close(fig)
