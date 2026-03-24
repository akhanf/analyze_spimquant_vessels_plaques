"""Statistical testing for treatment effects with significance annotations.

Reads  : data_{cohort}.parquet
Writes:
  fig_{cohort}_treatment_stats_boxplots.png  Boxplots with significance brackets
                                             for each metric × genotype combination
  {cohort}_treatment_anova_table.csv         Two-way ANOVA results per metric
  {cohort}_treatment_tukey_results.csv       Tukey HSD (or Welch t-test) results
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
        "treatment_stats_boxplots": f"{args.output_dir}/fig_{cohort}_treatment_stats_boxplots.png",
        "treatment_anova_table": f"{args.output_dir}/{cohort}_treatment_anova_table.csv",
        "treatment_tukey_results": f"{args.output_dir}/{cohort}_treatment_tukey_results.csv",
    }

# ── Constants ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
TREAT_ORDER = ["PBS", "Lecanemab"]
TREAT_PALETTE = {"PBS": "#4C72B0", "Lecanemab": "#DD8452"}
GENO_ORDER = ["ApoE3", "ApoE4"]
GENO_PALETTE = {"ApoE3": "#55A868", "ApoE4": "#C44E52"}
VOL_THRESH_ML = 1e-4
FDR_ALPHA = 0.05

METRICS = [
    ("plaque_count", "Total plaque count"),
    ("total_vol_ml", "Total plaque burden (mL)"),
    ("median_vol_ml", "Median plaque size (mL)"),
    ("median_diam_um", "Median plaque diameter (\u00b5m)"),
]


# ── Helper functions ─────────────────────────────────────────────────────────
def pval_to_stars(p):
    """Convert a p-value to a significance string."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "ns"
    if p <= 0.0001:
        return "****"
    if p <= 0.001:
        return "***"
    if p <= 0.01:
        return "**"
    if p <= 0.05:
        return "*"
    return "ns"


def add_significance_bracket(ax, x1, x2, y, h, stars, color="black", lw=1.2,
                              fontsize=11):
    """Draw a significance bracket between x positions x1 and x2."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=lw, color=color, clip_on=False)
    ax.text((x1 + x2) / 2, y + h * 1.05, stars,
            ha="center", va="bottom", fontsize=fontsize,
            color=color, clip_on=False)


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


def run_twoway_anova(data, outcome):
    """Two-way OLS ANOVA (or one-way when only one genotype present)."""
    n_geno = data["genotype"].nunique()
    if n_geno > 1:
        formula = f"{outcome} ~ C(treatment) * C(genotype)"
    else:
        formula = f"{outcome} ~ C(treatment)"
    model = ols(formula, data=data).fit()
    return anova_lm(model, typ=2)


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

# ── Two-way ANOVA table ───────────────────────────────────────────────────────
anova_rows = []
for col, label in METRICS:
    try:
        tbl = run_twoway_anova(subj, col)
        for term, row in tbl.iterrows():
            anova_rows.append({
                "metric": col,
                "metric_label": label,
                "term": term,
                "df": row.get("df", np.nan),
                "sum_sq": row.get("sum_sq", np.nan),
                "F": row.get("F", np.nan),
                "PR(>F)": row.get("PR(>F)", np.nan),
                "significant": (
                    row.get("PR(>F)", 1.0) <= FDR_ALPHA
                    if pd.notna(row.get("PR(>F)")) else False
                ),
                "stars": pval_to_stars(row.get("PR(>F)")),
            })
    except Exception:
        pass
anova_df = pd.DataFrame(anova_rows)
anova_df.to_csv(output_figs["treatment_anova_table"], index=False)

# ── Pairwise comparisons (Tukey HSD or Welch t-test) ─────────────────────────
pairwise_rows = []
if n_geno > 1:
    # Tukey HSD across all treatment × genotype combinations
    for col, label in METRICS:
        try:
            d = subj.copy()
            d["group"] = d["treatment"].astype(str) + " | " + d["genotype"].astype(str)
            tukey = pairwise_tukeyhsd(d[col], d["group"], alpha=FDR_ALPHA)
            result_data = tukey._results_table.data
            hdr = [str(c) for c in result_data[0]]
            for data_row in result_data[1:]:
                row_dict = dict(zip(hdr, data_row))
                row_dict["metric"] = col
                row_dict["metric_label"] = label
                row_dict["stars"] = pval_to_stars(float(row_dict.get("p-adj", 1.0)))
                pairwise_rows.append(row_dict)
        except Exception:
            pass
else:
    # Welch's t-test between PBS and Lecanemab
    for col, label in METRICS:
        pbs_vals = subj.loc[subj["treatment"] == "PBS", col].dropna().values
        lec_vals = subj.loc[subj["treatment"] == "Lecanemab", col].dropna().values
        if len(pbs_vals) >= 2 and len(lec_vals) >= 2:
            t_stat, p_val = stats.ttest_ind(lec_vals, pbs_vals, equal_var=False)
            pairwise_rows.append({
                "metric": col,
                "metric_label": label,
                "group1": "PBS",
                "group2": "Lecanemab",
                "meandiff": float(np.mean(lec_vals) - np.mean(pbs_vals)),
                "p-adj": float(p_val),
                "reject": bool(p_val <= FDR_ALPHA),
                "stars": pval_to_stars(p_val),
            })

pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_df.to_csv(output_figs["treatment_tukey_results"], index=False)


# ── Boxplots with significance brackets ──────────────────────────────────────
# Layout: n_geno rows (one per genotype) × n_metrics columns.
# Each subplot shows PBS vs Lecanemab for that genotype with a significance bracket.
n_metrics = len(METRICS)
fig, axes = plt.subplots(
    n_geno, n_metrics,
    figsize=(4.5 * n_metrics, 5.5 * n_geno),
    squeeze=False,
)

for row_idx, geno in enumerate(geno_present):
    g = subj[subj["genotype"] == geno]
    for col_idx, (col, label) in enumerate(METRICS):
        ax = axes[row_idx, col_idx]

        sns.boxplot(
            data=g, x="treatment", y=col,
            order=TREAT_ORDER, palette=TREAT_PALETTE,
            fill=False, linewidth=1.2, fliersize=0, ax=ax,
        )
        sns.stripplot(
            data=g, x="treatment", y=col,
            order=TREAT_ORDER, palette=TREAT_PALETTE,
            alpha=0.75, size=7, jitter=True, ax=ax,
        )

        # Compute Welch's t-test for this genotype
        pbs_vals = g.loc[g["treatment"] == "PBS", col].dropna().values
        lec_vals = g.loc[g["treatment"] == "Lecanemab", col].dropna().values
        stars = "ns"
        p_val = np.nan
        if len(pbs_vals) >= 2 and len(lec_vals) >= 2:
            _, p_val = stats.ttest_ind(lec_vals, pbs_vals, equal_var=False)
            stars = pval_to_stars(p_val)

        # Draw significance bracket between PBS (x=0) and Lecanemab (x=1)
        all_vals = g[col].dropna().values
        if len(all_vals) > 0:
            y_max = float(np.max(all_vals))
            y_min = float(np.min(all_vals))
            y_range = y_max - y_min if y_max > y_min else abs(y_max) * 0.2 + 1e-10
            bracket_y = y_max + y_range * 0.06
            bracket_h = y_range * 0.05
            add_significance_bracket(
                ax, 0, 1, bracket_y, bracket_h, stars,
                color="black", fontsize=12,
            )
            ax.set_ylim(top=bracket_y + bracket_h * 3.5)

        ax.set_ylabel(label if col_idx == 0 else "")
        ax.set_xlabel("")
        if row_idx == 0:
            ax.set_title(label, fontsize=11, pad=8)
        ax.text(
            0.98, 0.02,
            f"n={len(pbs_vals)} PBS / {len(lec_vals)} Lec",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray",
        )

    axes[row_idx, 0].set_ylabel(
        f"{geno}\n{axes[row_idx, 0].get_ylabel()}", fontsize=11
    )

fig.suptitle(
    f"Treatment effect (PBS vs Lecanemab) by genotype \u2014 {cohort}\n"
    "* p\u22640.05  ** p\u22640.01  *** p\u22640.001  **** p\u22640.0001  ns\u2009=\u2009not significant",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig(
    output_figs["treatment_stats_boxplots"], dpi=150, bbox_inches="tight"
)
plt.close(fig)
