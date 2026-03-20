"""Plaque–vessel spatial relationship analysis figures.

Reads  : data_{cohort}.parquet
Writes (one file per figure):
  fig_{cohort}_sdt_ecdf.png                    ECDF of signed distance transform
  fig_{cohort}_proximity_fractions_stacked.png stacked bar chart of proximity fractions (subject-level averages)
  fig_{cohort}_proximity_counts_stacked.png    grouped bar chart of subject-level proximity counts with error bars
  fig_{cohort}_proximity_fractions.png         boxplots of subject-level proximity fractions
  fig_{cohort}_proximity_interaction.png       interaction plots for proximity fractions
  fig_{cohort}_vessel_calibre_ecdf.png         ECDF of estimated vessel diameter
  fig_{cohort}_vessel_calibre_subject.png      subject-level boxplots of vessel calibre
  fig_{cohort}_vessel_diam_bins.png            fractions by vessel diameter bin
  fig_{cohort}_spatial_vessel_proximity.png    2-D scatter coloured by vessel relation
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from statsmodels.formula.api import ols  # noqa: E402
from statsmodels.stats.anova import anova_lm  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Snakemake integration ────────────────────────────────────────────────────
input_parquet = str(snakemake.input.parquet)  # noqa: F821
output_figs = dict(snakemake.output)  # noqa: F821
cohort = snakemake.wildcards.cohort  # noqa: F821

# ── Constants ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
TREAT_ORDER = ["PBS", "Lecanemab"]
TREAT_PALETTE = {"PBS": "#4C72B0", "Lecanemab": "#DD8452"}
GENO_ORDER = ["ApoE3", "ApoE4"]
GENO_PALETTE = {"ApoE3": "#55A868", "ApoE4": "#C44E52"}
VOL_THRESH_ML = 1e-4
TAU_UM = 5.0
VESSEL_BINS_UM = [0, 10, 20, np.inf]
VESSEL_BIN_LABELS = ["<10 \u00b5m", "10\u201320 \u00b5m", ">20 \u00b5m"]


# ── Helper functions ─────────────────────────────────────────────────────────
def classify_plaques(
    plaque_df,
    tau_um=TAU_UM,
    vessel_bins_um=VESSEL_BINS_UM,
    vessel_bin_labels=VESSEL_BIN_LABELS,
):
    """Add vessel-relation category and min-vessel-diameter estimate."""
    out = plaque_df.copy()
    out["vessel_relation"] = pd.cut(
        out["sdt_CD31_um"],
        bins=[-np.inf, 0, tau_um, np.inf],
        labels=[
            "inside vessel",
            f"near vessel (0\u2013{tau_um:g} \u00b5m)",
            f"far from vessel (>{tau_um:g} \u00b5m)",
        ],
        include_lowest=True,
        right=True,
    )
    out["min_vessel_diam_um"] = np.nan
    inside = out["sdt_CD31_um"] < 0
    out.loc[inside, "min_vessel_diam_um"] = (
        out.loc[inside, "equiv_diam_um"] - 2 * out.loc[inside, "sdt_CD31_um"]
    )
    out["vessel_diam_bin"] = pd.cut(
        out["min_vessel_diam_um"],
        bins=vessel_bins_um,
        labels=vessel_bin_labels,
        include_lowest=True,
        right=False,
    )
    return out


def subject_proximity_fractions(
    plaque_df,
    group_cols=("subject", "genotype", "treatment", "sex"),
):
    """Per-subject fraction in each vessel-relation class (wide format)."""
    counts = (
        plaque_df.groupby(list(group_cols) + ["vessel_relation"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = (
        plaque_df.groupby(list(group_cols), observed=True)
        .size()
        .rename("n_total")
        .reset_index()
    )
    out = counts.merge(totals, on=list(group_cols))
    out["fraction"] = out["n"] / out["n_total"]
    PROX_RENAME = {
        "inside vessel": "frac_inside_vessel",
        f"near vessel (0\u2013{TAU_UM:g} \u00b5m)": "frac_near_vessel",
        f"far from vessel (>{TAU_UM:g} \u00b5m)": "frac_far_vessel",
    }
    wide = (
        out.pivot_table(
            index=list(group_cols),
            columns="vessel_relation",
            values="fraction",
            fill_value=0.0,
        )
        .rename(columns=PROX_RENAME)
        .reset_index()
    )
    wide.columns.name = None
    return wide


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


# ── Load and prepare data ────────────────────────────────────────────────────
df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()
df["treatment"] = pd.Categorical(df["treatment"], categories=TREAT_ORDER, ordered=True)
geno_present = [g for g in GENO_ORDER if (df["genotype"] == g).any()]
df["genotype"] = pd.Categorical(df["genotype"], categories=geno_present, ordered=True)
geno_palette = {g: GENO_PALETTE[g] for g in geno_present}
n_geno = len(geno_present)

if "sdt_CD31_um" not in df.columns:
    df["sdt_CD31_um"] = df["sdt_CD31"] * 1000.0

PROX_PALETTE = {
    "inside vessel": "#C44E52",
    f"near vessel (0\u2013{TAU_UM:g} \u00b5m)": "#DD8452",
    f"far from vessel (>{TAU_UM:g} \u00b5m)": "#4C72B0",
}

# classify plaques for all figure types that need vessel_relation
df = classify_plaques(df)
relation_categories = df["vessel_relation"].cat.categories.tolist()

# ── Figure dispatch ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (hue, order, palette) in zip(
    axes,
    [
        ("treatment", TREAT_ORDER, TREAT_PALETTE),
        ("genotype", geno_present, geno_palette),
    ],
):
    sns.ecdfplot(data=df, x="sdt_CD31_um", hue=hue,
                 hue_order=order, palette=palette, ax=ax)
    ax.set_xlabel("SDT to CD31 vessel (\u00b5m)")
    ax.set_ylabel("Cumulative fraction")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlim(-100, 200)
axes[0].set_title("By treatment")
axes[1].set_title("By genotype")
fig.suptitle(
    f"SDT distribution by treatment/genotype \u2014 {cohort}", fontsize=13
)
plt.tight_layout()
plt.savefig(output_figs["sdt_ecdf"], dpi=150, bbox_inches="tight")
plt.close(fig)

# Compute per-subject fractions then average within each group so that
# groups with more subjects do not dominate the bar heights.
subj_stacked_counts = (
    df.groupby(
        ["subject", "treatment", "genotype", "vessel_relation"], observed=True
    )
    .size()
    .rename("n")
    .reset_index()
)
subj_stacked_totals = (
    df.groupby(["subject", "treatment", "genotype"], observed=True)
    .size()
    .rename("n_total")
    .reset_index()
)
subj_stacked_fracs = subj_stacked_counts.merge(
    subj_stacked_totals, on=["subject", "treatment", "genotype"]
)
subj_stacked_fracs["fraction"] = subj_stacked_fracs["n"] / subj_stacked_fracs["n_total"]
# Pivot to wide then melt so subjects with zero plaques in a category
# contribute 0 to the group mean rather than being absent.
subj_stacked_wide = subj_stacked_fracs.pivot_table(
    index=["subject", "treatment", "genotype"],
    columns="vessel_relation",
    values="fraction",
    fill_value=0.0,
).reset_index()
subj_stacked_wide.columns.name = None
subj_stacked_long = subj_stacked_wide.melt(
    id_vars=["subject", "treatment", "genotype"],
    var_name="vessel_relation",
    value_name="fraction",
)
prop_df = (
    subj_stacked_long.groupby(
        ["treatment", "genotype", "vessel_relation"], observed=True
    )["fraction"]
    .mean()
    .reset_index()
)
prop_df["group"] = (
    prop_df["treatment"].astype(str) + " | " + prop_df["genotype"].astype(str)
)

groups = prop_df["group"].unique().tolist()
bottoms = np.zeros(len(groups))
fig, ax = plt.subplots(figsize=(9, 5))
for cat in relation_categories:
    vals = [
        prop_df.loc[
            (prop_df["group"] == g) & (prop_df["vessel_relation"] == cat),
            "fraction",
        ].values[0]
        if len(
            prop_df.loc[
                (prop_df["group"] == g) & (prop_df["vessel_relation"] == cat)
            ]
        ) > 0
        else 0.0
        for g in groups
    ]
    color = PROX_PALETTE.get(cat, "grey")
    ax.bar(groups, vals, bottom=bottoms, label=cat, color=color, width=0.6)
    bottoms += np.array(vals)
ax.set_ylabel("Mean fraction of plaques per subject")
ax.set_xlabel("")
ax.set_xticklabels(groups, rotation=20, ha="right")
ax.legend(
    title="Vessel relation",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=8,
)
ax.set_title(f"Plaque\u2013vessel proximity fractions (subject avg) \u2014 {cohort}")
plt.tight_layout()
plt.savefig(output_figs["proximity_fractions_stacked"], dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Grouped bar chart: subject-level counts with error bars ──────────────────
subj_count_df = (
    df.groupby(
        ["subject", "treatment", "genotype", "vessel_relation"], observed=True
    )
    .size()
    .rename("n")
    .reset_index()
)
subj_count_agg = (
    subj_count_df.groupby(
        ["treatment", "genotype", "vessel_relation"], observed=True
    )["n"]
    .agg(["mean", "sem"])
    .reset_index()
)
subj_count_agg["group"] = (
    subj_count_agg["treatment"].astype(str)
    + " | "
    + subj_count_agg["genotype"].astype(str)
)
fig, ax = plt.subplots(figsize=(9, 5))
n_cats = len(relation_categories)
n_groups = len(groups)
group_width = 0.8
bar_width = group_width / n_cats
x = np.arange(n_groups)
for i, cat in enumerate(relation_categories):
    offsets = x + (i - (n_cats - 1) / 2) * bar_width
    cat_data = subj_count_agg[subj_count_agg["vessel_relation"] == cat]
    means = [
        cat_data.loc[cat_data["group"] == g, "mean"].values[0]
        if g in cat_data["group"].values
        else np.nan
        for g in groups
    ]
    sems = [
        cat_data.loc[cat_data["group"] == g, "sem"].values[0]
        if g in cat_data["group"].values
        else np.nan
        for g in groups
    ]
    color = PROX_PALETTE.get(cat, "grey")
    ax.bar(offsets, means, width=bar_width, label=cat, color=color)
    # NaN sems (e.g. single-subject groups) are silently dropped by errorbar
    ax.errorbar(offsets, means, yerr=sems, fmt="none", color="black", capsize=3)
ax.set_ylabel("Mean number of plaques per subject")
ax.set_xlabel("")
ax.set_xticks(x)
ax.set_xticklabels(groups, rotation=20, ha="right")
ax.legend(
    title="Vessel relation",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=8,
)
ax.set_title(f"Plaque\u2013vessel proximity counts (subject avg) \u2014 {cohort}")
plt.tight_layout()
plt.savefig(output_figs["proximity_counts_stacked"], dpi=150, bbox_inches="tight")
plt.close(fig)

subj_prox = subject_proximity_fractions(df)
subj_prox["genotype"] = pd.Categorical(
    subj_prox["genotype"], categories=geno_present, ordered=True
)
prox_frac_cols = [c for c in subj_prox.columns if c.startswith("frac_")]

fig, axes = plt.subplots(
    1, len(prox_frac_cols), figsize=(5 * len(prox_frac_cols), 5)
)
if len(prox_frac_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, prox_frac_cols):
    label = col.replace("frac_", "").replace("_", " ")
    if n_geno > 1:
        boxstrip(ax, subj_prox, x="treatment", y=col, hue="genotype",
                 order=TREAT_ORDER, hue_order=geno_present, palette=geno_palette,
                 ylabel=f"Fraction {label}")
    else:
        sns.boxplot(data=subj_prox, x="treatment", y=col, order=TREAT_ORDER,
                    palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                    ax=ax)
        sns.stripplot(data=subj_prox, x="treatment", y=col, order=TREAT_ORDER,
                      palette=TREAT_PALETTE, alpha=0.7, size=6, jitter=True, ax=ax)
        ax.set_ylabel(f"Fraction {label}")
    ax.set_xlabel("")
    ax.set_title(label)
fig.suptitle(
    f"Subject-level proximity fractions \u2014 {cohort}", fontsize=13
)
plt.tight_layout()
plt.savefig(output_figs["proximity_fractions"], dpi=150, bbox_inches="tight")
plt.close(fig)

subj_prox_int = subject_proximity_fractions(df)
subj_prox_int["genotype"] = pd.Categorical(
    subj_prox_int["genotype"], categories=geno_present, ordered=True
)
prox_int_cols = [c for c in subj_prox_int.columns if c.startswith("frac_")]

fig, axes = plt.subplots(
    1, len(prox_int_cols), figsize=(5 * len(prox_int_cols), 5)
)
if len(prox_int_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, prox_int_cols):
    label = col.replace("frac_", "").replace("_", " ")
    agg = (
        subj_prox_int.groupby(["treatment", "genotype"], observed=True)[col]
        .agg(["mean", "sem"])
        .reset_index()
    )
    for geno, grp in agg.groupby("genotype", observed=True):
        color = GENO_PALETTE[geno]
        ax.plot(grp["treatment"].astype(str), grp["mean"],
                marker="o", linewidth=2, color=color, label=geno)
        ax.errorbar(grp["treatment"].astype(str), grp["mean"],
                    yerr=grp["sem"], fmt="none", color=color, capsize=4)
    ax.set_ylabel(f"Fraction {label}")
    ax.set_xlabel("Treatment")
    ax.legend(title="Genotype", fontsize=9)
    ax.set_title(label)
fig.suptitle(
    f"Proximity interaction plots \u2014 {cohort}", fontsize=13
)
plt.tight_layout()
plt.savefig(output_figs["proximity_interaction"], dpi=150, bbox_inches="tight")
plt.close(fig)

df_inside = df.loc[df["sdt_CD31_um"] < 0].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (hue, order, palette) in zip(
    axes,
    [
        ("treatment", TREAT_ORDER, TREAT_PALETTE),
        ("genotype", geno_present, geno_palette),
    ],
):
    sns.ecdfplot(data=df_inside, x="min_vessel_diam_um",
                 hue=hue, hue_order=order, palette=palette, ax=ax)
    ax.set_xlabel("Estimated min vessel diameter (\u00b5m)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0, 100)
axes[0].set_title("By treatment")
axes[1].set_title("By genotype")
fig.suptitle(
    f"Estimated vessel diameter (intravascular plaques) \u2014 {cohort}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["vessel_calibre_ecdf"], dpi=150, bbox_inches="tight")
plt.close(fig)

subj_calibre = (
    df_inside.groupby(
        ["subject", "treatment", "genotype", "sex"], observed=True
    )
    .agg(
        n_intravascular=("min_vessel_diam_um", "size"),
        mean_min_diam_um=("min_vessel_diam_um", "mean"),
        median_min_diam_um=("min_vessel_diam_um", "median"),
        mean_sdt_inside_um=("sdt_CD31_um", "mean"),
        median_sdt_inside_um=("sdt_CD31_um", "median"),
    )
    .reset_index()
)
subj_calibre["genotype"] = pd.Categorical(
    subj_calibre["genotype"], categories=geno_present, ordered=True
)
calibre_metrics = [
    ("mean_min_diam_um", "Mean estimated min vessel diam. (\u00b5m)"),
    ("median_sdt_inside_um", "Median SDT inside vessel (\u00b5m)"),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (col, label) in zip(axes, calibre_metrics):
    if n_geno > 1:
        boxstrip(ax, subj_calibre, x="treatment", y=col, hue="genotype",
                 order=TREAT_ORDER, hue_order=geno_present, palette=geno_palette,
                 ylabel=label)
    else:
        sns.boxplot(data=subj_calibre, x="treatment", y=col, order=TREAT_ORDER,
                    palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                    ax=ax)
        sns.stripplot(data=subj_calibre, x="treatment", y=col, order=TREAT_ORDER,
                      palette=TREAT_PALETTE, alpha=0.75, size=7, jitter=True, ax=ax)
        ax.set_ylabel(label)
    ax.set_xlabel("")
fig.suptitle(
    f"Subject-level vessel-calibre estimates \u2014 {cohort}", fontsize=13
)
plt.tight_layout()
plt.savefig(output_figs["vessel_calibre_subject"], dpi=150, bbox_inches="tight")
plt.close(fig)

def subject_vessel_diam_fractions(
    plaque_df,
    group_cols=("subject", "genotype", "treatment", "sex"),
):
    inside = plaque_df.loc[plaque_df["sdt_CD31_um"] < 0]
    counts = (
        inside.groupby(list(group_cols) + ["vessel_diam_bin"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = (
        inside.groupby(list(group_cols), observed=True)
        .size()
        .rename("n_total")
        .reset_index()
    )
    out = counts.merge(totals, on=list(group_cols))
    out["fraction"] = out["n"] / out["n_total"]
    return out

subj_diam = subject_vessel_diam_fractions(df)
subj_diam["genotype"] = pd.Categorical(
    subj_diam["genotype"], categories=geno_present, ordered=True
)
diam_bins = df["vessel_diam_bin"].cat.categories.tolist()
fig, axes = plt.subplots(1, len(diam_bins), figsize=(5 * len(diam_bins), 5))
if len(diam_bins) == 1:
    axes = [axes]
for ax, bin_label in zip(axes, diam_bins):
    bin_data = subj_diam.loc[subj_diam["vessel_diam_bin"] == bin_label]
    if n_geno > 1:
        boxstrip(ax, bin_data, x="treatment", y="fraction", hue="genotype",
                 order=TREAT_ORDER, hue_order=geno_present, palette=geno_palette,
                 ylabel="Fraction of intravascular plaques")
    else:
        sns.boxplot(data=bin_data, x="treatment", y="fraction", order=TREAT_ORDER,
                    palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                    ax=ax)
        sns.stripplot(data=bin_data, x="treatment", y="fraction", order=TREAT_ORDER,
                      palette=TREAT_PALETTE, alpha=0.75, size=7, jitter=True, ax=ax)
        ax.set_ylabel("Fraction of intravascular plaques")
    ax.set_title(f"Vessel diam. {bin_label}", fontsize=10)
    ax.set_xlabel("")
fig.suptitle(
    f"Intravascular plaques by vessel diameter bin \u2014 {cohort}", fontsize=13
)
plt.tight_layout()
plt.savefig(output_figs["vessel_diam_bins"], dpi=150, bbox_inches="tight")
plt.close(fig)

n_sample = 30_000
sample = df.sample(min(n_sample, len(df)), random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(
    data=sample, x="template_x", y="template_y",
    hue="vessel_relation", hue_order=relation_categories,
    palette=PROX_PALETTE, alpha=0.3, s=4, ax=axes[0],
)
axes[0].set_aspect("equal")
axes[0].set_title("XY projection")
axes[0].get_legend().remove()

sns.scatterplot(
    data=sample, x="template_x", y="template_z",
    hue="vessel_relation", hue_order=relation_categories,
    palette=PROX_PALETTE, alpha=0.3, s=4, ax=axes[1],
)
axes[1].set_aspect("equal")
axes[1].set_title("XZ projection")
axes[1].legend(
    title="Vessel relation",
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    fontsize=8,
    markerscale=3,
)
fig.suptitle(
    f"2-D spatial distribution coloured by vessel proximity \u2014 {cohort}"
    f"  (n={n_sample:,} sample)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["spatial_vessel_proximity"], dpi=150, bbox_inches="tight")
plt.close(fig)
