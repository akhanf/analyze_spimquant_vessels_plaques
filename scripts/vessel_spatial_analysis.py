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
  fig_{cohort}_spatial_plaques_size.png        XY/XZ projections with marker size ∝ equiv. diameter (mm)
  fig_{cohort}_spatial_plaques_vessel_dist_bins.png  XY/XZ projections filtered by 3 vessel-distance bins
  fig_{cohort}_spatial_plaques_vessel_dist_anim.gif  animated sweep through vessel-distance window
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.lines as mlines  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from statsmodels.formula.api import ols  # noqa: E402
from statsmodels.stats.anova import anova_lm  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Snakemake integration ────────────────────────────────────────────────────
input_parquet = str(snakemake.input.parquet)  # noqa: F821
output_figs = dict(snakemake.output)  # noqa: F821
cohort = snakemake.wildcards.cohort  # noqa: F821
cfg = snakemake.params.plot_config  # noqa: F821

# ── Extract plot config ───────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)

# factors[0] = primary factor (e.g. treatment), factors[1] = secondary (e.g. genotype),
# factors[2] = tertiary (e.g. sex).  Swapping entries in config.yml changes which
# factor plays which role without touching any script.
_factors = cfg["factors"]
treat_cfg = _factors[0]
TREAT_COL = treat_cfg["column"]
TREAT_ORDER = treat_cfg["order"]
TREAT_PALETTE = treat_cfg["palette"]

geno_cfg = _factors[1] if len(_factors) > 1 else {}
GENO_COL = geno_cfg.get("column", "")
GENO_ORDER = geno_cfg.get("order", [])
GENO_PALETTE = geno_cfg.get("palette", {})

sex_cfg = _factors[2] if len(_factors) > 2 else {}
SEX_COL = sex_cfg.get("column", "")

VOL_THRESH_ML = cfg.get("volume_threshold_ml", 1e-4)

# Fallback colour used when a category value is absent from the configured palette.
DEFAULT_COLOR = "#888888"

treat_label = TREAT_COL.replace("_", " ").title()
geno_label = GENO_COL.replace("_", " ").title()
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
    group_cols=None,
):
    """Per-subject fraction in each vessel-relation class (wide format)."""
    if group_cols is None:
        group_cols = ("subject", GENO_COL, TREAT_COL, SEX_COL)
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
    """Two-way OLS ANOVA (or one-way when only one group present)."""
    n_geno = data[GENO_COL].nunique()
    if n_geno > 1:
        formula = f"{outcome} ~ C({TREAT_COL}) * C({GENO_COL})"
    else:
        formula = f"{outcome} ~ C({TREAT_COL})"
    model = ols(formula, data=data).fit()
    return anova_lm(model, typ=2)


# ── Load and prepare data ────────────────────────────────────────────────────
df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()
df[TREAT_COL] = pd.Categorical(df[TREAT_COL], categories=TREAT_ORDER, ordered=True)
geno_present = [g for g in GENO_ORDER if (df[GENO_COL] == g).any()]
df[GENO_COL] = pd.Categorical(df[GENO_COL], categories=geno_present, ordered=True)
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
        (TREAT_COL, TREAT_ORDER, TREAT_PALETTE),
        (GENO_COL, geno_present, geno_palette),
    ],
):
    sns.ecdfplot(data=df, x="sdt_CD31_um", hue=hue,
                 hue_order=order, palette=palette, ax=ax)
    ax.set_xlabel("SDT to CD31 vessel (\u00b5m)")
    ax.set_ylabel("Cumulative fraction")
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlim(-100, 200)
axes[0].set_title(f"By {treat_label}")
axes[1].set_title(f"By {geno_label}")
fig.suptitle(
    f"SDT distribution by {treat_label}/{geno_label} \u2014 {cohort}", fontsize=13
)
plt.tight_layout()
plt.savefig(output_figs["sdt_ecdf"], dpi=150, bbox_inches="tight")
plt.close(fig)

# Compute per-subject fractions then average within each group so that
# groups with more subjects do not dominate the bar heights.
subj_stacked_counts = (
    df.groupby(
        ["subject", TREAT_COL, GENO_COL, "vessel_relation"], observed=True
    )
    .size()
    .rename("n")
    .reset_index()
)
subj_stacked_totals = (
    df.groupby(["subject", TREAT_COL, GENO_COL], observed=True)
    .size()
    .rename("n_total")
    .reset_index()
)
subj_stacked_fracs = subj_stacked_counts.merge(
    subj_stacked_totals, on=["subject", TREAT_COL, GENO_COL]
)
subj_stacked_fracs["fraction"] = subj_stacked_fracs["n"] / subj_stacked_fracs["n_total"]
subj_stacked_wide = subj_stacked_fracs.pivot_table(
    index=["subject", TREAT_COL, GENO_COL],
    columns="vessel_relation",
    values="fraction",
    fill_value=0.0,
).reset_index()
subj_stacked_wide.columns.name = None
subj_stacked_long = subj_stacked_wide.melt(
    id_vars=["subject", TREAT_COL, GENO_COL],
    var_name="vessel_relation",
    value_name="fraction",
)
prop_df = (
    subj_stacked_long.groupby(
        [TREAT_COL, GENO_COL, "vessel_relation"], observed=True
    )["fraction"]
    .mean()
    .reset_index()
)
prop_df["group"] = (
    prop_df[TREAT_COL].astype(str) + " | " + prop_df[GENO_COL].astype(str)
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
        ["subject", TREAT_COL, GENO_COL, "vessel_relation"], observed=True
    )
    .size()
    .rename("n")
    .reset_index()
)
subj_count_agg = (
    subj_count_df.groupby(
        [TREAT_COL, GENO_COL, "vessel_relation"], observed=True
    )["n"]
    .agg(["mean", "sem"])
    .reset_index()
)
subj_count_agg["group"] = (
    subj_count_agg[TREAT_COL].astype(str)
    + " | "
    + subj_count_agg[GENO_COL].astype(str)
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
subj_prox[GENO_COL] = pd.Categorical(
    subj_prox[GENO_COL], categories=geno_present, ordered=True
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
        boxstrip(ax, subj_prox, x=TREAT_COL, y=col, hue=GENO_COL,
                 order=TREAT_ORDER, hue_order=geno_present, palette=geno_palette,
                 ylabel=f"Fraction {label}")
    else:
        sns.boxplot(data=subj_prox, x=TREAT_COL, y=col, order=TREAT_ORDER,
                    palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                    ax=ax)
        sns.stripplot(data=subj_prox, x=TREAT_COL, y=col, order=TREAT_ORDER,
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
subj_prox_int[GENO_COL] = pd.Categorical(
    subj_prox_int[GENO_COL], categories=geno_present, ordered=True
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
        subj_prox_int.groupby([TREAT_COL, GENO_COL], observed=True)[col]
        .agg(["mean", "sem"])
        .reset_index()
    )
    for geno, grp in agg.groupby(GENO_COL, observed=True):
        color = GENO_PALETTE.get(geno, DEFAULT_COLOR)
        ax.plot(grp[TREAT_COL].astype(str), grp["mean"],
                marker="o", linewidth=2, color=color, label=geno)
        ax.errorbar(grp[TREAT_COL].astype(str), grp["mean"],
                    yerr=grp["sem"], fmt="none", color=color, capsize=4)
    ax.set_ylabel(f"Fraction {label}")
    ax.set_xlabel(treat_label)
    ax.legend(title=geno_label, fontsize=9)
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
        (TREAT_COL, TREAT_ORDER, TREAT_PALETTE),
        (GENO_COL, geno_present, geno_palette),
    ],
):
    sns.ecdfplot(data=df_inside, x="min_vessel_diam_um",
                 hue=hue, hue_order=order, palette=palette, ax=ax)
    ax.set_xlabel("Estimated min vessel diameter (\u00b5m)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_xlim(0, 100)
axes[0].set_title(f"By {treat_label}")
axes[1].set_title(f"By {geno_label}")
fig.suptitle(
    f"Estimated vessel diameter (intravascular plaques) \u2014 {cohort}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["vessel_calibre_ecdf"], dpi=150, bbox_inches="tight")
plt.close(fig)

subj_calibre = (
    df_inside.groupby(
        ["subject", TREAT_COL, GENO_COL, SEX_COL], observed=True
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
subj_calibre[GENO_COL] = pd.Categorical(
    subj_calibre[GENO_COL], categories=geno_present, ordered=True
)
calibre_metrics = [
    ("mean_min_diam_um", "Mean estimated min vessel diam. (\u00b5m)"),
    ("median_sdt_inside_um", "Median SDT inside vessel (\u00b5m)"),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (col, label) in zip(axes, calibre_metrics):
    if n_geno > 1:
        boxstrip(ax, subj_calibre, x=TREAT_COL, y=col, hue=GENO_COL,
                 order=TREAT_ORDER, hue_order=geno_present, palette=geno_palette,
                 ylabel=label)
    else:
        sns.boxplot(data=subj_calibre, x=TREAT_COL, y=col, order=TREAT_ORDER,
                    palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                    ax=ax)
        sns.stripplot(data=subj_calibre, x=TREAT_COL, y=col, order=TREAT_ORDER,
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
    group_cols=None,
):
    if group_cols is None:
        group_cols = ("subject", GENO_COL, TREAT_COL, SEX_COL)
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
subj_diam[GENO_COL] = pd.Categorical(
    subj_diam[GENO_COL], categories=geno_present, ordered=True
)
diam_bins = df["vessel_diam_bin"].cat.categories.tolist()
fig, axes = plt.subplots(1, len(diam_bins), figsize=(5 * len(diam_bins), 5))
if len(diam_bins) == 1:
    axes = [axes]
for ax, bin_label in zip(axes, diam_bins):
    bin_data = subj_diam.loc[subj_diam["vessel_diam_bin"] == bin_label]
    if n_geno > 1:
        boxstrip(ax, bin_data, x=TREAT_COL, y="fraction", hue=GENO_COL,
                 order=TREAT_ORDER, hue_order=geno_present, palette=geno_palette,
                 ylabel="Fraction of intravascular plaques")
    else:
        sns.boxplot(data=bin_data, x=TREAT_COL, y="fraction", order=TREAT_ORDER,
                    palette=TREAT_PALETTE, fill=False, linewidth=1.2, fliersize=0,
                    ax=ax)
        sns.stripplot(data=bin_data, x=TREAT_COL, y="fraction", order=TREAT_ORDER,
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

# ── Figure: XY / XZ projections with marker size ∝ equiv. diameter (mm) ─────
# Build ordered list of (treatment, genotype) groups present in the data.
group_list = [
    (t, g)
    for t in TREAT_ORDER
    for g in GENO_ORDER
    if ((df[TREAT_COL] == t) & (df[GENO_COL] == g)).any()
]
n_groups = len(group_list)

df["equiv_diam_mm"] = df["equiv_diam_um"] / 1000.0

# Cap extreme diameters for display (99th percentile) to avoid single huge dots
diam_cap_mm = df["equiv_diam_mm"].quantile(0.99)
df["equiv_diam_mm_capped"] = df["equiv_diam_mm"].clip(upper=diam_cap_mm)
# Marker area ∝ (diameter_mm)²; scale so median maps to ~10 pt²
_median_diam = df["equiv_diam_mm_capped"].median()
DIAM_SCALE = 10.0 / (_median_diam ** 2) if _median_diam > 0 else 1.0
df["s_marker"] = (df["equiv_diam_mm_capped"] ** 2) * DIAM_SCALE

PROJECTIONS = [
    ("template_x", "template_y", "XY"),
    ("template_x", "template_z", "XZ"),
]
N_SAMPLE_GROUP = 5_000  # per-group sample for clarity

fig, axes = plt.subplots(
    2, n_groups, figsize=(4.5 * n_groups, 8), squeeze=False
)
for col_idx, (treat, geno) in enumerate(group_list):
    mask = (df[TREAT_COL] == treat) & (df[GENO_COL] == geno)
    subset = df.loc[mask].sample(
        min(N_SAMPLE_GROUP, int(mask.sum())), random_state=42
    )
    color = TREAT_PALETTE[treat]
    for row_idx, (xcol, ycol, proj_label) in enumerate(PROJECTIONS):
        ax = axes[row_idx, col_idx]
        ax.scatter(
            subset[xcol], subset[ycol],
            s=subset["s_marker"], c=color, alpha=0.35, linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(xcol.replace("template_", "").upper() + " (mm)")
        ax.set_ylabel(ycol.replace("template_", "").upper() + " (mm)")
        if row_idx == 0:
            ax.set_title(f"{treat} / {geno}\n{proj_label}", fontsize=9)
        else:
            ax.set_title(proj_label, fontsize=9)

# Size legend: show 3 representative diameters
legend_diams_mm = [
    df["equiv_diam_mm_capped"].quantile(0.25),
    df["equiv_diam_mm_capped"].median(),
    diam_cap_mm,
]
legend_handles = [
    mlines.Line2D(
        [], [], linestyle="none", marker="o",
        markerfacecolor="grey", markeredgecolor="none", alpha=0.6,
        markersize=np.sqrt((d ** 2) * DIAM_SCALE),
        label=f"{d * 1000:.0f} \u00b5m",
    )
    for d in legend_diams_mm
]
axes[0, -1].legend(
    handles=legend_handles, title="Equiv. diam.", fontsize=8,
    loc="upper right",
)
fig.suptitle(
    f"Plaque projections (size \u221d equiv. diameter) \u2014 {cohort}"
    f"  (n\u2264{N_SAMPLE_GROUP:,}/group)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["spatial_plaques_size"], dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure: XY / XZ projections filtered by 3 vessel-distance bins ───────────
SDT_BIN_EDGES_UM = [-np.inf, 0.0, 50.0, np.inf]
SDT_BIN_LABELS_UM = [
    "< 0 \u00b5m (inside vessel)",
    "0\u201350 \u00b5m (near vessel)",
    "\u2265 50 \u00b5m (far from vessel)",
]
SDT_BIN_COLORS = ["#C44E52", "#DD8452", "#4C72B0"]

df["sdt_dist_bin"] = pd.cut(
    df["sdt_CD31_um"],
    bins=SDT_BIN_EDGES_UM,
    labels=SDT_BIN_LABELS_UM,
    include_lowest=True,
    right=True,
)
n_sdt_bins = len(SDT_BIN_LABELS_UM)

fig, axes = plt.subplots(
    n_sdt_bins * 2, n_groups,
    figsize=(4.0 * n_groups, 4.0 * n_sdt_bins),
    squeeze=False,
)
for col_idx, (treat, geno) in enumerate(group_list):
    mask = (df[TREAT_COL] == treat) & (df[GENO_COL] == geno)
    group_df = df.loc[mask]
    for bin_idx, (bin_label, bin_color) in enumerate(
        zip(SDT_BIN_LABELS_UM, SDT_BIN_COLORS)
    ):
        bin_subset = group_df.loc[group_df["sdt_dist_bin"] == bin_label]
        bin_sample = bin_subset.sample(
            min(N_SAMPLE_GROUP, len(bin_subset)), random_state=42
        )
        for proj_idx, (xcol, ycol, proj_label) in enumerate(PROJECTIONS):
            row_idx = bin_idx * 2 + proj_idx
            ax = axes[row_idx, col_idx]
            ax.scatter(
                bin_sample[xcol], bin_sample[ycol],
                s=4, c=bin_color, alpha=0.3, linewidths=0,
            )
            ax.set_aspect("equal")
            ax.set_xlabel(xcol.replace("template_", "").upper() + " (mm)")
            ax.set_ylabel(ycol.replace("template_", "").upper() + " (mm)")
            if col_idx == 0:
                ax.set_ylabel(
                    f"{bin_label}\n"
                    + ycol.replace("template_", "").upper() + " (mm)"
                )
            if bin_idx == 0 and proj_idx == 0:
                ax.set_title(f"{treat} / {geno}\n{proj_label}", fontsize=9)
            else:
                ax.set_title(proj_label, fontsize=9)
            n_shown = len(bin_sample)
            ax.text(
                0.02, 0.97, f"n={n_shown:,}",
                transform=ax.transAxes, fontsize=7,
                va="top", ha="left", color="black",
            )
fig.suptitle(
    f"Plaque projections by vessel-distance bin \u2014 {cohort}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(
    output_figs["spatial_plaques_vessel_dist_bins"], dpi=150, bbox_inches="tight"
)
plt.close(fig)

# ── Animated figure: sliding vessel-distance window sweep ────────────────────
ANIM_WIN_UM = 25.0        # window width in µm
ANIM_STEP_UM = 5.0        # step size between frames
ANIM_FPS = 8              # frames per second
# Constrain animation to the biological range of interest; clamp to ±50 / 130 µm
# to exclude rare extreme outliers while still covering the full vessel-distance
# distribution (inside vessel → far from vessel).
ANIM_SDT_MIN = max(float(df["sdt_CD31_um"].min()), -50.0)
ANIM_SDT_MAX = min(float(df["sdt_CD31_um"].max()), 130.0)
N_ANIM_SAMPLE = 2_000     # max points per group per frame

frame_starts = np.arange(ANIM_SDT_MIN, ANIM_SDT_MAX - ANIM_WIN_UM + ANIM_STEP_UM, ANIM_STEP_UM)

# Pre-compute axis limits from the full dataset
_xlim = (df["template_x"].quantile(0.01), df["template_x"].quantile(0.99))
_ylim_xy = (df["template_y"].quantile(0.01), df["template_y"].quantile(0.99))
_ylim_xz = (df["template_z"].quantile(0.01), df["template_z"].quantile(0.99))

fig_anim, axes_anim = plt.subplots(
    2, n_groups, figsize=(4.0 * n_groups, 7.0), squeeze=False
)
# Initialise empty scatter objects and store metadata
_scat_info = {}
for col_idx, (treat, geno) in enumerate(group_list):
    for row_idx, (xcol, ycol, proj_label) in enumerate(PROJECTIONS):
        ax = axes_anim[row_idx, col_idx]
        sc = ax.scatter([], [], s=5, c=TREAT_PALETTE[treat], alpha=0.4, linewidths=0)
        ax.set_aspect("equal")
        ax.set_xlim(_xlim)
        ax.set_ylim(_ylim_xy if row_idx == 0 else _ylim_xz)
        ax.set_xlabel(xcol.replace("template_", "").upper() + " (mm)", fontsize=8)
        ax.set_ylabel(ycol.replace("template_", "").upper() + " (mm)", fontsize=8)
        if row_idx == 0:
            ax.set_title(f"{treat} / {geno}\n{proj_label}", fontsize=8)
        else:
            ax.set_title(proj_label, fontsize=8)
        _scat_info[(col_idx, row_idx)] = (sc, xcol, ycol, treat, geno)

title_text = fig_anim.suptitle("", fontsize=11)
plt.tight_layout()


def _anim_update(frame_idx):
    start_um = frame_starts[frame_idx]
    end_um = start_um + ANIM_WIN_UM
    title_text.set_text(
        f"{cohort} \u2014 plaques within "
        f"{start_um:.0f}\u2013{end_um:.0f} \u00b5m vessel distance"
    )
    # Window is [start_um, end_um) — half-open interval so consecutive windows
    # do not double-count plaques exactly on a boundary.
    in_window = (df["sdt_CD31_um"] >= start_um) & (df["sdt_CD31_um"] < end_um)
    artists = []
    for (col_idx, row_idx), (sc, xcol, ycol, treat, geno) in _scat_info.items():
        group_mask = in_window & (df[TREAT_COL] == treat) & (df[GENO_COL] == geno)
        pts = df.loc[group_mask, [xcol, ycol]]
        if len(pts) > N_ANIM_SAMPLE:
            pts = pts.sample(N_ANIM_SAMPLE, random_state=frame_idx)
        if len(pts) > 0:
            sc.set_offsets(pts.values)
        else:
            sc.set_offsets(np.empty((0, 2)))
        artists.append(sc)
    return artists


anim = FuncAnimation(
    fig_anim, _anim_update, frames=len(frame_starts), blit=True,
    interval=1000 // ANIM_FPS,
)
writer = PillowWriter(fps=ANIM_FPS)
anim.save(output_figs["spatial_plaques_vessel_dist_anim"], writer=writer, dpi=100)
plt.close(fig_anim)
