"""Regional (ROI) and atlas-level analysis figures.

Reads  : roi_data_{cohort}.parquet  +  tpl-ABAv3_seg-all_dseg.nii.gz
Writes (one file per figure):
  fig_{cohort}_roi_top_density.png         bar chart of top 20 regions by plaque density
  fig_{cohort}_roi_density_boxplot.png     boxplot of density in top regions by treatment
  fig_{cohort}_roi_metric_heatmap.png      multi-metric normalised heatmap
  fig_{cohort}_roi_proximity_fractions.png vessel-proximity fraction boxplots (top regions)
  fig_{cohort}_roi_fold_change.png         treatment fold-change heatmap (region × genotype)
  fig_{cohort}_atlas_density_all.png       atlas MIP coloured by mean plaque density
  fig_{cohort}_atlas_density_groups.png    atlas MIPs per treatment × genotype group
  fig_{cohort}_atlas_fold_change.png       atlas MIPs coloured by log₂ fold-change
  fig_{cohort}_atlas_mean_diam.png         atlas MIP coloured by mean plaque diameter
  fig_{cohort}_atlas_frac_inside.png       atlas MIP coloured by fraction inside vessel
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

warnings.filterwarnings("ignore")

input_roi = str(snakemake.input.roi_parquet)  # noqa: F821
input_atlas = str(snakemake.input.atlas)  # noqa: F821
output_figs = dict(snakemake.output)  # noqa: F821
cohort = snakemake.wildcards.cohort  # noqa: F821

# ── Constants ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
TREAT_ORDER = ["PBS", "Lecanemab"]
TREAT_PALETTE = {"PBS": "#4C72B0", "Lecanemab": "#DD8452"}
GENO_ORDER = ["ApoE3", "ApoE4"]
GENO_PALETTE = {"ApoE3": "#55A868", "ApoE4": "#C44E52"}
TOP_N = 20


# ── Helper functions ─────────────────────────────────────────────────────────
def make_metric_volume(atlas_data, region_series):
    """Paint a metric onto the atlas volume using vectorised index lookup."""
    max_idx = int(atlas_data.max()) + 1
    lookup = np.full(max_idx, np.nan, dtype=np.float32)
    for ridx, val in region_series.items():
        if 0 <= int(ridx) < max_idx:
            lookup[int(ridx)] = float(val)
    vol = lookup[atlas_data]
    vol[atlas_data == 0] = np.nan
    return vol


def plot_atlas_heatmap(vol, title="", cmap="hot", cbar_label="",
                       vmin=None, vmax=None, savepath=None):
    """Display max-intensity projections along the three atlas axes."""
    projs = [np.nanmax(vol, axis=a) for a in range(3)]
    view_names = ["Projection along x", "Projection along y", "Projection along z"]

    _vmin = vmin if vmin is not None else np.nanmin(vol)
    _vmax = vmax if vmax is not None else np.nanmax(vol)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, proj, pname in zip(axes, projs, view_names):
        masked = np.ma.masked_invalid(proj)
        im = ax.imshow(
            masked.T, origin="lower", cmap=cmap,
            vmin=_vmin, vmax=_vmax, interpolation="nearest",
        )
        ax.set_title(pname, fontsize=10)
        ax.axis("off")

    plt.colorbar(im, ax=axes[-1], label=cbar_label, shrink=0.8)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.close(fig)


def roi_fold_change(roi_df, metric="plaque_density", top_idx=None, top_regions=None,
                    geno_order=None):
    """Return a (region × genotype) DataFrame of median fold-changes Lec / PBS."""
    if geno_order is None:
        geno_order = GENO_ORDER
    sub = roi_df.loc[roi_df["index"].isin(top_idx)].copy()
    rows = []
    for geno in geno_order:
        for _, row in top_regions.iterrows():
            pbs = sub.loc[
                (sub["genotype"] == geno) & (sub["treatment"] == "PBS")
                & (sub["index"] == row["index"]), metric
            ]
            lec = sub.loc[
                (sub["genotype"] == geno) & (sub["treatment"] == "Lecanemab")
                & (sub["index"] == row["index"]), metric
            ]
            fc = (
                lec.median() / pbs.median()
                if (len(pbs) > 0 and len(lec) > 0 and pbs.median() > 0)
                else np.nan
            )
            rows.append({"genotype": geno, "name": row["name"], "fold_change": fc})
    return pd.DataFrame(rows).pivot(index="name", columns="genotype", values="fold_change")


# ── Load data ─────────────────────────────────────────────────────────────────
roi_df_raw = pd.read_parquet(input_roi)
atlas_img = nib.load(input_atlas)
atlas_data = atlas_img.get_fdata().astype(np.int32)

roi_df = roi_df_raw.loc[roi_df_raw["volume_mm3"] > 0].copy()
roi_df["treatment"] = pd.Categorical(roi_df["treatment"], categories=TREAT_ORDER, ordered=True)
geno_present = [g for g in GENO_ORDER if (roi_df["genotype"] == g).any()]
roi_df["genotype"] = pd.Categorical(roi_df["genotype"], categories=geno_present, ordered=True)
geno_palette = {g: GENO_PALETTE[g] for g in geno_present}
n_geno = len(geno_present)

METRIC_COLS = [
    "plaque_count", "plaque_density", "vol_density_ml",
    "mean_diam_um", "median_diam_um",
    "mean_sdt_um", "median_sdt_um",
    "frac_inside_vessel", "frac_near_vessel", "frac_far_vessel",
]
roi_mean = (
    roi_df.groupby(["index", "name"], observed=True)[METRIC_COLS].mean().reset_index()
)
top_regions = (
    roi_mean.nlargest(TOP_N, "plaque_density")[
        ["index", "name", "plaque_density", "plaque_count", "vol_density_ml",
         "mean_diam_um", "frac_inside_vessel", "frac_near_vessel", "frac_far_vessel"]
    ].reset_index(drop=True)
)
top_idx = top_regions["index"].tolist()
roi_top = roi_df.loc[roi_df["index"].isin(top_idx)].copy()
roi_top["region_abbr"] = roi_top["name"].str.replace(r"^(left|right) ", "", regex=True)


# ── Figure dispatch ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    top_regions["name"],
    top_regions["plaque_density"],
    color="steelblue", edgecolor="white", height=0.7,
)
ax.set_xlabel("Mean plaque density (plaques / mm\u00b3)")
ax.set_title(f"Top {TOP_N} regions by plaque density \u2014 {cohort}")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(output_figs["roi_top_density"], dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, n_geno, figsize=(8 * n_geno, 7), sharey=True, squeeze=False)
for ax, geno in zip(axes[0], geno_present):
    sub = roi_top.loc[roi_top["genotype"] == geno]
    sns.boxplot(
        data=sub, y="region_abbr", x="plaque_density", hue="treatment",
        hue_order=TREAT_ORDER, palette=TREAT_PALETTE, orient="h", width=0.6, ax=ax,
    )
    ax.set_title(geno, fontsize=13)
    ax.set_xlabel("Plaque density (plaques / mm\u00b3)")
    ax.set_ylabel("Region")
    ax.legend(title="Treatment", loc="lower right")
fig.suptitle(
    f"Top {TOP_N} regions \u2014 plaque density by treatment \u2014 {cohort}",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(output_figs["roi_density_boxplot"], dpi=150)
plt.close(fig)

DISPLAY_METRICS = {
    "plaque_density": "Density\n(n/mm\u00b3)",
    "vol_density_ml": "Vol density\n(mL/mm\u00b3)",
    "mean_diam_um": "Mean diam\n(\u00b5m)",
    "median_diam_um": "Median diam\n(\u00b5m)",
    "mean_sdt_um": "Mean SDT\n(\u00b5m)",
    "frac_inside_vessel": "Frac inside\nvessel",
    "frac_near_vessel": "Frac near\nvessel",
    "frac_far_vessel": "Frac far\nfrom vessel",
}
summary_tbl = (
    roi_mean.loc[roi_mean["index"].isin(top_idx)]
    .set_index("name")[list(DISPLAY_METRICS.keys())]
    .rename(columns=DISPLAY_METRICS)
    .loc[top_regions["name"]]
)
norm_tbl = (summary_tbl - summary_tbl.min()) / (
    summary_tbl.max() - summary_tbl.min() + 1e-12
)
fig, ax = plt.subplots(figsize=(13, 8))
im = ax.imshow(norm_tbl.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(norm_tbl.columns)))
ax.set_xticklabels(norm_tbl.columns, fontsize=9)
ax.set_yticks(range(len(norm_tbl)))
ax.set_yticklabels(norm_tbl.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Normalised value")
ax.set_title(
    f"Multi-metric summary \u2014 top {TOP_N} regions \u2014 {cohort}"
    " (all-subjects mean, normalised per column)"
)
plt.tight_layout()
plt.savefig(output_figs["roi_metric_heatmap"], dpi=150)
plt.close(fig)

prox_cols = ["frac_inside_vessel", "frac_near_vessel", "frac_far_vessel"]
prox_labels = ["Inside vessel", "Near vessel", "Far from vessel"]
fig, axes = plt.subplots(1, len(prox_cols), figsize=(18, 7), sharey=True)
for ax, col, label in zip(axes, prox_cols, prox_labels):
    sns.boxplot(
        data=roi_top, y="region_abbr", x=col, hue="treatment",
        hue_order=TREAT_ORDER, palette=TREAT_PALETTE, orient="h", width=0.6, ax=ax,
    )
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Fraction of plaques")
    ax.set_ylabel("Region" if ax is axes[0] else "")
    ax.legend(title="Treatment", fontsize=8)
fig.suptitle(
    f"Vessel-proximity fractions \u2014 top {TOP_N} regions \u2014 {cohort}",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(output_figs["roi_proximity_fractions"], dpi=150)
plt.close(fig)

fc_df = roi_fold_change(
    roi_df, top_idx=top_idx, top_regions=top_regions, geno_order=geno_present
)
log_fc = np.log2(fc_df.values.astype(float))
vabs = np.nanmax(np.abs(log_fc))
if not np.isfinite(vabs):
    vabs = 1.0
fig, ax = plt.subplots(figsize=(4 + 2 * n_geno, 8))
im = ax.imshow(log_fc, cmap="coolwarm", aspect="auto", vmin=-vabs, vmax=vabs)
ax.set_xticks(range(len(fc_df.columns)))
ax.set_xticklabels(fc_df.columns, fontsize=11)
ax.set_yticks(range(len(fc_df)))
ax.set_yticklabels(fc_df.index, fontsize=9)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("log\u2082 fold-change (Lecanemab / PBS)")
ax.set_title(f"Treatment fold-change per ROI \u2014 {cohort}", fontsize=13)
plt.tight_layout()
plt.savefig(output_figs["roi_fold_change"], dpi=150)
plt.close(fig)

density_all = roi_df.groupby("index", observed=True)["plaque_density"].mean()
vol_density = make_metric_volume(atlas_data, density_all)
plot_atlas_heatmap(
    vol_density,
    title=f"Mean plaque density \u2014 {cohort}",
    cmap="hot",
    cbar_label="Plaque density (n / mm\u00b3)",
    vmin=0,
    savepath=output_figs["atlas_density_all"],
)

group_vols = {}
for geno in geno_present:
    for treat in TREAT_ORDER:
        grp_density = (
            roi_df.loc[(roi_df["genotype"] == geno) & (roi_df["treatment"] == treat)]
            .groupby("index", observed=True)["plaque_density"]
            .mean()
        )
        group_vols[(geno, treat)] = make_metric_volume(atlas_data, grp_density)

valid_maxes = [
    float(np.nanmax(v)) for v in group_vols.values() if np.any(np.isfinite(v))
]
vmax_global = max(valid_maxes) if valid_maxes else 1.0

n_rows = n_geno
n_cols = len(TREAT_ORDER) * 3
fig, axes_grid = plt.subplots(
    n_rows, n_cols, figsize=(18, 5 * n_rows), squeeze=False
)
for gi, geno in enumerate(geno_present):
    for ti, treat in enumerate(TREAT_ORDER):
        v = group_vols[(geno, treat)]
        projs = [np.nanmax(v, axis=a) for a in range(3)]
        for pi, proj in enumerate(projs):
            ax = axes_grid[gi, ti * 3 + pi]
            masked = np.ma.masked_invalid(proj)
            im = ax.imshow(
                masked.T, origin="lower", cmap="hot",
                vmin=0, vmax=vmax_global, interpolation="nearest",
            )
            ax.axis("off")
            if pi == 1:
                ax.set_title(f"{geno} / {treat}", fontsize=10)
            if ti == len(TREAT_ORDER) - 1 and pi == 2:
                plt.colorbar(im, ax=ax, label="n / mm\u00b3", shrink=0.8)
fig.suptitle(
    f"Plaque density heatmaps by treatment \u00d7 genotype \u2014 {cohort}",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(output_figs["atlas_density_groups"], dpi=150)
plt.close(fig)

fc_vols = {}
for geno in geno_present:
    pbs_mean = (
        roi_df.loc[(roi_df["genotype"] == geno) & (roi_df["treatment"] == "PBS")]
        .groupby("index", observed=True)["plaque_density"]
        .mean()
    )
    lec_mean = (
        roi_df.loc[(roi_df["genotype"] == geno) & (roi_df["treatment"] == "Lecanemab")]
        .groupby("index", observed=True)["plaque_density"]
        .mean()
    )
    common = pbs_mean.index.intersection(lec_mean.index)
    log2fc = np.log2(
        (lec_mean.loc[common] + 1e-9) / (pbs_mean.loc[common] + 1e-9)
    )
    fc_vols[geno] = make_metric_volume(atlas_data, log2fc)

valid_abs = [
    float(np.nanmax(np.abs(v[np.isfinite(v)])))
    for v in fc_vols.values()
    if np.any(np.isfinite(v))
]
vabs_global = max(valid_abs) if valid_abs else 1.0

fig, axes_grid = plt.subplots(
    n_geno, 3, figsize=(14, 4 * n_geno), squeeze=False
)
for gi, geno in enumerate(geno_present):
    v = fc_vols[geno]
    projs = [np.nanmean(v, axis=a) for a in range(3)]
    for pi, proj in enumerate(projs):
        ax = axes_grid[gi, pi]
        masked = np.ma.masked_invalid(proj)
        im = ax.imshow(
            masked.T, origin="lower", cmap="coolwarm",
            vmin=-vabs_global, vmax=vabs_global, interpolation="nearest",
        )
        ax.axis("off")
        if pi == 1:
            ax.set_title(geno, fontsize=11)
    plt.colorbar(
        im, ax=axes_grid[gi, -1],
        label="log\u2082 FC (Lec / PBS)", shrink=0.8,
    )
fig.suptitle(
    f"Treatment fold-change (Lecanemab / PBS) \u2014 {cohort}", fontsize=14
)
plt.tight_layout()
plt.savefig(output_figs["atlas_fold_change"], dpi=150)
plt.close(fig)

diam_all = roi_df.groupby("index", observed=True)["mean_diam_um"].mean()
vol_diam = make_metric_volume(atlas_data, diam_all)
plot_atlas_heatmap(
    vol_diam,
    title=f"Mean plaque diameter \u2014 {cohort}",
    cmap="plasma",
    cbar_label="Mean equivalent diameter (\u00b5m)",
    vmin=0,
    savepath=output_figs["atlas_mean_diam"],
)

inside_all = roi_df.groupby("index", observed=True)["frac_inside_vessel"].mean()
vol_inside = make_metric_volume(atlas_data, inside_all)
plot_atlas_heatmap(
    vol_inside,
    title=f"Fraction of plaques inside vessels \u2014 {cohort}",
    cmap="viridis",
    cbar_label="Fraction inside vessel",
    vmin=0,
    savepath=output_figs["atlas_frac_inside"],
)
