"""Per-region confidence-interval visualizations for treatment effects.

Appropriate for small-N (2-3 subjects per group) experiments where
FDR-based inference has very low power.  Uses bootstrap CIs on log₂
fold-change to characterise per-region treatment effects.

Reads  : roi_data_{cohort}_seg-{seg}.parquet
         atlas NIfTI  (tpl-ABAv3 segmentation)
Writes:
  fig_{cohort}_{seg}_regional_ci_forest.png
      Forest plot of per-region log₂ FC (Lecanemab / PBS) with 95 %
      bootstrap CI.  Regions where the CI excludes zero are highlighted
      and labelled; all regions with enough data are shown sorted by
      effect size.
  fig_{cohort}_{seg}_atlas_ci.png
      Three-panel atlas figure.  Left panel: point estimate of log₂ FC
      painted on the atlas (all tested regions).  Middle panel: CI width
      (hi − lo) as an uncertainty map.  Right panel: regions where CI
      excludes zero are highlighted (coloured by direction).
  {cohort}_{seg}_roi_ci_table.csv
      Per-region CI summary: n per group, mean PBS/Lec, log₂ FC,
      bootstrap CI bounds, and a flag for CI-excludes-zero.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

# ── Snakemake integration ────────────────────────────────────────────────────
if "snakemake" in dir():
    input_roi = str(snakemake.input.roi_parquet)  # noqa: F821
    input_atlas = str(snakemake.input.atlas)  # noqa: F821
    output_figs = dict(snakemake.output)  # noqa: F821
    cohort = snakemake.wildcards.cohort  # noqa: F821
    seg = snakemake.wildcards.seg  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-parquet", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--seg", required=True)
    args = parser.parse_args()
    input_roi = args.roi_parquet
    input_atlas = args.atlas
    cohort = args.cohort
    seg = args.seg
    output_figs = {
        "regional_ci_forest": (
            f"{args.output_dir}/fig_{cohort}_{seg}_regional_ci_forest.png"
        ),
        "atlas_ci": (
            f"{args.output_dir}/fig_{cohort}_{seg}_atlas_ci.png"
        ),
        "regional_ci_table": (
            f"{args.output_dir}/{cohort}_{seg}_roi_ci_table.csv"
        ),
    }

# ── Constants ────────────────────────────────────────────────────────────────
METRIC = "total_vol_ml"
CI_LEVEL = 0.95
N_BOOT = 5_000
RNG_SEED = 42
N_FOREST_MAX = 40        # max regions shown in forest plot
PSEUDOCOUNT = 1e-9   # added to numerator and denominator before log ratios to avoid log(0)
GREY = "#BBBBBB"
CI_CMAP = "coolwarm"     # diverging for log2FC
WIDTH_CMAP = "YlOrRd"    # sequential for CI width (uncertainty)


# ── Helper functions ─────────────────────────────────────────────────────────
def bootstrap_log2fc_ci(pbs_vals, lec_vals, n_boot=N_BOOT, level=CI_LEVEL,
                        rng=None):
    """Percentile bootstrap CI for log₂(mean_lec / mean_pbs).

    Returns (point_estimate, ci_lo, ci_hi).  When n=1 per group the CI
    is returned as NaN (cannot bootstrap a single observation).
    """
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    n_pbs = len(pbs_vals)
    n_lec = len(lec_vals)
    mean_pbs = float(np.mean(pbs_vals)) if n_pbs > 0 else np.nan
    mean_lec = float(np.mean(lec_vals)) if n_lec > 0 else np.nan
    if n_pbs < 1 or n_lec < 1 or np.isnan(mean_pbs) or np.isnan(mean_lec):
        return np.nan, np.nan, np.nan
    point = np.log2((mean_lec + PSEUDOCOUNT) / (mean_pbs + PSEUDOCOUNT))
    if n_pbs == 1 and n_lec == 1:
        return point, np.nan, np.nan
    boot = np.empty(n_boot)
    for i in range(n_boot):
        b_pbs = rng.choice(pbs_vals, size=n_pbs, replace=True)
        b_lec = rng.choice(lec_vals, size=n_lec, replace=True)
        boot[i] = np.log2(
            (np.mean(b_lec) + PSEUDOCOUNT) / (np.mean(b_pbs) + PSEUDOCOUNT)
        )
    alpha = 1.0 - level
    lo = float(np.nanpercentile(boot, 100 * alpha / 2))
    hi = float(np.nanpercentile(boot, 100 * (1 - alpha / 2)))
    return point, lo, hi


def make_metric_volume(atlas_data, region_series):
    """Paint a per-region metric onto the atlas volume."""
    max_idx = int(atlas_data.max()) + 1
    lookup = np.full(max_idx, np.nan, dtype=np.float32)
    for ridx, val in region_series.items():
        if 0 <= int(ridx) < max_idx:
            lookup[int(ridx)] = float(val)
    vol = lookup[atlas_data]
    vol[atlas_data == 0] = np.nan
    return vol


# ── Load data ─────────────────────────────────────────────────────────────────
roi_df = pd.read_parquet(input_roi)
atlas_img = nib.load(input_atlas)
atlas_data = atlas_img.get_fdata().astype(np.int32)

rng = np.random.default_rng(RNG_SEED)

# ── Compute per-region bootstrap CIs ─────────────────────────────────────────
ci_rows = []
for (ridx, name), grp in roi_df.groupby(["index", "name"], observed=True):
    pbs = grp.loc[grp["treatment"] == "PBS", METRIC].dropna().values
    lec = grp.loc[grp["treatment"] == "Lecanemab", METRIC].dropna().values
    n_pbs, n_lec = len(pbs), len(lec)
    mean_pbs = float(np.mean(pbs)) if n_pbs > 0 else np.nan
    mean_lec = float(np.mean(lec)) if n_lec > 0 else np.nan
    log2fc, ci_lo, ci_hi = bootstrap_log2fc_ci(pbs, lec, rng=rng)
    ci_width = (ci_hi - ci_lo) if (np.isfinite(ci_lo) and np.isfinite(ci_hi)) else np.nan
    ci_excl = bool(ci_lo > 0 or ci_hi < 0) if (np.isfinite(ci_lo) and np.isfinite(ci_hi)) else False

    ci_rows.append({
        "index": ridx,
        "name": name,
        "n_PBS": n_pbs,
        "n_Lecanemab": n_lec,
        "mean_PBS": mean_pbs,
        "mean_Lecanemab": mean_lec,
        "log2fc_Lec_over_PBS": log2fc,
        "ci_lo_log2fc": ci_lo,
        "ci_hi_log2fc": ci_hi,
        "ci_width": ci_width,
        "ci_excludes_zero": ci_excl,
    })

ci_df = pd.DataFrame(ci_rows).sort_values("log2fc_Lec_over_PBS")
ci_df.to_csv(output_figs["regional_ci_table"], index=False)

# ── Subset for forest plot ────────────────────────────────────────────────────
# Show regions that have a finite point estimate, sorted by effect size.
# Prefer regions where CI excludes zero; if none, show top N by |log2FC|.
plot_df = ci_df[ci_df["log2fc_Lec_over_PBS"].notna()].copy()
excl_df = plot_df[plot_df["ci_excludes_zero"]].copy()

if len(excl_df) >= 4:
    # Show all CI-excludes-zero regions plus fill up to N_FOREST_MAX with
    # the largest remaining effects.
    rest = plot_df[~plot_df["ci_excludes_zero"]].copy()
    rest = rest.reindex(rest["log2fc_Lec_over_PBS"].abs().sort_values(ascending=False).index)
    n_fill = max(0, N_FOREST_MAX - len(excl_df))
    forest_df = pd.concat([excl_df, rest.head(n_fill)], ignore_index=True)
    forest_df = forest_df.sort_values("log2fc_Lec_over_PBS").reset_index(drop=True)
else:
    # Not enough CI-excludes-zero regions: show top by |log2FC|
    forest_df = plot_df.copy()
    forest_df = forest_df.reindex(
        forest_df["log2fc_Lec_over_PBS"].abs().sort_values(ascending=False).index
    ).head(N_FOREST_MAX).sort_values("log2fc_Lec_over_PBS").reset_index(drop=True)

# ── Figure 1: Forest plot ─────────────────────────────────────────────────────
n_rows = len(forest_df)
if n_rows == 0:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5, "Insufficient data for forest plot",
            ha="center", va="center", fontsize=13, color="gray",
            transform=ax.transAxes)
    ax.axis("off")
    fig.suptitle(f"Regional CI forest plot \u2014 {cohort} \u2022 {seg}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_figs["regional_ci_forest"], dpi=150, bbox_inches="tight")
    plt.close(fig)
else:
    fig_h = max(4.0, 0.32 * n_rows + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y_pos = np.arange(n_rows)

    for i, (_, row) in enumerate(forest_df.iterrows()):
        has_ci = np.isfinite(row["ci_lo_log2fc"]) and np.isfinite(row["ci_hi_log2fc"])
        ci_excl = bool(row["ci_excludes_zero"])
        # colour scheme: orange = CI excludes 0, grey = CI spans 0, darker = no CI
        if ci_excl:
            color = "#DD8452" if row["log2fc_Lec_over_PBS"] > 0 else "#4C72B0"
        elif has_ci:
            color = GREY
        else:
            color = "#999999"

        if has_ci:
            ax.plot(
                [row["ci_lo_log2fc"], row["ci_hi_log2fc"]], [i, i],
                color=color, lw=1.8, zorder=2,
            )
            ax.plot([row["ci_lo_log2fc"]] * 2, [i - 0.12, i + 0.12],
                    color=color, lw=1.8, zorder=2)
            ax.plot([row["ci_hi_log2fc"]] * 2, [i - 0.12, i + 0.12],
                    color=color, lw=1.8, zorder=2)

        ax.scatter(
            [row["log2fc_Lec_over_PBS"]], [i],
            color=color, s=55, zorder=3,
            edgecolors="white", linewidths=0.6,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(forest_df["name"].tolist(), fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("log\u2082 fold-change (Lecanemab / PBS)", fontsize=11)
    ax.set_title(
        f"Per-region treatment effect (log\u2082 FC) \u2014 {cohort} \u2022 {seg}\n"
        f"{int(CI_LEVEL * 100)}\u2009% bootstrap CI  \u2022  "
        f"top {n_rows} regions by |log\u2082 FC|  \u2022  "
        "coloured = CI excludes zero",
        fontsize=11,
    )
    ax.grid(axis="x", alpha=0.35)

    legend_els = [
        Patch(facecolor="#DD8452", label="CI excludes 0 (increase)"),
        Patch(facecolor="#4C72B0", label="CI excludes 0 (decrease)"),
        Patch(facecolor=GREY, label="CI spans 0 (uncertain)"),
    ]
    ax.legend(handles=legend_els, fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_figs["regional_ci_forest"], dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Figure 2: Atlas CI panels ─────────────────────────────────────────────────
# Panel 1 (left)  : all-region log₂ FC point estimate (grey = tested, color = value).
# Panel 2 (middle): CI width (uncertainty map; only regions with finite CI).
# Panel 3 (right) : CI-excludes-zero map (blue = decrease, orange = increase).

fc_series = ci_df.set_index("index")["log2fc_Lec_over_PBS"]
ci_width_series = ci_df.set_index("index")["ci_width"]
excl_series = ci_df.set_index("index").apply(
    lambda r: r["log2fc_Lec_over_PBS"] if r["ci_excludes_zero"] else np.nan, axis=1
)

vol_fc = make_metric_volume(atlas_data, fc_series)
vol_width = make_metric_volume(atlas_data, ci_width_series)
vol_excl = make_metric_volume(atlas_data, excl_series)

# For the grey background (all tested regions)
vol_tested = make_metric_volume(
    atlas_data,
    ci_df.set_index("index")["log2fc_Lec_over_PBS"].where(
        ci_df.set_index("index")["log2fc_Lec_over_PBS"].notna()
    ),
)

vabs_fc = float(np.nanmax(np.abs(fc_series.values))) if fc_series.notna().any() else 1.0
if not np.isfinite(vabs_fc) or vabs_fc == 0:
    vabs_fc = 1.0

vabs_excl = float(np.nanmax(np.abs(excl_series.values))) if excl_series.notna().any() else 1.0
if not np.isfinite(vabs_excl) or vabs_excl == 0:
    vabs_excl = 1.0

vmax_width = float(np.nanpercentile(ci_width_series.dropna().values, 95)) if ci_width_series.notna().any() else 1.0
if not np.isfinite(vmax_width) or vmax_width == 0:
    vmax_width = 1.0

# Use coronal projection (axis 0) for all panels
proj_fc = np.nanmax(vol_fc, axis=0)
proj_width = np.nanmax(vol_width, axis=0)
proj_excl = np.nanmax(vol_excl, axis=0)
proj_tested = np.nanmax(vol_tested, axis=0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: log₂ FC (all tested regions, grey bg)
ax = axes[0]
grey_bg = np.where(np.isfinite(proj_tested), 0.5, np.nan)
ax.imshow(np.ma.masked_invalid(grey_bg).T, origin="lower",
          cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
im1 = ax.imshow(np.ma.masked_invalid(proj_fc).T, origin="lower",
                cmap=CI_CMAP, vmin=-vabs_fc, vmax=vabs_fc,
                interpolation="nearest")
plt.colorbar(im1, ax=ax, shrink=0.8).set_label("log\u2082 FC")
ax.set_title("log\u2082 FC (Lecanemab / PBS)\n(all tested regions)", fontsize=10)
ax.axis("off")

# Panel 2: CI width (uncertainty)
ax = axes[1]
ax.imshow(np.ma.masked_invalid(grey_bg).T, origin="lower",
          cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
im2 = ax.imshow(np.ma.masked_invalid(proj_width).T, origin="lower",
                cmap=WIDTH_CMAP, vmin=0, vmax=vmax_width,
                interpolation="nearest")
plt.colorbar(im2, ax=ax, shrink=0.8).set_label(f"{int(CI_LEVEL * 100)}\u2009% CI width")
ax.set_title(f"Bootstrap CI width\n(uncertainty, {int(CI_LEVEL * 100)}\u2009%)", fontsize=10)
ax.axis("off")

# Panel 3: regions where CI excludes zero
ax = axes[2]
ax.imshow(np.ma.masked_invalid(grey_bg).T, origin="lower",
          cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
im3 = ax.imshow(np.ma.masked_invalid(proj_excl).T, origin="lower",
                cmap=CI_CMAP, vmin=-vabs_excl, vmax=vabs_excl,
                interpolation="nearest")
plt.colorbar(im3, ax=ax, shrink=0.8).set_label("log\u2082 FC")
n_excl = int(ci_df["ci_excludes_zero"].sum())
ax.set_title(
    f"CI excludes zero (n\u2009=\u2009{n_excl} regions)\n"
    "orange/blue = increase/decrease",
    fontsize=10,
)
ax.axis("off")

fig.suptitle(
    f"Per-region CI atlas \u2014 {cohort} \u2022 {seg}  "
    f"(metric: {METRIC})",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["atlas_ci"], dpi=150, bbox_inches="tight")
plt.close(fig)
