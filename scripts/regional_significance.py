"""Regional significance visualizations from per-region treatment statistics.

Reads  : {cohort}_{seg}_roi_treatment_stats.csv   (from regional_analysis rule)
         atlas NIfTI  (tpl-ABAv3 segmentation)
Writes:
  fig_{cohort}_{seg}_atlas_significance.png
      Atlas projections coloured by log\u2082 fold-change;
      only FDR-significant regions are coloured (others are greyed out).
  fig_{cohort}_{seg}_regional_volcano.png
      Volcano plot: x = log\u2082 FC (Lecanemab / PBS),
                    y = \u2212log\u2081\u2080(FDR-adjusted p-value).
      Significant regions are highlighted and the top hits are labelled.
  fig_{cohort}_{seg}_significant_regions_bar.png
      Horizontal bar chart of FDR-significant regions sorted by effect size,
      with significance asterisks annotated on each bar.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

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
    input_stats = str(snakemake.input.stats_csv)  # noqa: F821
    input_atlas = str(snakemake.input.atlas)  # noqa: F821
    output_figs = dict(snakemake.output)  # noqa: F821
    cohort = snakemake.wildcards.cohort  # noqa: F821
    seg = snakemake.wildcards.seg  # noqa: F821
    cfg = snakemake.params.plot_config  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-csv", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--seg", required=True)
    args = parser.parse_args()
    input_stats = args.stats_csv
    input_atlas = args.atlas
    cohort = args.cohort
    seg = args.seg
    cfg = DEFAULT_PLOT_CONFIG
    output_figs = {
        "atlas_significance": (
            f"{args.output_dir}/fig_{cohort}_{seg}_atlas_significance.png"
        ),
        "regional_volcano": (
            f"{args.output_dir}/fig_{cohort}_{seg}_regional_volcano.png"
        ),
        "significant_regions_bar": (
            f"{args.output_dir}/fig_{cohort}_{seg}_significant_regions_bar.png"
        ),
    }

# ── Extract plot config ───────────────────────────────────────────────────────
treat_cfg = cfg["factors"]["treatment"]
TREAT_ORDER = treat_cfg["order"]
TREAT_PALETTE = treat_cfg["palette"]

GROUP1 = TREAT_ORDER[0]
GROUP2 = TREAT_ORDER[1]

# Fallback colour used when a category value is absent from the configured palette.
DEFAULT_COLOR = "#888888"

# ── Constants ────────────────────────────────────────────────────────────────
FDR_ALPHA = 0.05
N_LABEL = 10          # max labelled points in the volcano plot
N_BAR_MAX = 30        # max bars shown in the bar chart

SIG_CMAP = "coolwarm"   # diverging colormap for log2FC
GREY = "#BBBBBB"


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


def make_metric_volume(atlas_data, region_series):
    """Paint a metric onto the atlas volume via vectorised index lookup."""
    max_idx = int(atlas_data.max()) + 1
    lookup = np.full(max_idx, np.nan, dtype=np.float32)
    for ridx, val in region_series.items():
        if 0 <= int(ridx) < max_idx:
            lookup[int(ridx)] = float(val)
    vol = lookup[atlas_data]
    vol[atlas_data == 0] = np.nan
    return vol


# ── Load data ─────────────────────────────────────────────────────────────────
stats_df = pd.read_csv(input_stats)
atlas_img = nib.load(input_atlas)
atlas_data = atlas_img.get_fdata().astype(np.int32)

# Ensure required columns exist
required_cols = {"index", "name", "log2fc", "p_value_fdr"}
if not required_cols.issubset(stats_df.columns):
    raise ValueError(
        f"Stats CSV is missing columns. Expected: {required_cols}. "
        f"Got: {set(stats_df.columns)}"
    )

stats_df = stats_df.copy()
stats_df["significant"] = stats_df["p_value_fdr"] <= FDR_ALPHA
stats_df["stars"] = stats_df["p_value_fdr"].apply(pval_to_stars)
stats_df["neg_log10_fdr"] = -np.log10(stats_df["p_value_fdr"].clip(lower=1e-10))

sig_df = stats_df[stats_df["significant"]].copy()

# ── Figure 1: Atlas – FDR-significant regions coloured by log₂ FC ─────────
# Significant regions are shown with the coolwarm colormap (centred at 0).
# Non-significant regions are painted a neutral grey.
sig_fc = stats_df.set_index("index")["log2fc"].copy()
sig_fc_masked = sig_fc.where(stats_df.set_index("index")["significant"])

# Build two-layer atlas: grey background + coloured significant regions
vol_sig = make_metric_volume(atlas_data, sig_fc_masked)
vol_any = make_metric_volume(
    atlas_data,
    stats_df.set_index("index")["log2fc"].where(
        stats_df.set_index("index")["p_value_fdr"].notna()
    ),
)

vabs = float(np.nanmax(np.abs(sig_fc.values))) if sig_fc.notna().any() else 1.0
if not np.isfinite(vabs) or vabs == 0:
    vabs = 1.0

projs_sig = [np.nanmax(vol_sig, axis=a) for a in range(3)]
projs_any = [np.nanmax(vol_any, axis=a) for a in range(3)]
view_names = ["Coronal", "Sagittal", "Axial"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, proj_sig, proj_any, pname in zip(axes, projs_sig, projs_any, view_names):
    # Grey background for all tested regions
    grey_bg = np.where(np.isfinite(proj_any), 0.5, np.nan)
    ax.imshow(
        np.ma.masked_invalid(grey_bg).T,
        origin="lower", cmap="Greys",
        vmin=0, vmax=1, interpolation="nearest",
    )
    # Coloured overlay for significant regions
    im = ax.imshow(
        np.ma.masked_invalid(proj_sig).T,
        origin="lower", cmap=SIG_CMAP,
        vmin=-vabs, vmax=vabs, interpolation="nearest",
    )
    ax.set_title(pname, fontsize=10)
    ax.axis("off")

cbar = plt.colorbar(im, ax=axes[-1], shrink=0.8)
cbar.set_label(f"log\u2082 FC ({GROUP2} / {GROUP1})")
n_sig = int(stats_df["significant"].sum())
fig.suptitle(
    f"FDR-significant regions (n\u2009=\u2009{n_sig}, FDR\u2009<\u2009{FDR_ALPHA})"
    f" \u2014 {cohort} \u2022 {seg}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["atlas_significance"], dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Volcano plot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Non-significant: grey
ns_mask = ~stats_df["significant"] & stats_df["p_value_fdr"].notna()
ax.scatter(
    stats_df.loc[ns_mask, "log2fc"],
    stats_df.loc[ns_mask, "neg_log10_fdr"],
    color=GREY, alpha=0.5, s=20, linewidths=0, label="Not significant",
)

# Significant decrease (group2 < group1): colour of group1
dec_mask = stats_df["significant"] & (stats_df["log2fc"] < 0)
ax.scatter(
    stats_df.loc[dec_mask, "log2fc"],
    stats_df.loc[dec_mask, "neg_log10_fdr"],
    color=TREAT_PALETTE.get(GROUP1, DEFAULT_COLOR), alpha=0.85, s=45, linewidths=0,
    label=f"Significant decrease ({GROUP2} < {GROUP1})",
    zorder=3,
)

# Significant increase (group2 > group1): colour of group2
inc_mask = stats_df["significant"] & (stats_df["log2fc"] >= 0)
ax.scatter(
    stats_df.loc[inc_mask, "log2fc"],
    stats_df.loc[inc_mask, "neg_log10_fdr"],
    color=TREAT_PALETTE.get(GROUP2, DEFAULT_COLOR), alpha=0.85, s=45, linewidths=0,
    label=f"Significant increase ({GROUP2} > {GROUP1})",
    zorder=3,
)

# Label top hits (by FDR)
top_hits = (
    sig_df.nsmallest(N_LABEL, "p_value_fdr")
    if len(sig_df) > 0
    else pd.DataFrame()
)
for _, row in top_hits.iterrows():
    ax.annotate(
        row["name"],
        xy=(row["log2fc"], row["neg_log10_fdr"]),
        xytext=(6, 2), textcoords="offset points",
        fontsize=7, va="center",
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
    )

# Threshold line
threshold_y = -np.log10(FDR_ALPHA)
ax.axhline(threshold_y, color="red", linestyle="--", linewidth=0.9,
           label=f"FDR = {FDR_ALPHA}")
ax.axvline(0, color="black", linestyle="--", linewidth=0.7, alpha=0.5)

ax.set_xlabel(f"log\u2082 fold-change ({GROUP2} / {GROUP1})", fontsize=11)
ax.set_ylabel("\u2212log\u2081\u2080 (FDR-adjusted p-value)", fontsize=11)
ax.set_title(
    f"Volcano plot \u2014 per-region treatment effect \u2014 {cohort} \u2022 {seg}",
    fontsize=13,
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(output_figs["regional_volcano"], dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: Significant regions bar chart ───────────────────────────────────
if len(sig_df) == 0:
    # No significant regions: output a placeholder figure
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(
        0.5, 0.5,
        "No FDR-significant regions found",
        ha="center", va="center", fontsize=14, color="gray",
        transform=ax.transAxes,
    )
    ax.axis("off")
    fig.suptitle(
        f"Significant regions \u2014 {cohort} \u2022 {seg}", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(
        output_figs["significant_regions_bar"], dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
else:
    # Sort by log2FC (most negative = strongest Lecanemab reduction first)
    plot_df = sig_df.sort_values("log2fc").head(N_BAR_MAX).reset_index(drop=True)

    colors = [
        TREAT_PALETTE.get(GROUP1, DEFAULT_COLOR) if fc < 0
        else TREAT_PALETTE.get(GROUP2, DEFAULT_COLOR)
        for fc in plot_df["log2fc"]
    ]

    fig_h = max(4, 0.38 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    bars = ax.barh(
        plot_df["name"],
        plot_df["log2fc"],
        color=colors, edgecolor="white", height=0.7,
    )

    # Annotate each bar with significance stars
    for bar, stars in zip(bars, plot_df["stars"]):
        bar_w = bar.get_width()
        x_pos = bar_w + (0.02 if bar_w >= 0 else -0.02)
        ha = "left" if bar_w >= 0 else "right"
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2,
            stars, ha=ha, va="center", fontsize=10, color="black",
            fontweight="bold",
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"log\u2082 fold-change ({GROUP2} / {GROUP1})", fontsize=11)
    ax.set_title(
        f"FDR-significant regions (n\u2009=\u2009{len(sig_df)}) \u2014 "
        f"{cohort} \u2022 {seg}\n"
        "* p\u22640.05  ** p\u22640.01  *** p\u22640.001  **** p\u22640.0001",
        fontsize=12,
    )
    ax.invert_yaxis()

    # Legend for colours
    from matplotlib.patches import Patch  # noqa: PLC0415

    legend_elements = [
        Patch(facecolor=TREAT_PALETTE.get(GROUP1, DEFAULT_COLOR),
              label=f"{GROUP2} < {GROUP1} (reduction)"),
        Patch(facecolor=TREAT_PALETTE.get(GROUP2, DEFAULT_COLOR),
              label=f"{GROUP2} > {GROUP1} (increase)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        output_figs["significant_regions_bar"], dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
