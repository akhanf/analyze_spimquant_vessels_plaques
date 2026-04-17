"""3-D KDE plots showing plaque burden (plaque locations weighted by equiv. diam.).

Reads  : data_{cohort}.parquet
Writes (one file per figure):
  fig_{cohort}_kde_plaque_burden.png        KDE projections (XY and XZ) by genotype,
                                             with PBS and Lecanemab overlaid to show
                                             the treatment effect.
  fig_{cohort}_kde_plaque_burden_zslices.png 20 XY density slices along the Z axis
                                             (equally spaced in 1/20 increments),
                                             per genotype with treatments overlaid.

Method
------
A weighted 3-D Gaussian KDE is computed per group using
``scipy.stats.gaussian_kde`` with ``weights=equiv_diam_um``, so larger plaques
contribute proportionally more to the density estimate ("plaque burden").
The 3-D density is evaluated on a regular grid and then either marginalised
(summed) along the Z or Y axis to produce XY and XZ projections, or sliced at
20 equally-spaced Z positions to show the dorsal–ventral depth structure.
"""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats  # noqa: E402
import seaborn as sns  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Snakemake integration ────────────────────────────────────────────────────
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
        "kde_plaque_burden": f"{args.output_dir}/fig_{cohort}_kde_plaque_burden.png",
        "kde_plaque_burden_zslices": f"{args.output_dir}/fig_{cohort}_kde_plaque_burden_zslices.png",
    }

# ── Extract plot config ───────────────────────────────────────────────────────
sns.set_theme(style="white", font_scale=1.1)

treat_cfg = cfg["factors"]["treatment"]
TREAT_COL = treat_cfg["column"]
TREAT_ORDER = treat_cfg["order"]
TREAT_PALETTE = treat_cfg["palette"]

geno_cfg = cfg["factors"]["genotype"]
GENO_COL = geno_cfg["column"]
GENO_ORDER = geno_cfg["order"]

VOL_THRESH_ML = cfg.get("volume_threshold_ml", 1e-4)

treat_label = TREAT_COL.replace("_", " ").title()
geno_label = GENO_COL.replace("_", " ").title()

# KDE/grid parameters
N_KDE_MAX = 12_000   # max plaques per group for KDE fitting (balances accuracy vs runtime)
GRID_N_XY = 45       # grid resolution for X and Y axes
GRID_N_Z = 20        # 20 equally-spaced Z levels (each level = one Z-slice in the slices figure)
KDE_EVAL_BATCH = 20_000   # batch size for KDE grid evaluation (controls peak memory use)
MIN_PLAQUES_FOR_KDE = 20  # minimum plaques in a group to attempt KDE fitting
CONTOUR_PERCENTILES = [70, 85, 95]  # density percentiles at which to draw contour lines

# Z-slices figure layout (GRID_N_Z slices arranged as ZSLICE_NROWS × ZSLICE_NCOLS)
ZSLICE_NCOLS = 5
ZSLICE_NROWS = GRID_N_Z // ZSLICE_NCOLS  # 4 rows × 5 cols = 20 panels

PROJECTIONS = [
    ("template_x", "template_y", "z", "XY projection (A\u2013P \u00d7 M\u2013L)"),
    ("template_x", "template_z", "y", "XZ projection (A\u2013P \u00d7 D\u2013V)"),
]


# ── Transparent colormaps per treatment ─────────────────────────────────────
def _alpha_cmap(hex_color, name):
    """Single-hue colormap with alpha going from 0 (low density) to 0.85 (high)."""
    r, g, b = mcolors.to_rgb(hex_color)
    cmap_data = {
        "red":   [(0.0, r, r), (1.0, r, r)],
        "green": [(0.0, g, g), (1.0, g, g)],
        "blue":  [(0.0, b, b), (1.0, b, b)],
        "alpha": [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (1.0, 0.85, 0.85)],
    }
    return mcolors.LinearSegmentedColormap(name, cmap_data)


TREAT_CMAP = {
    treat: _alpha_cmap(color, f"cmap_treat_{i}")
    for i, (treat, color) in enumerate(TREAT_PALETTE.items())
}


# ── Helper: 3-D KDE evaluation ───────────────────────────────────────────────
def compute_kde_3d(x, y, z, weights, x_lim, y_lim, z_lim, rng_seed=42):
    """Fit a weighted 3-D KDE and evaluate it on a regular grid.

    Parameters
    ----------
    x, y, z : 1-D arrays of plaque template coordinates (mm).
    weights  : 1-D array of equiv_diam_um values (positive, not yet normalised).
    x_lim, y_lim, z_lim : (min, max) tuples that define the evaluation grid.
    rng_seed : random seed used when subsampling.

    Returns
    -------
    dict with keys:
        ``density_3d``  : ndarray, shape (GRID_N_XY, GRID_N_XY, GRID_N_Z)
        ``xi``, ``yi``, ``zi`` : 1-D coordinate arrays for each axis.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Subsample if group is too large for practical KDE fitting
    n = len(x)
    if n > N_KDE_MAX:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(n, N_KDE_MAX, replace=False)
        x, y, z, weights = x[idx], y[idx], z[idx], weights[idx]

    weights = weights / weights.sum()  # normalise

    # Build 3-D KDE
    pts = np.vstack([x, y, z])
    kde = scipy.stats.gaussian_kde(pts, weights=weights)

    # Define evaluation grids
    xi = np.linspace(x_lim[0], x_lim[1], GRID_N_XY)
    yi = np.linspace(y_lim[0], y_lim[1], GRID_N_XY)
    zi = np.linspace(z_lim[0], z_lim[1], GRID_N_Z)

    # Build full 3-D evaluation grid (indexing='ij' → shape (nx, ny, nz))
    gx3, gy3, gz3 = np.meshgrid(xi, yi, zi, indexing="ij")
    eval_pts = np.vstack([gx3.ravel(), gy3.ravel(), gz3.ravel()])

    # Evaluate in batches to avoid very large array allocations
    n_eval = eval_pts.shape[1]
    density_flat = np.empty(n_eval, dtype=float)
    for start in range(0, n_eval, KDE_EVAL_BATCH):
        end = min(start + KDE_EVAL_BATCH, n_eval)
        density_flat[start:end] = kde(eval_pts[:, start:end])

    density_3d = density_flat.reshape(GRID_N_XY, GRID_N_XY, GRID_N_Z)

    return {"density_3d": density_3d, "xi": xi, "yi": yi, "zi": zi}


def kde_projections(kde_data):
    """Derive marginalised XY and XZ 2-D projections from a ``compute_kde_3d`` result.

    Parameters
    ----------
    kde_data : dict as returned by ``compute_kde_3d``.

    Returns
    -------
    dict with keys ``"XY"`` and ``"XZ"``.
        Each value is ``{"gx": ..., "gy": ..., "density": ...}``.
    """
    density_3d = kde_data["density_3d"]
    xi = kde_data["xi"]
    yi = kde_data["yi"]
    zi = kde_data["zi"]

    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]

    # XY projection: marginalise over Z (axis=2)
    xy_density = density_3d.sum(axis=2) * dz
    gx_xy, gy_xy = np.meshgrid(xi, yi, indexing="ij")

    # XZ projection: marginalise over Y (axis=1)
    xz_density = density_3d.sum(axis=1) * dy
    gx_xz, gz_xz = np.meshgrid(xi, zi, indexing="ij")

    return {
        "XY": {"gx": gx_xy, "gy": gy_xy, "density": xy_density},
        "XZ": {"gx": gx_xz, "gy": gz_xz, "density": xz_density},
    }


# ── Helper: render two overlaid treatment densities onto an axes ─────────────
def _render_density_overlay(ax, treat_data, treat_present, proj_key, vmax=None):
    """Draw pcolormesh + contours for each treatment on *ax*.

    Parameters
    ----------
    ax        : matplotlib Axes.
    treat_data : dict mapping treatment → projection dict (from ``kde_projections``).
    treat_present : ordered list of treatment labels present in *treat_data*.
    proj_key  : ``"XY"`` or ``"XZ"``.
    vmax      : shared density scale; auto-computed if None.
    """
    if vmax is None:
        vmax = max(
            treat_data[t][proj_key]["density"].max()
            for t in treat_present if t in treat_data
        )

    for treat in treat_present:
        if treat not in treat_data:
            continue
        proj = treat_data[treat][proj_key]
        gx, gy, dens = proj["gx"], proj["gy"], proj["density"]
        dens_norm = dens / vmax if vmax > 0 else dens

        ax.pcolormesh(
            gx, gy, dens_norm,
            cmap=TREAT_CMAP[treat],
            shading="gouraud",
            rasterized=True,
            vmin=0.0, vmax=1.0,
        )
        levels = np.percentile(dens_norm[dens_norm > 0], CONTOUR_PERCENTILES)
        levels = np.unique(levels)
        if len(levels) >= 2:
            ax.contour(
                gx, gy, dens_norm,
                levels=levels,
                colors=[TREAT_PALETTE[treat]],
                linewidths=0.8,
                alpha=0.9,
            )


# ── Load and prepare data ────────────────────────────────────────────────────
df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()

geno_present = [g for g in GENO_ORDER if (df[GENO_COL] == g).any()]
treat_present = [t for t in TREAT_ORDER if (df[TREAT_COL] == t).any()]

df[GENO_COL] = pd.Categorical(df[GENO_COL], categories=geno_present, ordered=True)
df[TREAT_COL] = pd.Categorical(df[TREAT_COL], categories=treat_present, ordered=True)

# Shared axis limits derived from entire dataset (consistent across groups)
x_lim = (df["template_x"].quantile(0.002), df["template_x"].quantile(0.998))
y_lim = (df["template_y"].quantile(0.002), df["template_y"].quantile(0.998))
z_lim = (df["template_z"].quantile(0.002), df["template_z"].quantile(0.998))

# ── Pre-compute 3-D KDE for every (genotype, treatment) group ────────────────
all_kde = {}  # {(geno, treat): {"density_3d": ..., "xi": ..., "yi": ..., "zi": ...}}
for geno in geno_present:
    for treat in treat_present:
        mask = (df[GENO_COL] == geno) & (df[TREAT_COL] == treat)
        tdf = df.loc[mask]
        if len(tdf) < MIN_PLAQUES_FOR_KDE:
            continue
        all_kde[(geno, treat)] = compute_kde_3d(
            tdf["template_x"].values,
            tdf["template_y"].values,
            tdf["template_z"].values,
            tdf["equiv_diam_um"].values,
            x_lim=x_lim,
            y_lim=y_lim,
            z_lim=z_lim,
        )

# ── Figure 1: XY and XZ projections (marginalised over Z / Y) ────────────────
# Layout: rows = genotypes, cols = projections (XY, XZ)
n_genos = len(geno_present)
n_proj = len(PROJECTIONS)

fig, axes = plt.subplots(
    n_genos, n_proj,
    figsize=(6.5 * n_proj, 5.5 * n_genos),
    squeeze=False,
)

for row_idx, geno in enumerate(geno_present):
    # Build projection dicts for each treatment in this genotype
    treat_projs = {
        treat: kde_projections(all_kde[(geno, treat)])
        for treat in treat_present if (geno, treat) in all_kde
    }

    for col_idx, (xcol, ycol, marginal_axis, proj_label) in enumerate(PROJECTIONS):
        proj_key = "XY" if marginal_axis == "z" else "XZ"
        ax = axes[row_idx, col_idx]

        if not treat_projs:
            ax.set_visible(False)
            continue

        all_densities = [treat_projs[t][proj_key]["density"] for t in treat_present if t in treat_projs]
        vmax = max(d.max() for d in all_densities)
        _render_density_overlay(ax, treat_projs, treat_present, proj_key, vmax=vmax)

        ax.set_aspect("equal")
        ax.set_xlabel(xcol.replace("template_", "").upper() + " (mm)", fontsize=9)
        ax.set_ylabel(ycol.replace("template_", "").upper() + " (mm)", fontsize=9)
        ax.set_title(f"{geno} \u2014 {proj_label}", fontsize=10)

        # Legend patches (only on the first row, last column)
        if row_idx == 0 and col_idx == n_proj - 1:
            legend_handles = [
                Patch(facecolor=TREAT_PALETTE[t], alpha=0.75, label=t)
                for t in treat_present
            ]
            ax.legend(
                handles=legend_handles,
                title=treat_label,
                loc="upper right",
                fontsize=9,
            )

fig.suptitle(
    f"Plaque burden KDE (weighted by equiv. diameter) \u2014 {cohort}",
    fontsize=14,
)
plt.tight_layout()
plt.savefig(output_figs["kde_plaque_burden"], dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: 20 XY slices along the Z axis per genotype ────────────────────
# Layout: each genotype gets a block of ZSLICE_NROWS × ZSLICE_NCOLS subplots.
# All genotype blocks are stacked vertically in a single figure.

# Shared vmax across all groups and slices (for consistent colour scale)
slice_vmax = max(
    data["density_3d"].max() for data in all_kde.values()
)

fig2, axes2 = plt.subplots(
    n_genos * ZSLICE_NROWS,
    ZSLICE_NCOLS,
    figsize=(3.5 * ZSLICE_NCOLS, 3.5 * ZSLICE_NROWS * n_genos),
    squeeze=False,
)

# Use the Z grid from the first available KDE (all share the same grid)
zi_ref = next(iter(all_kde.values()))["zi"]
xi_ref = next(iter(all_kde.values()))["xi"]
yi_ref = next(iter(all_kde.values()))["yi"]
gx_slice, gy_slice = np.meshgrid(xi_ref, yi_ref, indexing="ij")

for geno_idx, geno in enumerate(geno_present):
    row_offset = geno_idx * ZSLICE_NROWS

    for slice_idx in range(GRID_N_Z):
        row = row_offset + slice_idx // ZSLICE_NCOLS
        col = slice_idx % ZSLICE_NCOLS
        ax = axes2[row, col]

        z_val = zi_ref[slice_idx]

        for treat in treat_present:
            if (geno, treat) not in all_kde:
                continue
            dens_slice = all_kde[(geno, treat)]["density_3d"][:, :, slice_idx]
            dens_norm = dens_slice / slice_vmax if slice_vmax > 0 else dens_slice

            ax.pcolormesh(
                gx_slice, gy_slice, dens_norm,
                cmap=TREAT_CMAP[treat],
                shading="gouraud",
                rasterized=True,
                vmin=0.0, vmax=1.0,
            )
            levels = np.percentile(dens_norm[dens_norm > 0], CONTOUR_PERCENTILES)
            levels = np.unique(levels)
            if len(levels) >= 2:
                ax.contour(
                    gx_slice, gy_slice, dens_norm,
                    levels=levels,
                    colors=[TREAT_PALETTE[treat]],
                    linewidths=0.6,
                    alpha=0.9,
                )

        ax.set_aspect("equal")
        ax.set_title(f"z = {z_val:.2f} mm", fontsize=7)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel("Y (mm)", fontsize=7)
        if row == row_offset + ZSLICE_NROWS - 1:
            ax.set_xlabel("X (mm)", fontsize=7)

    # Genotype label on the leftmost panel of the first row of each block
    axes2[row_offset, 0].annotate(
        geno,
        xy=(0, 0.5), xycoords="axes fraction",
        xytext=(-0.35, 0.5), textcoords="axes fraction",
        fontsize=9, fontweight="bold", va="center", ha="right",
        rotation=90,
    )

# Shared legend (top-right of the figure)
legend_handles = [
    Patch(facecolor=TREAT_PALETTE[t], alpha=0.75, label=t)
    for t in treat_present
]
fig2.legend(
    handles=legend_handles,
    title=treat_label,
    loc="upper right",
    fontsize=9,
    bbox_to_anchor=(1.0, 1.0),
)

fig2.suptitle(
    f"Plaque burden KDE \u2014 Z-axis slices (1/{GRID_N_Z} increments) \u2014 {cohort}",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(output_figs["kde_plaque_burden_zslices"], dpi=150, bbox_inches="tight")
plt.close(fig2)
