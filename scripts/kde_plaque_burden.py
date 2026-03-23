"""3-D KDE plots showing plaque burden (plaque locations weighted by equiv. diam.).

Reads  : data_{cohort}.parquet
Writes (one file per figure):
  fig_{cohort}_kde_plaque_burden.png  KDE projections (XY and XZ) by genotype,
                                       with PBS and Lecanemab overlaid to show
                                       the treatment effect.

Method
------
A weighted 3-D Gaussian KDE is computed per group using
``scipy.stats.gaussian_kde`` with ``weights=equiv_diam_um``, so larger plaques
contribute proportionally more to the density estimate ("plaque burden").
The 3-D density is evaluated on a regular grid and then marginalised (summed)
along the Z or Y axis to produce XY and XZ projections respectively.
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
if "snakemake" in dir():
    input_parquet = str(snakemake.input.parquet)  # noqa: F821
    output_figs = dict(snakemake.output)  # noqa: F821
    cohort = snakemake.wildcards.cohort  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()
    input_parquet = args.parquet
    cohort = args.cohort
    output_figs = {
        "kde_plaque_burden": f"{args.output_dir}/fig_{cohort}_kde_plaque_burden.png",
    }

# ── Constants ────────────────────────────────────────────────────────────────
sns.set_theme(style="white", font_scale=1.1)

TREAT_ORDER = ["PBS", "Lecanemab"]
TREAT_PALETTE = {"PBS": "#4C72B0", "Lecanemab": "#DD8452"}
GENO_ORDER = ["ApoE3", "ApoE4"]

VOL_THRESH_ML = 1e-4  # max plaque volume filter (removes large artifacts, same as other scripts)

# KDE/grid parameters
N_KDE_MAX = 12_000   # max plaques per group for KDE fitting (balances accuracy vs runtime)
GRID_N_XY = 45       # grid resolution for X and Y axes
GRID_N_Z = 25        # grid resolution for Z axis (smaller typical Z range)
KDE_EVAL_BATCH = 20_000   # batch size for KDE grid evaluation (controls peak memory use)
MIN_PLAQUES_FOR_KDE = 20  # minimum plaques in a group to attempt KDE fitting
CONTOUR_PERCENTILES = [70, 85, 95]  # density percentiles at which to draw contour lines

PROJECTIONS = [
    ("template_x", "template_y", "z", "XY projection (A–P × M–L)"),
    ("template_x", "template_z", "y", "XZ projection (A–P × D–V)"),
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
    treat: _alpha_cmap(color, f"cmap_{treat.lower()}")
    for treat, color in TREAT_PALETTE.items()
}


# ── Helper: 3-D KDE → 2-D projections ───────────────────────────────────────
def kde_projections(x, y, z, weights, x_lim, y_lim, z_lim, rng_seed=42):
    """Compute a weighted 3-D KDE and return marginalised XY and XZ projections.

    Parameters
    ----------
    x, y, z : 1-D arrays of plaque template coordinates (mm).
    weights  : 1-D array of equiv_diam_um values (positive, not yet normalised).
    x_lim, y_lim, z_lim : (min, max) tuples that define the evaluation grid.
    rng_seed : random seed used when subsampling.

    Returns
    -------
    projections : dict with keys ``"XY"`` and ``"XZ"``.
        Each value is a dict ``{"gx": ..., "gy": ..., "density": ...}``
        where ``gx``/``gy`` are 2-D meshgrid arrays and ``density`` is the
        marginalised KDE intensity (same shape).
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

    dy = yi[1] - yi[0]
    dz = zi[1] - zi[0]

    # Build full 3-D evaluation grid (indexing='ij' → shape (nx, ny, nz))
    gx3, gy3, gz3 = np.meshgrid(xi, yi, zi, indexing="ij")
    eval_pts = np.vstack([gx3.ravel(), gy3.ravel(), gz3.ravel()])

    # Evaluate in batches to avoid very large array allocations
    batch = KDE_EVAL_BATCH
    n_eval = eval_pts.shape[1]
    density_flat = np.empty(n_eval, dtype=float)
    for start in range(0, n_eval, batch):
        end = min(start + batch, n_eval)
        density_flat[start:end] = kde(eval_pts[:, start:end])

    density_3d = density_flat.reshape(GRID_N_XY, GRID_N_XY, GRID_N_Z)

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


# ── Load and prepare data ────────────────────────────────────────────────────
df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()

geno_present = [g for g in GENO_ORDER if (df["genotype"] == g).any()]
treat_present = [t for t in TREAT_ORDER if (df["treatment"] == t).any()]

df["genotype"] = pd.Categorical(df["genotype"], categories=geno_present, ordered=True)
df["treatment"] = pd.Categorical(df["treatment"], categories=treat_present, ordered=True)

# Shared axis limits derived from entire dataset (consistent across groups)
x_lim = (df["template_x"].quantile(0.002), df["template_x"].quantile(0.998))
y_lim = (df["template_y"].quantile(0.002), df["template_y"].quantile(0.998))
z_lim = (df["template_z"].quantile(0.002), df["template_z"].quantile(0.998))

# ── Main figure ───────────────────────────────────────────────────────────────
# Layout: rows = genotypes, cols = projections (XY, XZ)
n_genos = len(geno_present)
n_proj = len(PROJECTIONS)

fig, axes = plt.subplots(
    n_genos, n_proj,
    figsize=(6.5 * n_proj, 5.5 * n_genos),
    squeeze=False,
)

for row_idx, geno in enumerate(geno_present):
    gdf = df[df["genotype"] == geno]

    # Pre-compute KDE projections for each treatment present in this genotype
    treat_projs = {}
    for treat in treat_present:
        tdf = gdf[gdf["treatment"] == treat]
        if len(tdf) < MIN_PLAQUES_FOR_KDE:
            continue
        treat_projs[treat] = kde_projections(
            tdf["template_x"].values,
            tdf["template_y"].values,
            tdf["template_z"].values,
            tdf["equiv_diam_um"].values,
            x_lim=x_lim,
            y_lim=y_lim,
            z_lim=z_lim,
        )

    for col_idx, (xcol, ycol, marginal_axis, proj_label) in enumerate(PROJECTIONS):
        proj_key = "XY" if marginal_axis == "z" else "XZ"
        ax = axes[row_idx, col_idx]

        # Find shared density scale across treatments for this genotype/projection
        all_densities = [
            treat_projs[t][proj_key]["density"]
            for t in treat_present if t in treat_projs
        ]
        if not all_densities:
            ax.set_visible(False)
            continue
        vmax = max(d.max() for d in all_densities)

        for treat in treat_present:
            if treat not in treat_projs:
                continue
            proj = treat_projs[treat][proj_key]
            gx, gy, dens = proj["gx"], proj["gy"], proj["density"]

            # Normalise density to [0, 1] relative to the shared scale
            dens_norm = dens / vmax if vmax > 0 else dens

            ax.pcolormesh(
                gx, gy, dens_norm,
                cmap=TREAT_CMAP[treat],
                shading="gouraud",
                rasterized=True,
                vmin=0.0, vmax=1.0,
            )
            # Overlay a few contour lines at the top density percentiles
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

        ax.set_aspect("equal")
        ax.set_xlabel(xcol.replace("template_", "").upper() + " (mm)", fontsize=9)
        ax.set_ylabel(ycol.replace("template_", "").upper() + " (mm)", fontsize=9)
        ax.set_title(f"{geno} — {proj_label}", fontsize=10)

        # Legend patches (only on the first row, last column)
        if row_idx == 0 and col_idx == n_proj - 1:
            legend_handles = [
                Patch(facecolor=TREAT_PALETTE[t], alpha=0.75, label=t)
                for t in treat_present
            ]
            ax.legend(
                handles=legend_handles,
                title="Treatment",
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
