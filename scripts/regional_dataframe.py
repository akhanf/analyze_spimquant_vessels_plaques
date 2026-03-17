"""Aggregate plaque-level data to per-subject × per-atlas-region summaries.

Reads  : data_{cohort}.parquet  +  tpl-ABAv3_seg-all_dseg.tsv
Writes : roi_data_{cohort}.parquet
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Snakemake integration ────────────────────────────────────────────────────
if "snakemake" in dir():
    input_parquet = str(snakemake.input.parquet)  # noqa: F821
    input_lut = str(snakemake.input.lut)  # noqa: F821
    output_parquet = str(snakemake.output[0])  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--lut", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_parquet = args.parquet
    input_lut = args.lut
    output_parquet = args.output

# ── Constants ────────────────────────────────────────────────────────────────
VOL_THRESH_ML = 1e-4
TAU_UM = 5.0
VESSEL_BINS_UM = [0, 10, 20, np.inf]
VESSEL_BIN_LABELS = ["<10 µm", "10\u201320 µm", ">20 µm"]
TREAT_ORDER = ["PBS", "Lecanemab"]
GENO_ORDER = ["ApoE3", "ApoE4"]
GROUP_COLS = ["subject", "treatment", "genotype", "sex", "index", "name"]


# ── Functions ────────────────────────────────────────────────────────────────
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
    )
    return out


def build_roi_dataframe(plaque_df, group_cols=GROUP_COLS):
    """Aggregate plaque metrics per subject per atlas ROI."""
    grp = plaque_df.groupby(group_cols, observed=True)

    agg = grp.agg(
        plaque_count=("nvoxels", "size"),
        total_vol_ml=("plaque_vol_ml", "sum"),
        mean_vol_ml=("plaque_vol_ml", "mean"),
        median_vol_ml=("plaque_vol_ml", "median"),
        mean_diam_um=("equiv_diam_um", "mean"),
        median_diam_um=("equiv_diam_um", "median"),
        total_vol_um3=("plaque_vol_um3", "sum"),
        mean_sdt_um=("sdt_CD31_um", "mean"),
        median_sdt_um=("sdt_CD31_um", "median"),
    ).reset_index()

    rel_counts = (
        plaque_df.groupby(group_cols + ["vessel_relation"], observed=True)
        .size()
        .rename("n_rel")
        .reset_index()
        .merge(agg[group_cols + ["plaque_count"]], on=group_cols)
    )
    rel_counts["frac"] = rel_counts["n_rel"] / rel_counts["plaque_count"]

    PROX_RENAME = {
        "inside vessel": "frac_inside_vessel",
        f"near vessel (0\u2013{TAU_UM:g} \u00b5m)": "frac_near_vessel",
        f"far from vessel (>{TAU_UM:g} \u00b5m)": "frac_far_vessel",
    }
    rel_wide = (
        rel_counts.pivot_table(
            index=group_cols,
            columns="vessel_relation",
            values="frac",
            fill_value=0.0,
        )
        .rename(columns=PROX_RENAME)
        .reset_index()
    )
    rel_wide.columns.name = None
    return agg.merge(rel_wide, on=group_cols, how="left")


# ── Main ─────────────────────────────────────────────────────────────────────
lut = pd.read_csv(input_lut, sep="\t")
print(f"LUT: {len(lut):,} regions")

df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()
df["treatment"] = pd.Categorical(df["treatment"], categories=TREAT_ORDER, ordered=True)
df["genotype"] = pd.Categorical(df["genotype"], categories=GENO_ORDER, ordered=True)
print(
    f"Retained {len(df):,} plaques from {df['subject'].nunique()} subjects "
    f"(removed {len(df_raw) - len(df):,} oversized plaques)"
)

df = classify_plaques(df)
roi_df = build_roi_dataframe(df)

roi_df = roi_df.merge(lut[["index", "volume_mm3"]], on="index", how="left")
roi_df["plaque_density"] = roi_df["plaque_count"] / roi_df["volume_mm3"]
roi_df["vol_density_ml"] = roi_df["total_vol_ml"] / roi_df["volume_mm3"]

roi_df.to_parquet(output_parquet)
print(
    f"Saved {output_parquet}  —  "
    f"{len(roi_df):,} rows \u00d7 {len(roi_df.columns)} columns"
)
