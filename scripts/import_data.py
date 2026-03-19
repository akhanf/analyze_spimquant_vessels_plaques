"""Import raw SPIMquant data for one cohort and save a cleaned parquet file.

Inputs (via Snakemake or CLI):
  participant_tsv : path to BIDS participants.tsv
  spimquant_dir   : path to SPIMquant derivatives folder
  cohort          : cohort name (e.g. "early" or "late")

Output:
  data_{cohort}.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

participant_tsv = snakemake.params.participant_tsv  # noqa: F821
spimquant_dir = snakemake.params.spimquant_dir  # noqa: F821
output_parquet = str(snakemake.output[0])  # noqa: F821
cohort = snakemake.wildcards.cohort  # noqa: F821

# ── Constants ────────────────────────────────────────────────────────────────
VOXEL_VOL_UM3 = 1.6 * 1.6 * 2.75  # µm³ per voxel
VOXEL_VOL_ML = 0.0016 * 0.0016 * 0.00275  # mL per voxel


# ── Functions ────────────────────────────────────────────────────────────────
def load_subject_df(spimquant_dir: str, subject: str) -> pd.DataFrame | None:
    regionpropstats_tsv = (
        f"{spimquant_dir}/{subject}/micr/"
        f"{subject}_sample-brain_acq-imaris4x_stain-Abeta_seg-all_"
        f"from-ABAv3_level-5_desc-otsu+k3i2_regionpropstats.tsv"
    )
    if not Path(regionpropstats_tsv).exists():
        return None
    df_subject = pd.read_csv(regionpropstats_tsv, sep="\t")
    df_subject["subject"] = subject
    return df_subject


# ── Main ─────────────────────────────────────────────────────────────────────
df_participants = pd.read_csv(participant_tsv, sep="\t")

subject_dfs = [
    df_subject
    for subject in df_participants["participant_id"]
    if (df_subject := load_subject_df(spimquant_dir, subject)) is not None
]

if not subject_dfs:
    raise RuntimeError(f"No data found for cohort '{cohort}'")

df_dataset = pd.concat(subject_dfs, ignore_index=False).merge(
    df_participants,
    left_on="subject",
    right_on="participant_id",
    how="left",
)

# keep only Lecanemab and PBS (vehicle) arms
df = df_dataset.query("treatment == 'Lecanemab' or treatment == 'PBS'").copy()

# add derived variables
df["sdt_CD31_um"] = df["sdt_CD31"] * 1000.0
df["plaque_vol_um3"] = df["nvoxels"] * VOXEL_VOL_UM3
df["plaque_vol_ml"] = df["nvoxels"] * VOXEL_VOL_ML
df["equiv_diam_um"] = 2 * ((3 * df["plaque_vol_um3"]) / (4 * np.pi)) ** (1 / 3)

df.to_parquet(output_parquet)
print(
    f"Saved {output_parquet}: {len(df):,} plaques "
    f"from {df['subject'].nunique()} subjects"
)
