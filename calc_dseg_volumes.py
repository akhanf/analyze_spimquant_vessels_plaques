#!/usr/bin/env python3
"""Calculate the volume (mm^3) for each atlas region in the dseg.nii.gz
and add it as a new column in the dseg.tsv."""

import numpy as np
import nibabel as nib
import pandas as pd

DSEG_NII = "tpl-ABAv3_seg-all_dseg.nii.gz"
DSEG_TSV = "tpl-ABAv3_seg-all_dseg.tsv"


def calc_dseg_volumes(dseg_nii: str = DSEG_NII, dseg_tsv: str = DSEG_TSV) -> None:
    # Load segmentation image
    img = nib.load(dseg_nii)
    data = np.asarray(img.dataobj, dtype=int)

    # Voxel volume in mm^3 (product of voxel dimensions)
    zooms = img.header.get_zooms()
    voxel_vol_mm3 = float(np.prod(zooms))

    # Count voxels per label
    labels, counts = np.unique(data, return_counts=True)
    label_to_count = dict(zip(labels, counts))

    # Load TSV
    df = pd.read_csv(dseg_tsv, sep="\t")

    # Map each index to its voxel count, then multiply by voxel volume
    df["volume_mm3"] = df["index"].map(label_to_count).fillna(0) * voxel_vol_mm3

    # Write back to the same TSV
    df.to_csv(dseg_tsv, sep="\t", index=False)
    print(f"Updated {dseg_tsv} with volume_mm3 column.")


if __name__ == "__main__":
    calc_dseg_volumes()
