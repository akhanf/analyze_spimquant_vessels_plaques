#!/usr/bin/env python3
"""Calculate the volume (mm^3) for each atlas region in the dseg.nii.gz
and add it as a new column in the dseg.tsv."""

import numpy as np
import nibabel as nib
import pandas as pd


# Load segmentation image
img = nib.load(snakemake.input.nii)
data = np.asarray(img.dataobj, dtype=int)

# Voxel volume in mm^3 (product of voxel dimensions)
zooms = img.header.get_zooms()
voxel_vol_mm3 = float(np.prod(zooms))

# Count voxels per label
labels, counts = np.unique(data, return_counts=True)
label_to_count = dict(zip(labels, counts))

# Load TSV
df = pd.read_csv(snakemake.input.lut, sep="\t")

# Map each index to its voxel count, then multiply by voxel volume
df["volume_mm3"] = df["index"].map(label_to_count).fillna(0) * voxel_vol_mm3

# Write back to the same TSV
df.to_csv(snakemake.output.lut, sep="\t", index=False)
print(f"Updated {snakemake.output.lut} with volume_mm3 column.")


