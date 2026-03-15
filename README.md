# Lecanemab Lightsheet data - March 2026

The data.parquet table has instance-level data for Abeta plaque in humanized mouse models 
of Alzheimer's disease, where some are treated with Lecanemab, and some are treated with PBS.

Description of the data is below, with variable names referenced as `indicated`.

## Plaque segmentation

Plaques are segmented, and individual instances are quantified by their size (`nvoxels`) and 
centroid location (`x`,`y`,`z`). Volume is calculated using the voxel size (~1.6x1.6x2.75 um)
(`plaque_vol_ml`), and equivalent diameter is computed using a spherical assumption (`equiv_diam_um`).
The segmentation approach has already applied a minimum size threshold of 200 nvoxels, 
but no maximum has been computed thus some additional filtering based on what is plausible 
for plaque aggregate should be used.

## Anatomical labels

The Allen Brain Atlas (ABAv3) label name (`label`), and number (`index`) is 
sampled on each. Coordinates are also non-linearly transformed 
to the CCFv3 space (`template_x`,`template_y`,`template_z`). 

## Proximity to vessels

Vascular segmentation is also performed with CD31 and a deep learning model, 
and a signed distance map (neg: inside vessel, pos: outside vessel) in millimeters is produced, 
which is also sampled at each point (`sdt_CD31`). 

## Demographic info

The ApoE isoform (`genotype`), `sex` and type of injection (`treatment`) is annotated 
for each `subject` plaque instance. 

