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

## Slide deck

A [Marp](https://marp.app)-based presentation summarising the methods and results is
available in [`presentation.md`](presentation.md).

### Building the slides locally

You need [Node.js](https://nodejs.org) installed, then run:

```bash
# Install Marp CLI (one-time)
npm install -g @marp-team/marp-cli

# Build HTML slides
marp presentation.md --html --allow-local-files --output slides.html

# Build PDF slides
marp presentation.md --allow-local-files --output presentation.pdf
```

### Enabling GitHub Pages deployment

The included GitHub Actions workflow (`.github/workflows/marp-deploy.yml`) automatically
builds and publishes the slide deck to GitHub Pages on every push to `main`.

To activate it:

1. Go to your repository on GitHub and open **Settings → Pages**.
2. Under **Build and deployment**, set the **Source** to **GitHub Actions**.
3. Push any change to `main` (or trigger the workflow manually via
   **Actions → Build and Deploy Marp Slides → Run workflow**).

The live slide deck will be available at:

```
https://<owner>.github.io/<repository>/
```

A PDF version of the slides is also saved as a downloadable workflow artifact
(`presentation-pdf`) after each run.

