# Snakemake workflow for SPIMquant vessel & plaque analysis
#
# Each rule produces a different plot (or intermediate data file).
#
# Usage
# -----
# Run the full pipeline (requires data_*.parquet already present):
#   snakemake --cores all
#
# Re-generate figures from existing parquet files:
#   snakemake --cores all --rerun-triggers mtime
#
# Re-import raw data (requires config.yaml with dataset paths):
#   snakemake --cores all --config datasets=config.yaml data_early.parquet

# ── Cohorts ───────────────────────────────────────────────────────────────────
COHORTS = ["early", "late"]

# ── Figure groups ─────────────────────────────────────────────────────────────
REGIONAL_FIGS = [
    "roi_top_density",
    "roi_density_boxplot",
    "roi_metric_heatmap",
    "roi_proximity_fractions",
    "roi_fold_change",
    "atlas_density_all",
    "atlas_density_groups",
    "atlas_fold_change",
    "atlas_mean_diam",
    "atlas_frac_inside",
]

TREATMENT_FIGS = [
    "boxplots_treatment_genotype",
    "violin_plaque_size",
    "interaction_plots",
    "stratified_by_genotype",
    "ecdf_size_by_genotype",
    "size_distribution_by_genotype",
]

VESSEL_FIGS = [
    "sdt_ecdf",
    "proximity_fractions_stacked",
    "proximity_fractions",
    "proximity_interaction",
    "vessel_calibre_ecdf",
    "vessel_calibre_subject",
    "vessel_diam_bins",
    "spatial_vessel_proximity",
]

ALL_FIGS = REGIONAL_FIGS + TREATMENT_FIGS + VESSEL_FIGS


# ── Default target ────────────────────────────────────────────────────────────
rule all:
    input:
        expand("roi_data_{cohort}.parquet", cohort=COHORTS),
        expand("fig_{cohort}_{fig}.png", cohort=COHORTS, fig=ALL_FIGS),


# ── Data import (from raw SPIMquant outputs) ──────────────────────────────────
# Requires config entry:
#   datasets:
#     early:
#       participant_tsv: /path/to/early/participants.tsv
#       spimquant_dir:   /path/to/early/derivatives/spimquant
#     late:
#       participant_tsv: /path/to/late/participants.tsv
#       spimquant_dir:   /path/to/late/derivatives/spimquant
rule import_data:
    output:
        "data_{cohort}.parquet",
    params:
        participant_tsv=lambda wc: config.get("datasets", {}).get(wc.cohort, {}).get("participant_tsv", ""),
        spimquant_dir=lambda wc: config.get("datasets", {}).get(wc.cohort, {}).get("spimquant_dir", ""),
    log:
        "logs/import_data_{cohort}.log",
    script:
        "scripts/import_data.py"


# ── ROI aggregation ───────────────────────────────────────────────────────────
rule regional_dataframe:
    input:
        parquet="data_{cohort}.parquet",
        lut="tpl-ABAv3_seg-all_dseg.tsv",
    output:
        "roi_data_{cohort}.parquet",
    log:
        "logs/regional_dataframe_{cohort}.log",
    script:
        "scripts/regional_dataframe.py"


# ── Regional / atlas figures ──────────────────────────────────────────────────
rule regional_analysis:
    input:
        roi_parquet="roi_data_{cohort}.parquet",
        atlas="tpl-ABAv3_seg-all_dseg.nii.gz",
    output:
        "fig_{cohort}_{fig}.png",
    log:
        "logs/regional_analysis_{cohort}_{fig}.log",
    wildcard_constraints:
        fig="|".join(REGIONAL_FIGS),
    script:
        "scripts/regional_analysis.py"


# ── Treatment-effect figures ──────────────────────────────────────────────────
rule treatment_analysis:
    input:
        parquet="data_{cohort}.parquet",
    output:
        "fig_{cohort}_{fig}.png",
    log:
        "logs/treatment_analysis_{cohort}_{fig}.log",
    wildcard_constraints:
        fig="|".join(TREATMENT_FIGS),
    script:
        "scripts/treatment_effect_analysis.py"


# ── Vessel-proximity figures ──────────────────────────────────────────────────
rule vessel_analysis:
    input:
        parquet="data_{cohort}.parquet",
    output:
        "fig_{cohort}_{fig}.png",
    log:
        "logs/vessel_analysis_{cohort}_{fig}.log",
    wildcard_constraints:
        fig="|".join(VESSEL_FIGS),
    script:
        "scripts/vessel_spatial_analysis.py"
