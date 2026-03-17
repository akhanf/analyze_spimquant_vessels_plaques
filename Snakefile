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
        roi_top_density="fig_{cohort}_roi_top_density.png",
        roi_density_boxplot="fig_{cohort}_roi_density_boxplot.png",
        roi_metric_heatmap="fig_{cohort}_roi_metric_heatmap.png",
        roi_proximity_fractions="fig_{cohort}_roi_proximity_fractions.png",
        roi_fold_change="fig_{cohort}_roi_fold_change.png",
        atlas_density_all="fig_{cohort}_atlas_density_all.png",
        atlas_density_groups="fig_{cohort}_atlas_density_groups.png",
        atlas_fold_change="fig_{cohort}_atlas_fold_change.png",
        atlas_mean_diam="fig_{cohort}_atlas_mean_diam.png",
        atlas_frac_inside="fig_{cohort}_atlas_frac_inside.png",
    log:
        "logs/regional_analysis_{cohort}.log",
    script:
        "scripts/regional_analysis.py"


# ── Treatment-effect figures ──────────────────────────────────────────────────
rule treatment_analysis:
    input:
        parquet="data_{cohort}.parquet",
    output:
        boxplots_treatment_genotype="fig_{cohort}_boxplots_treatment_genotype.png",
        violin_plaque_size="fig_{cohort}_violin_plaque_size.png",
        interaction_plots="fig_{cohort}_interaction_plots.png",
        stratified_by_genotype="fig_{cohort}_stratified_by_genotype.png",
        ecdf_size_by_genotype="fig_{cohort}_ecdf_size_by_genotype.png",
        size_distribution_by_genotype="fig_{cohort}_size_distribution_by_genotype.png",
    log:
        "logs/treatment_analysis_{cohort}.log",
    script:
        "scripts/treatment_effect_analysis.py"


# ── Vessel-proximity figures ──────────────────────────────────────────────────
rule vessel_analysis:
    input:
        parquet="data_{cohort}.parquet",
    output:
        sdt_ecdf="fig_{cohort}_sdt_ecdf.png",
        proximity_fractions_stacked="fig_{cohort}_proximity_fractions_stacked.png",
        proximity_fractions="fig_{cohort}_proximity_fractions.png",
        proximity_interaction="fig_{cohort}_proximity_interaction.png",
        vessel_calibre_ecdf="fig_{cohort}_vessel_calibre_ecdf.png",
        vessel_calibre_subject="fig_{cohort}_vessel_calibre_subject.png",
        vessel_diam_bins="fig_{cohort}_vessel_diam_bins.png",
        spatial_vessel_proximity="fig_{cohort}_spatial_vessel_proximity.png",
    log:
        "logs/vessel_analysis_{cohort}.log",
    script:
        "scripts/vessel_spatial_analysis.py"
