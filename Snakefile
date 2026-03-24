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

configfile: 'config.yml'


wildcard_constraints:
    cohort='[0-9a-zA-Z]+'

# ── Cohorts ───────────────────────────────────────────────────────────────────
COHORTS = config['datasets'].keys()

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

REGIONAL_CSVS = [
    "roi_treatment_stats",
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
    "proximity_counts_stacked",
    "proximity_fractions",
    "proximity_interaction",
    "vessel_calibre_ecdf",
    "vessel_calibre_subject",
    "vessel_diam_bins",
    "spatial_vessel_proximity",
    "spatial_plaques_size",
    "spatial_plaques_vessel_dist_bins",
]

VESSEL_GIFS = [
    "spatial_plaques_vessel_dist_anim",
]

KDE_FIGS = [
    "kde_plaque_burden",
    "kde_plaque_burden_zslices",
]

ALL_FIGS = REGIONAL_FIGS + TREATMENT_FIGS + VESSEL_FIGS + KDE_FIGS


# ── Default target ────────────────────────────────────────────────────────────
rule all:
    input:
        expand("roi_data_{cohort}_seg-{seg}.parquet", cohort=COHORTS, seg=config['segs']),
        expand("figures/fig_{cohort}_{seg}_{fig}.png", cohort=COHORTS, fig=REGIONAL_FIGS, seg=config['segs']),
        expand("figures/fig_{cohort}_{fig}.png", cohort=COHORTS, fig=TREATMENT_FIGS + VESSEL_FIGS + KDE_FIGS),
        expand("figures/fig_{cohort}_{fig}.gif", cohort=COHORTS, fig=VESSEL_GIFS),
        expand("{cohort}_{csv}.csv", cohort=COHORTS, csv=REGIONAL_CSVS),


rule add_vol_to_dseg_tsv:
    input:
        lut=lambda wildcards: Path(config['template_dir']) / "seg-{seg}_tpl-ABAv3_dseg.tsv",
        nii=lambda wildcards: Path(config['template_dir']) / "seg-{seg}_tpl-ABAv3_dseg.nii.gz",
    output:
        lut="tpl-ABAv3_seg-{seg}_dseg.tsv"
    script:
        "scripts/calc_dseg_volumes.py"


rule import_data:
    input:
        participant_tsv=lambda wc: config.get("datasets", {}).get(wc.cohort, {}).get("participant_tsv", ""),
        spimquant_dir=lambda wc: config.get("datasets", {}).get(wc.cohort, {}).get("spimquant_dir", ""),
    output:
        "data_{cohort}.parquet",
    params:
    log:
        "logs/import_data_{cohort}.log",
    script:
        "scripts/import_data.py"


# ── ROI aggregation ───────────────────────────────────────────────────────────
rule regional_dataframe:
    input:
        parquet="data_{cohort}.parquet",
        lut="tpl-ABAv3_seg-{seg}_dseg.tsv"
    output:
        "roi_data_{cohort}_seg-{seg}.parquet",
    log:
        "logs/regional_dataframe_{cohort}_{seg}.log",
    script:
        "scripts/regional_dataframe.py"


# ── Regional / atlas figures ──────────────────────────────────────────────────
rule regional_analysis:
    input:
        roi_parquet="roi_data_{cohort}_seg-{seg}.parquet",
        atlas=lambda wildcards: Path(config['datasets'][wildcards.cohort]['spimquant_dir']) / "tpl-ABAv3"  / "seg-{seg}_tpl-ABAv3_dseg.nii.gz",
    params:
        metric="plaque_density",
    output:
        roi_top_density="figures/fig_{cohort}_{seg}_roi_top_density.png",
        roi_density_boxplot="figures/fig_{cohort}_{seg}_roi_density_boxplot.png",
        roi_metric_heatmap="figures/fig_{cohort}_{seg}_roi_metric_heatmap.png",
        roi_proximity_fractions="figures/fig_{cohort}_{seg}_roi_proximity_fractions.png",
        roi_fold_change="figures/fig_{cohort}_{seg}_roi_fold_change.png",
        atlas_density_all="figures/fig_{cohort}_{seg}_atlas_density_all.png",
        atlas_density_groups="figures/fig_{cohort}_{seg}_atlas_density_groups.png",
        atlas_fold_change="figures/fig_{cohort}_{seg}_atlas_fold_change.png",
        atlas_mean_diam="figures/fig_{cohort}_{seg}_atlas_mean_diam.png",
        atlas_frac_inside="figures/fig_{cohort}_{seg}_atlas_frac_inside.png",
        roi_treatment_stats="{cohort}_{seg}_roi_treatment_stats.csv",
    log:
        "logs/regional_analysis_{cohort}_{seg}.log",
    script:
        "scripts/regional_analysis.py"


# ── Treatment-effect figures ──────────────────────────────────────────────────
rule treatment_analysis:
    input:
        parquet="data_{cohort}.parquet",
    output:
        boxplots_treatment_genotype="figures/fig_{cohort}_boxplots_treatment_genotype.png",
        violin_plaque_size="figures/fig_{cohort}_violin_plaque_size.png",
        interaction_plots="figures/fig_{cohort}_interaction_plots.png",
        stratified_by_genotype="figures/fig_{cohort}_stratified_by_genotype.png",
        ecdf_size_by_genotype="figures/fig_{cohort}_ecdf_size_by_genotype.png",
        size_distribution_by_genotype="figures/fig_{cohort}_size_distribution_by_genotype.png",
    log:
        "logs/treatment_analysis_{cohort}.log",
    script:
        "scripts/treatment_effect_analysis.py"


# ── Vessel-proximity figures ──────────────────────────────────────────────────
rule vessel_analysis:
    input:
        parquet="data_{cohort}.parquet",
    output:
        sdt_ecdf="figures/fig_{cohort}_sdt_ecdf.png",
        proximity_fractions_stacked="figures/fig_{cohort}_proximity_fractions_stacked.png",
        proximity_counts_stacked="figures/fig_{cohort}_proximity_counts_stacked.png",
        proximity_fractions="figures/fig_{cohort}_proximity_fractions.png",
        proximity_interaction="figures/fig_{cohort}_proximity_interaction.png",
        vessel_calibre_ecdf="figures/fig_{cohort}_vessel_calibre_ecdf.png",
        vessel_calibre_subject="figures/fig_{cohort}_vessel_calibre_subject.png",
        vessel_diam_bins="figures/fig_{cohort}_vessel_diam_bins.png",
        spatial_vessel_proximity="figures/fig_{cohort}_spatial_vessel_proximity.png",
        spatial_plaques_size="figures/fig_{cohort}_spatial_plaques_size.png",
        spatial_plaques_vessel_dist_bins="figures/fig_{cohort}_spatial_plaques_vessel_dist_bins.png",
        spatial_plaques_vessel_dist_anim="figures/fig_{cohort}_spatial_plaques_vessel_dist_anim.gif",
    log:
        "logs/vessel_analysis_{cohort}.log",
    script:
        "scripts/vessel_spatial_analysis.py"


# ── 3-D KDE plaque-burden figures ────────────────────────────────────────────
rule kde_analysis:
    input:
        parquet="data_{cohort}.parquet",
    output:
        kde_plaque_burden="figures/fig_{cohort}_kde_plaque_burden.png",
        kde_plaque_burden_zslices="figures/fig_{cohort}_kde_plaque_burden_zslices.png",
    log:
        "logs/kde_analysis_{cohort}.log",
    script:
        "scripts/kde_plaque_burden.py"
