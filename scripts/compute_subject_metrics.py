import pandas as pd
import numpy as np

# ── Snakemake integration ────────────────────────────────────────────────────
if "snakemake" in dir():
    input_parquet = str(snakemake.input.parquet)  # noqa: F821
    output_figs = dict(snakemake.output)  # noqa: F821
    cohort = snakemake.wildcards.cohort  # noqa: F821
    cfg = snakemake.params.plot_config  # noqa: F821
else:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--cohort", required=True)
    args = parser.parse_args()
    input_parquet = args.parquet
    cohort = args.cohort
    cfg = DEFAULT_PLOT_CONFIG
    output_figs = {
        "treatment_stats_boxplots": f"{args.output_dir}/fig_{cohort}_treatment_stats_boxplots.png",
        "treatment_anova_table": f"{args.output_dir}/{cohort}_treatment_anova_table.csv",
        "treatment_tukey_results": f"{args.output_dir}/{cohort}_treatment_tukey_results.csv",
    }



# factors[0] = primary factor (e.g. treatment), factors[1] = secondary (e.g. genotype),
# factors[2] = tertiary (e.g. sex).  Swapping entries in config.yml changes which
# factor plays which role without touching any script.
_factors = cfg["factors"]
treat_cfg = _factors[0]
TREAT_COL = treat_cfg["column"]
TREAT_ORDER = treat_cfg["order"]
TREAT_PALETTE = treat_cfg["palette"]

geno_cfg = _factors[1] if len(_factors) > 1 else {}
GENO_COL = geno_cfg.get("column", "")
GENO_ORDER = geno_cfg.get("order", [])
GENO_PALETTE = geno_cfg.get("palette", {})

sex_cfg = _factors[2] if len(_factors) > 2 else {}
SEX_COL = sex_cfg.get("column", "")

VOL_THRESH_ML = cfg.get("volume_threshold_ml", 1e-4)


def compute_subject_metrics(plaque_df):
    """Aggregate plaque-level data to one row per subject."""
    grp = plaque_df.groupby(
        ["subject", TREAT_COL, GENO_COL, SEX_COL], observed=True
    )
    metrics = grp.agg(
        plaque_count=("nvoxels", "size"),
        total_vol_ml=("plaque_vol_ml", "sum"),
        mean_vol_ml=("plaque_vol_ml", "mean"),
        median_vol_ml=("plaque_vol_ml", "median"),
        mean_diam_um=("equiv_diam_um", "mean"),
        median_diam_um=("equiv_diam_um", "median"),
        total_vol_um3=("plaque_vol_um3", "sum"),
    ).reset_index()
    for col in [
        "plaque_count", "total_vol_ml", "mean_vol_ml", "median_vol_ml",
        "mean_diam_um", "median_diam_um", "total_vol_um3",
    ]:
        metrics[f"log_{col}"] = np.log(metrics[col])
    return metrics


# ── Load and prepare data ────────────────────────────────────────────────────
df_raw = pd.read_parquet(input_parquet)
df = df_raw.loc[df_raw["plaque_vol_ml"] <= VOL_THRESH_ML].copy()
df[TREAT_COL] = pd.Categorical(df[TREAT_COL], categories=TREAT_ORDER, ordered=True)
geno_present = [g for g in GENO_ORDER if (df[GENO_COL] == g).any()]
df[GENO_COL] = pd.Categorical(df[GENO_COL], categories=geno_present, ordered=True)
geno_palette = {g: GENO_PALETTE[g] for g in geno_present}
n_geno = len(geno_present)

subj = compute_subject_metrics(df)
subj[GENO_COL] = pd.Categorical(subj[GENO_COL], categories=geno_present, ordered=True)


subj.to_csv(snakemake.output.tsv,sep='\t',index=False)

