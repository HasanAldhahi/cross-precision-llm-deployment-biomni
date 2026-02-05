# Calibration Dataset

This folder contains the calibration dataset used for quantizing the models.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total samples** | 123 |
| **Source** | Baseline R0 evaluation (successful completions only) |
| **Format** | prompt + full_response (complete trajectories) |
| **Average tokens** | 25,718 |
| **Total tokens** | 3,163,274 |

## Task Distribution

| Task | Samples |
|------|---------|
| crispr_delivery | 8 |
| gwas_causal_gene_gwas_catalog | 13 |
| gwas_causal_gene_opentargets | 13 |
| gwas_causal_gene_pharmaprojects | 13 |
| gwas_variant_prioritization | 13 |
| lab_bench_dbqa | 13 |
| lab_bench_seqqa | 13 |
| patient_gene_detection | 13 |
| rare_disease_diagnosis | 12 |
| screen_gene_retrieval | 12 |

## Files

- `calibration_data.json` - The final calibration dataset
- `calibration_preview.txt` - Detailed statistics and sample preview
- `Data_r0_annotated_cleaned.jsonl` - Cleaned source data
- `prepare_calibration.py` - Script to prepare calibration data
- `clean_calibration_data.py` - Script to clean and filter the data
