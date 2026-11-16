# Changelog

## 1.3.0 — ablation and independent benchmark

### Added
- `colab/cell_17_ablation_order_aware.py`: 2 × 2 ablation of the order-aware modification (positional embedding,
  padding masks), with the Cell 3 harness; human test and external species (manuscript Section 3.11, Table 9).
- `colab/cell_18_genetrust_on_genomic_benchmarks.py`: the GeneTrust protocol on two Genomic Benchmarks tasks
  (human enhancers with mouse enhancers as external set; human non-TATA promoters as a length-shortcut control)
  (manuscript Section 3.12, Table 10).
- `colab/cell_19_duplicate_check.py`: length and label agreement of the duplicated and near-duplicated test
  sequences in the Genomic Benchmarks tasks (manuscript Section 3.12).
- Results: `results/tables/Cell17_*`, `Cell19_*`, `Cell18_*` (including homology-aware partitions of both benchmark tasks and
  `Cell17_ablation_contrasts_external.csv`), `results/predictions/Cell17_ablation_predictions.npz`.

## 1.2.0 — manuscript revision

### Changed
- Manuscript retitled; README and `CITATION.cff` updated to the new title.
- `analysis/cross_species_final.py` now also writes `results/tables/motif_direct_comparison.csv`: one-sided paired
  Wilcoxon tests that the order-aware Transformer depends more than the order-blind one on each motif in chicken and
  zebrafish (as Supplementary Table S18 does for mouse; reported in Supplementary Table S21).
- `figures/Figure 7.png` renamed to `figures/Figure7.png` (consistent with Figure1–6 and the README).

### Removed
- `HOW_TO_APPLY.txt` (instructions for applying the v1.1 update; not part of the repository).

## 1.1.0 — cross-species extension

### Added
- `colab/cell_13_build_external_species.py`: chicken (*Gallus gallus*) and zebrafish (*Danio rerio*) external sets,
  built with the same rules as the mouse set (cell 10). Strict-subset flags follow the partitioning rule directly,
  computed in the same MMseqs2 search as the identity × coverage score.
- `colab/cell_14_external_species_evaluation.py`: human-trained TensorFlow models (five seeds) applied without
  retraining, plus composition-preserving motif shuffles (as cell 11).
- `colab/cell_15_external_species_dnabert.py`: DNABERT-6 fine-tuned on human data (three seeds) and applied to all
  external sets; mouse is re-predicted as a consistency check; fine-tuned weights are kept.
- `analysis/external_species_analysis.py`: species-agnostic version of `mouse_analysis.py` with command-line paths;
  adds balanced accuracy. Verified to reproduce Table 6 (mouse) from `Cell11_mouse_predictions.npz`.
- `colab/cell_16_external_baselines.py`: length-only, length+GC+structure and 1-NN baselines for all external
  species (mouse recomputed as a check against Table 6), and a table of labels of human 5S rRNA-family records.
- `analysis/fold_external.py`, `analysis/cross_species_summary.py` (cross-species table and Figure 7).

### Changed
- README, `data/README.md` and `CITATION.cff` (title updated to the current manuscript; version 1.1.0).

### Removed
- `figures/Figure7.png`: an outdated version of the mouse figure (strict subset n = 9,215, from the identity × coverage
  score) that was inconsistent with the manuscript (n = 8,377, partitioning rule). `Figure7.png` is now the
  cross-species figure produced by `analysis/cross_species_summary.py`.

### To do afte### Results added
- `data/zebrafish_external.csv`; `results/predictions/Cell14_{chicken,zebrafish}_predictions.npz`,
  `Cell16_{mouse,chicken,zebrafish}_baselines.npz`, `runs_fm_ext_multi/` (DNABERT, Cell 15).
- `results/tables/`: `cross_species_{metrics,contrasts,rRNA,composition}.csv`, `supp_S21–S23_*.csv`,
  `Cell13_*_summary.json`, `Cell14_*_motif_shuffle.csv`, `Cell15_dnabert_external_runs.csv`,
  `Cell16_baselines_summary.csv`, `Cell16_human_5S_family_labels.csv`.
- `figures/Figure7.png` and `analysis/cross_species_final.py`, which produces them.
- Checks: Cell 16 reproduces the mouse baselines of Table 6 exactly (0.361 / 0.380 / 0.505); the re-fine-tuned
  DNABERT gives 0.819 (human) and 0.630 (mouse) versus 0.813 and 0.624 before.

### Still to add by hand
- `data/chicken_external.csv` (in Google Drive, MyDrive/JBSD_Revision/).
