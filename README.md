# GeneTrust: Leakage-Controlled Genomic AI Benchmark

**GeneTrust** is a leakage-controlled, cross-species benchmark for gene-type classifiers: homology-aware human partitions (80% and 60% identity, ≥80% coverage), external sets of 12,781 mouse genes and of chicken and zebrafish genes (87–429 million years of divergence from human), baselines, reference-model predictions and diagnostics.

**GeneTrust: Successive Evaluation Controls Change the Conclusions Drawn about Genomic Sequence Classifiers.**

This repository provides the code, partitions, external data, predictions, and result tables supporting the manuscript

> ALHAJAHMAD, B. *GeneTrust: Successive Evaluation Controls Change the Conclusions Drawn about Genomic Sequence Classifiers* (manuscript prepared for submission).

All analyses are post hoc relative to the original evaluation of the model.

## Key findings and AI evaluation implications

| Finding | Evidence |
|---|---|
| The provided Kaggle partitions leak most test sequences | 100% of validation and 85.8% of test records are identical to training records; a 1-nearest-neighbour 6-mer lookup beats the Transformer (95.7% vs 92.9%) |
| Under homology-aware evaluation, an order-blind Transformer ≈ sequence length | 73.9% vs 73.3% Gene Type accuracy; driven by 295-nt Sharpr-MPRA reporter tiles (23% of the test set) |
| Models learn transferable sequence content | On 12,781 external **mouse** genes (no length artifact) all sequence models beat length baselines by ~20 points |
| Positional information and multi-task learning help | Consistent gains at 80% and 60% identity thresholds and in mouse |
| The order-aware model depends on canonical RNA motifs | Shuffling the tRNA B box reverses 78% of correct predictions (control: 4%); position-defined A/D boxes are used mainly when positional information is available; replicated in mouse |
| Explicit RNA-folding features | Help on the human test set only; do not replicate at 60% identity or in mouse |
| Cross-species generalization (v1.1) | On classes in the human test set, the advantage over length is smaller in the more distant species (order-aware Transformer: 21.5 / 18.9 / 10.9 points in mouse / chicken / zebrafish; DNABERT: 26.8 / 25.4 / 18.9); the original order-blind model no longer beats length in zebrafish. With three species that also differ in composition and annotation, this trend is descriptive |
| Majority-annotation class (v1.1) | 80% of human 5S rRNA-family records are labeled pseudogenes; on zebrafish 5S genes, lookup follows sequence similarity (93% rRNA), DNABERT the label prior (95% pseudogene), and Transformer seeds range from 8% to 98% rRNA — a divergence invisible to the human test set, which has no rRNA |
| Ablation (v1.3) | The order-aware gain needs both components: on the human test set, positional embedding alone +2.0 points, padding masks alone −7.2 (below the length baseline), both +4.0; only the combination beats the original model in all three external species |
| Independent benchmark (v1.3) | On Genomic Benchmarks, 38.0% of enhancer test sequences occur in training (full-length, label-consistent); homology-aware re-splitting drops 1-NN lookup from 0.828 to 0.627 and widens the CNN's lead from 1.6 to 19.1 points; on promoters, lookup beats the CNN under both splits |

## Repository layout

```
colab/            19 Google Colab cells (run order below)
analysis/         local post hoc analyses (baselines, cluster bootstrap, RNA folding, motif tests, figures)
data/             homology-aware partitions (80%, 60%) and the external datasets (mouse; chicken and zebrafish from cell 13)  -> see data/README.md
frozen_models/    frozen seed-42 models of the original study (for the reproduction check)
tools/            build_checkpoint_zip.py: builds the input archive for cell 01 from the Kaggle files
results/tables/   all result tables (per-seed runs, contrasts with CIs, per-class F1, motif statistics, mouse)
results/predictions/  test labels and predictions used for the statistics
figures/          manuscript figures (300 dpi)
```

## Reproducing the study (Google Colab)

1. **Get the data.** Download `train.csv`, `validation.csv`, `test.csv` from the
   [Kaggle dataset](https://www.kaggle.com/datasets/harshvardhan21/dna-sequence-prediction) (not redistributed here).
2. **Build the input archive** (checksums are verified):
   ```bash
   python tools/build_checkpoint_zip.py --kaggle_dir /path/to/kaggle_csvs
   ```
   Upload the resulting `Genomic_Transformer_Checkpoint_20260913_2127.zip` to Colab or to your Google Drive.
3. **Run the cells.** Results are written to the historical experiment directory `MyDrive/JBSD_Revision/` and every training run is resumable. The directory name is retained for reproducibility; it does not indicate the target journal.

| Session | Cells | Purpose | GPU time (approx.) |
|---|---|---|---|
| A — TensorFlow | 01 | Reproduction check of frozen results + leakage audit | minutes |
| | 02 | Homology-aware partition at 80% identity (MMseqs2) | CPU, 10–20 min |
| | 03 | Training harness, smoke test, learning-rate selection | 1–2.5 h (T4) |
| | 04 | Five-seed runs of all models | 2–4 h (T4) |
| | 05 | Integrated Gradients (tRNA/snoRNA) | ~10 min |
| | 06 | In silico motif mutagenesis | ~15 min |
| | 09 | Composition-preserving motif shuffle control | ~15 min |
| | 10 | Build the mouse external dataset (NCBI Datasets) | CPU, 20–40 min |
| | 11 | Evaluate human-trained models on mouse + motif shuffling | ~15 min |
| | 13 | Build the chicken and zebrafish external datasets (NCBI Datasets) | CPU, 20–40 min per species |
| | 14 | Evaluate human-trained models on chicken and zebrafish + motif shuffling | ~10–15 min per species |
| | 16 | Length-only, folding and 1-NN baselines on all external species; human 5S-family label check | CPU, 15–25 min |
| | 07 | 60% identity sensitivity analysis (**replaces session variables — run last**) | 1.5–2 h |
| B — PyTorch (alone) | 08 | DNABERT-6 fine-tuning baseline | 30–45 min (A100) |
| C — PyTorch (alone) | 12 | DNABERT-6 on the mouse set | 20–30 min (A100) |
| D — PyTorch (alone) | 15 | DNABERT-6 on all external sets (mouse, chicken, zebrafish); keeps fine-tuned weights | 25–40 min (A100) |
| E — standalone | 17 | Ablation of the order-aware modification (positional embedding × padding masks) | GPU, 1.5–2.5 h |
| F — standalone | 18 | GeneTrust protocol on Genomic Benchmarks (enhancers with mouse enhancers as external set; promoters) | GPU, 1–2 h |
| G — standalone | 19 | Length and label agreement of duplicated Genomic Benchmarks test sequences | CPU, 10–20 min |

**Important:** TensorFlow reserves the GPU memory, so the PyTorch cells (08, 12) must each run in a fresh session
(*Runtime → Restart session*). After a restart, re-run cell 01 and cell 03 before any other TensorFlow cell
(cell 03 skips finished runs in ~2 min). Cell 13 finds the human data itself and needs `split_homology80.csv`. Cells 14 and 16 run after cells 01 and 03 in the same notebook. Cells 17, 18 and 19 are standalone: run each in its own notebook. To skip partitioning, copy `data/split_homology80.csv` into
`MyDrive/JBSD_Revision/` instead of running cell 02.

## Local analyses

`analysis/` contains the scripts used for the classical and biophysical baselines (length-only, logistic regression,
1-NN, ViennaRNA folding features), homology-cluster bootstrap statistics, identity-binned accuracy, motif enrichment,
the 60%-identity and mouse analyses, and the figures. They were run in the author's environment and use absolute
input paths; edit the paths at the top of each script. Requirements: `requirements-analysis.txt`.

## Cross-species extension (cells 13–15)

After cells 13–15, run locally (paths are command-line arguments; no editing needed):

```bash
# ViennaRNA features for each new species (and once for the human train/val sequences, if not already available)
python analysis/fold_external.py --ext data/chicken_external.csv   --out mfe200_chicken.pkl
python analysis/fold_external.py --ext data/zebrafish_external.csv --out mfe200_zebrafish.pkl

# Accuracy, balanced accuracy, cluster-bootstrap contrasts and per-class F1 (repeat for zebrafish; mouse reproduces Table 6)
python analysis/external_species_analysis.py --species chicken \
    --ext data/chicken_external.csv --pred results/predictions/Cell14_chicken_predictions.npz \
    --kaggle_dir /path/to/kaggle --split data/split_homology80.csv \
    --runs_dir /path/to/MyDrive/JBSD_Revision/runs --best_lr results/tables/Cell03_best_lr.json \
    --dnabert_dir /path/to/MyDrive/JBSD_Revision/runs_fm_ext_multi \
    --mfe_human mfe200.pkl --mfe_ext mfe200_chicken.pkl --out results/tables

# Final cross-species analysis used in the manuscript (Table 8, Figure 7, Tables S21–S23)
# inputs: Cell11/Cell14 predictions, Cell16 baselines, Cell15 DNABERT predictions (all in results/predictions)
python analysis/cross_species_final.py --inp results/predictions --tables results/tables --fig figures/Figure7.png
```

For new species the strict-subset flags (`strict80`, `strict60`) in `<species>_external.csv` follow the partitioning rule
directly. For mouse, use `--strict_file results/tables/mouse_strict_rule_based.csv` (see `data/README.md`).

## Requirements

- `requirements-tensorflow.txt` — cells 01–07, 09–11, 13–14, 16–18 (cells 18–19 install `genomic-benchmarks` and MMseqs2 themselves)
- `requirements-pytorch.txt` — cells 08, 12, 15
- `requirements-analysis.txt` — local analyses

## Citation

Please cite the manuscript using the title above until a formal publication record is available. See `CITATION.cff`, and update its title if it still contains the previous manuscript title. Publication details will be added upon publication.

## License

Code: MIT (see `LICENSE`). The original Kaggle data are subject to their own terms and are not included.
The mouse, chicken and zebrafish external datasets are derived from NCBI Gene.
