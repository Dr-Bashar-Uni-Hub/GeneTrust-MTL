# GeneTrust-MTL

**Leakage-controlled evaluation of multi-task Transformers for gene-type annotation: memorization, sequence-length shortcuts, homology leakage and genuine sequence learning.**

This repository contains all code, partitions, external data, predictions and result tables for the manuscript

> ALHAJAHMAD, B. *Memorization, length shortcuts, and genuine sequence learning: a leakage-controlled evaluation of multi-task Transformers for gene-type annotation* (under review).

All analyses are post hoc relative to the original evaluation of the model.

## Key findings

| Finding | Evidence |
|---|---|
| The provided Kaggle partitions leak most test sequences | 100% of validation and 85.8% of test records are identical to training records; a 1-nearest-neighbour 6-mer lookup beats the Transformer (95.7% vs 92.9%) |
| Under homology-aware evaluation, an order-blind Transformer ≈ sequence length | 73.9% vs 73.3% Gene Type accuracy; driven by 295-nt Sharpr-MPRA reporter tiles (23% of the test set) |
| Models learn transferable sequence content | On 12,781 external **mouse** genes (no length artifact) all sequence models beat length baselines by ~20 points |
| Positional information and multi-task learning help | Consistent gains at 80% and 60% identity thresholds and in mouse |
| The order-aware model depends on canonical RNA motifs | Shuffling the tRNA B box reverses 78% of correct predictions (control: 4%); position-defined A/D boxes are used mainly when positional information is available; replicated in mouse |
| Explicit RNA-folding features | Help on the human test set only; do not replicate at 60% identity or in mouse |

## Repository layout

```
colab/            12 Google Colab cells (run order below)
analysis/         local post hoc analyses (baselines, cluster bootstrap, RNA folding, motif tests, figures)
data/             homology-aware partitions (80%, 60%) and the mouse external dataset  -> see data/README.md
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
3. **Run the cells.** Results are written to `MyDrive/JBSD_Revision/` and every training run is resumable.

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
| | 07 | 60% identity sensitivity analysis (**replaces session variables — run last**) | 1.5–2 h |
| B — PyTorch (alone) | 08 | DNABERT-6 fine-tuning baseline | 30–45 min (A100) |
| C — PyTorch (alone) | 12 | DNABERT-6 on the mouse set | 20–30 min (A100) |

**Important:** TensorFlow reserves the GPU memory, so the PyTorch cells (08, 12) must each run in a fresh session
(*Runtime → Restart session*). After a restart, re-run cell 01 and cell 03 before any other TensorFlow cell
(cell 03 skips finished runs in ~2 min). To skip partitioning, copy `data/split_homology80.csv` into
`MyDrive/JBSD_Revision/` instead of running cell 02.

## Local analyses

`analysis/` contains the scripts used for the classical and biophysical baselines (length-only, logistic regression,
1-NN, ViennaRNA folding features), homology-cluster bootstrap statistics, identity-binned accuracy, motif enrichment,
the 60%-identity and mouse analyses, and the figures. They were run in the author's environment and use absolute
input paths; edit the paths at the top of each script. Requirements: `requirements-analysis.txt`.

## Requirements

- `requirements-tensorflow.txt` — cells 01–07, 09–11
- `requirements-pytorch.txt` — cells 08, 12
- `requirements-analysis.txt` — local analyses

## Citation

See `CITATION.cff`. The manuscript citation will be added upon publication.

## License

Code: MIT (see `LICENSE`). The original Kaggle data are subject to their own terms and are not included.
The mouse external dataset is derived from NCBI Gene.
