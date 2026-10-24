Supplementary Data — Memorization, length shortcuts, and genuine sequence learning (JBSD submission)
All analyses are post hoc relative to the original evaluation.

colab_cells/        Colab cells run on GPU: 01 reproduction check + leakage audit; 02 homology-aware split (MMseqs2);
                    03 training harness + learning-rate selection; 04 five-seed runs; 05 Integrated Gradients; 06 in silico mutagenesis;
                    07 60% identity sensitivity analysis; 08 DNABERT fine-tuning (PyTorch, run alone);
                    09 composition-preserving motif shuffle; 10 mouse external dataset (NCBI Datasets);
                    11 mouse evaluation (TensorFlow); 12 DNABERT on mouse (PyTorch, run alone).
analysis_scripts/   Local analyses: classical baselines (classic.py), cluster-bootstrap statistics (stats.py),
                    identity bins and per-class F1 (bins.py), RNA folding (fold.py), structure baseline
                    (rna_structure_analysis.py, struct_ci.py), combiner (stack.py), motif tests (motif.py), figures.
results/            split_homology80.csv (gene ID, homology component, partition, identity to training),
                    per-run metrics (Cell04_seed_runs.csv), LR search, contrasts, accuracy with CIs,
                    motif statistics, structure contrasts, test labels.
Figure1-5.png       Main-text figures at 300 dpi.
Original data: Kaggle "DNA Sequence Prediction" (Harshvardhan21); not redistributed.
results/mouse_external/  Mouse external set (gene IDs, sequences, labels, identity to human training), predictions.
