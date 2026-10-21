# Data

| File | Content |
|---|---|
| `split_homology80.csv` | Homology-aware partition at 80% identity: NCBI Gene ID, homology component, partition (train/val/test), length, identity × coverage of each test gene to the training set |
| `split_homology60.csv` | Same, at 60% identity (sensitivity analysis) |
| `mouse_external.csv` | External mouse set (12,781 genes; NCBI Gene via NCBI Datasets): gene ID, symbol, description, Gene Type, sequence, length, identity × coverage to the human training set, within-mouse homology component, strict-subset flags |

The original human dataset (Kaggle, *DNA Sequence Prediction*, user harshvardhan21) is **not redistributed**.
Download `train.csv`, `validation.csv` and `test.csv` from
https://www.kaggle.com/datasets/harshvardhan21/dna-sequence-prediction and verify:

| File | SHA-256 |
|---|---|
| train.csv | 158040253e4975790882c59e7c1fd1756a3bab32dc6e2226aba13e54c39cdb8e |
| validation.csv | 335bad65e0d69b005b0f3017714bcd9bc2b95ef9ac76ef2a85251798d8dd09a6 |
| test.csv | ea5cd67d1ea97a36e67ae50b975d14482f20fa5a441af7c3aa279f668ff45d94 |

The partition files contain only gene identifiers and can be joined to the Kaggle files on `NCBIGeneID`.
