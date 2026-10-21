"""Build the checkpoint archive expected by colab/cell_01 from the public Kaggle files.

Usage:
    python tools/build_checkpoint_zip.py --kaggle_dir /path/to/kaggle_csvs [--out Genomic_Transformer_Checkpoint_20260913_2127.zip]

The Kaggle directory must contain train.csv, validation.csv and test.csv from
https://www.kaggle.com/datasets/harshvardhan21/dna-sequence-prediction
SHA-256 checksums are verified against the files used in the study.
"""
import argparse, hashlib, os, zipfile

SHA256 = {
    "train.csv": "158040253e4975790882c59e7c1fd1756a3bab32dc6e2226aba13e54c39cdb8e",
    "validation.csv": "335bad65e0d69b005b0f3017714bcd9bc2b95ef9ac76ef2a85251798d8dd09a6",
    "test.csv": "ea5cd67d1ea97a36e67ae50b975d14482f20fa5a441af7c3aa279f668ff45d94",
}
ROOT = "Genomic_Transformer_Checkpoint_20260913_2127"
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, "..", "frozen_models")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_dir", required=True)
    ap.add_argument("--out", default=f"{ROOT}.zip")
    a = ap.parse_args()
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, ref in SHA256.items():
            p = os.path.join(a.kaggle_dir, name)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            print(f"{name}: {'checksum OK' if h == ref else 'CHECKSUM DIFFERS (dataset version may have changed)'}")
            z.write(p, f"{ROOT}/data/{name}")
        for m in ["Transformer_Baseline.keras", "CNN.keras", "Step21_MultiTask_BiLSTM.keras"]:
            z.write(os.path.join(FROZEN, m), f"{ROOT}/Results/{m}")
    print("Written:", a.out, "- upload it to Colab (or your Drive) and run colab/cell_01.")

if __name__ == "__main__":
    main()
