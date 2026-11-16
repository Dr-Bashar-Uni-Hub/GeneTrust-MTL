"""ViennaRNA features (first 200 nt; MFE, window length, paired fraction) for an external species set,
as analysis/foldm.py does for mouse. Resumable: re-run until it reports all genes folded.
  python analysis/fold_external.py --ext data/chicken_external.csv --out mfe200_chicken.pkl
"""
import argparse, os, pickle, time
import pandas as pd
import RNA

ap = argparse.ArgumentParser(); ap.add_argument("--ext", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max_seconds", type=float, default=float("inf"))
a = ap.parse_args()
M = pd.read_csv(a.ext)
done = pickle.load(open(a.out, "rb")) if os.path.exists(a.out) else {}
t, n = time.time(), 0
for g, q in zip(M.NCBIGeneID, M.seq):
    if g in done: continue
    w = q[:200].replace("T", "U"); ss, mfe = RNA.fold(w); done[g] = (mfe, len(w), ss.count("(") * 2 / len(w)); n += 1
    if n % 2000 == 0: pickle.dump(done, open(a.out, "wb"))
    if time.time() - t > a.max_seconds: break
pickle.dump(done, open(a.out, "wb")); print("folded", n, "total", len(done), "of", len(M))
