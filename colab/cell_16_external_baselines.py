# ======================================================================
# CELL 16 — BASELINES ON THE EXTERNAL SPECIES  (post hoc)
# Run AFTER Cell 1 and Cell 3 in the same notebook (like Cell 14). CPU is
# enough; ~15–25 min. Needs <species>_external.csv in the Drive folder.
# Same baselines as for mouse (analysis/mouse_analysis.py), trained on the
# HUMAN 80% training partition only:
#   - Length only (gradient boosting on log length)
#   - Length + GC + structure (ViennaRNA, first 200 nt)
#   - 1-NN lookup (cosine on 6-mer counts, first 517 nt)
# Mouse is recomputed as a check against Table 6 (0.361 / 0.380 / 0.505).
# Also tabulates the labels of human 5S rRNA-family records (label check).
# ======================================================================
SPECIES = ["mouse", "chicken", "zebrafish"]
import os, itertools, pickle, subprocess, sys, time
import numpy as np, pandas as pd
if not ("S" in globals() and isinstance(globals().get("S"), dict) and "le" in globals() and "OUT" in globals()):
    raise RuntimeError("Run Cell 1 and Cell 3 first in this notebook, then this cell.")
try:
    import RNA
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ViennaRNA"], check=True); import RNA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import scipy.sparse as sps

tr, va = S["train"], S["val"]
ytr = le.transform(tr.GeneType)

# ---- 0. label check: human 5S rRNA-family records -------------------------
H = pd.concat([S["train"].assign(part="train"), S["val"].assign(part="val"), S["test"].assign(part="test")], ignore_index=True)
fam5s = H[H.Symbol.astype(str).str.match(r"^RNA5S") | H.Description.astype(str).str.contains("5S ribosomal", case=False)]
print("Human 5S rRNA-family records by label and partition:")
print(pd.crosstab(fam5s.GeneType, fam5s.part, margins=True).to_string())
print("Lengths (median):", fam5s.groupby("GeneType").seq.apply(lambda s: int(s.str.len().median())).to_dict(), "\n")
fam5s[["NCBIGeneID", "Symbol", "Description", "GeneType", "part"]].to_csv(f"{OUT}/Cell16_human_5S_family_labels.csv", index=False)

# ---- 1. features ----------------------------------------------------------
FOLD_CACHE = f"{OUT}/Cell16_mfe200_cache.pkl"
mfe = pickle.load(open(FOLD_CACHE, "rb")) if os.path.exists(FOLD_CACHE) else {}
def fold_feats(keys, seqs):
    t, n = time.time(), 0
    for k, q in zip(keys, seqs):
        if k in mfe: continue
        w = q[:200].replace("T", "U"); ss, e = RNA.fold(w); mfe[k] = (e, len(w), ss.count("(") * 2 / len(w)); n += 1
        if n % 5000 == 0: pickle.dump(mfe, open(FOLD_CACHE, "wb")); print(f"    folded {n:,} ({time.time()-t:.0f} s)", flush=True)
    pickle.dump(mfe, open(FOLD_CACHE, "wb"))
def feats(keys, seqs):
    seqs = list(seqs); L = np.array([len(q) for q in seqs])
    m = np.array([mfe[k] for k in keys])
    gc = np.array([(q[:200].count("G") + q[:200].count("C")) / max(1, len(q[:200])) for q in seqs])
    return np.c_[np.log1p(np.minimum(L, 517)), gc, m[:, 0] / m[:, 1], m[:, 2]]
idx6 = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=6))}
def kmers(seqs):
    r, c = [], []
    for i, q in enumerate(seqs):
        q = q[:517]
        for j in range(len(q) - 5):
            k = idx6.get(q[j:j + 6])
            if k is not None: r.append(i); c.append(k)
    return normalize(sps.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(seqs), 4096)))

print("Folding human training sequences (cached after the first run)...")
hk = [("h", g) for g in tr.NCBIGeneID]; fold_feats(hk, tr.seq)
Ftr = feats(hk, tr.seq)
gbm_len = HistGradientBoostingClassifier(random_state=0).fit(Ftr[:, :1], ytr)
gbm_lgs = HistGradientBoostingClassifier(random_state=0).fit(Ftr, ytr)
nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(kmers(tr.seq.tolist()))

# ---- 2. apply to each species -------------------------------------------------
summary = []
for sp in SPECIES:
    f = f"{OUT}/{sp}_external.csv"
    if not os.path.exists(f): print(f"[{sp}] {f} not found; skipped"); continue
    E = pd.read_csv(f); E = E[E.GeneType.isin(le.classes_)].reset_index(drop=True)
    y = le.transform(E.GeneType); ek = [(sp, g) for g in E.NCBIGeneID]
    print(f"[{sp}] folding {len(E):,} sequences..."); fold_feats(ek, E.seq)
    Fe = feats(ek, E.seq)
    P = {"length_only": gbm_len.predict(Fe[:, :1]), "length_gc_structure": gbm_lgs.predict(Fe),
         "nn1_lookup": ytr[nn.kneighbors(kmers(E.seq.tolist()))[1][:, 0]]}
    np.savez_compressed(f"{OUT}/Cell16_{sp}_baselines.npz", gene_ids=E.NCBIGeneID.values, y_true=y,
                        classes=le.classes_, **P)
    for k, p in P.items():
        summary.append([sp, k, round(float((p == y).mean()), 3)])
    print(f"[{sp}] " + " | ".join(f"{k} {float((p == y).mean()):.3f}" for k, p in P.items()))
print("\nMouse check against Table 6: length_only 0.361, length_gc_structure 0.380, nn1_lookup 0.505")
pd.DataFrame(summary, columns=["species", "baseline", "acc_all"]).to_csv(f"{OUT}/Cell16_baselines_summary.csv", index=False)
print("\nPlease send me the printed output and, from Drive: Cell16_<species>_baselines.npz (3 files), "
      "Cell16_human_5S_family_labels.csv, and Cell13_chicken_summary.json, Cell14_chicken_motif_shuffle.csv.")
