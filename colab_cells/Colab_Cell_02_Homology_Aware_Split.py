# ======================================================================
# CELL 2 — HOMOLOGY-AWARE RE-SPLIT  (post hoc; CPU only, ~10–20 min)
# Requires Cell 1 to have run in this session (uses DATA, OUT).
# Pool = unique genes from the original train ∪ test files (validation rows
# are exact copies of training rows). Sequences are clustered with MMseqs2
# and whole homology components are assigned to train / validation / test.
# ======================================================================
import os, subprocess, shutil, json, warnings
warnings.filterwarnings("ignore", message="The least populated class")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ID_THR, COV_THR, SPLIT_SEED = 0.80, 0.80, 42
MM = "/content/mm" if os.path.exists("/content") else "/tmp/mmcell"
shutil.rmtree(MM, ignore_errors=True); os.makedirs(MM)

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:]); raise RuntimeError(cmd)
    return r.stdout

# ---- 1. MMseqs2 -----------------------------------------------------------
if shutil.which("mmseqs") is None:
    sh("wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mm.tgz && "
       "tar xzf /tmp/mm.tgz -C /tmp && cp /tmp/mmseqs/bin/mmseqs /usr/local/bin/")
print(sh("mmseqs version").strip())

# ---- 2. Build the de-duplicated pool ------------------------------------
V = set("ACGT")
raw = pd.concat([pd.read_csv(f"{DATA}/train.csv"), pd.read_csv(f"{DATA}/test.csv")])
raw = raw.drop_duplicates("NCBIGeneID").reset_index(drop=True)
raw["seq"] = raw.NucleotideSequence.map(lambda s: "".join(c for c in str(s).upper() if c in V))
raw["L"] = raw.seq.str.len()
n_short = int((raw.L < 6).sum())
pool = raw[raw.L >= 6].reset_index(drop=True)          # <6 nt yields no 6-mer
pool["uid"] = "g" + pool.NCBIGeneID.astype(str)
print(f"Pool: {len(pool):,} unique genes (removed {n_short} with <6 nt)")
with open(f"{MM}/pool.fa", "w") as f:
    for u, s in zip(pool.uid, pool.seq): f.write(f">{u}\n{s}\n")

# ---- 3. Homology graph: all-vs-all search, connected components --------
# Edge = alignment with >= ID_THR identity covering >= COV_THR of either
# sequence. Connected components guarantee that no such pair crosses splits.
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components
sh(f"mmseqs easy-search {MM}/pool.fa {MM}/pool.fa {MM}/all.m8 {MM}/tmp_all --search-type 3 "
   f"--min-seq-id {ID_THR} -c {COV_THR} --cov-mode 2 --max-seqs 1000 --threads 8 -v 1")
h = pd.read_csv(f"{MM}/all.m8", sep="\t", header=None)
h = h[h[0] != h[1]]
ix = {u: i for i, u in enumerate(pool.uid)}
G = sps.coo_matrix((np.ones(len(h)), (h[0].map(ix).values, h[1].map(ix).values)), shape=(len(pool),) * 2)
n_comp, comp = connected_components(G, directed=False)
pool["cluster"] = comp; mode_used = "connected components of all-vs-all homology graph"
cs = pool.cluster.value_counts()
print(f"Homology components: {len(cs):,}; singletons {int((cs==1).sum()):,}; "
      f"largest {cs.iloc[0]} ({cs.iloc[0]/len(pool):.1%}); top-5 sizes {cs.iloc[:5].tolist()}")
for c in cs.index[:3]:
    print("  component", c, pool.loc[pool.cluster == c, "GeneType"].value_counts().head(3).to_dict())
print("Components per Gene Type:", pool.groupby("GeneType").cluster.nunique().to_dict())

# ---- 4. Stratified group split. Oversized components (>2% of pool) are
#         fixed to training so a single repeat family cannot dominate
#         validation or test; the rest is split ~20% test / ~10% validation.
big = set(cs.index[cs > 0.02 * len(pool)])
rest = pool[~pool.cluster.isin(big)]
tr_i, te_i = next(StratifiedGroupKFold(5, shuffle=True, random_state=SPLIT_SEED)
                  .split(rest, rest.GeneType, rest.cluster))
sub = rest.iloc[tr_i]
_, b_i = next(StratifiedGroupKFold(7, shuffle=True, random_state=SPLIT_SEED)
              .split(sub, sub.GeneType, sub.cluster))
pool["split"] = "train"
pool.loc[rest.index[te_i], "split"] = "test"
pool.loc[sub.index[b_i], "split"] = "val"
assert pool.groupby("cluster").split.nunique().max() == 1, "component crosses splits"
print(f"Components fixed to train: {len(big)} ({int(pool.cluster.isin(big).sum())} genes)")

# ---- 5. Residual-homology check: purge val/test sequences that still hit
#         training at >= ID_THR identity and >= COV_THR query coverage ---------
def write(split):
    p = f"{MM}/{split}.fa"
    with open(p, "w") as f:
        for u, s in pool.loc[pool.split == split, ["uid", "seq"]].values: f.write(f">{u}\n{s}\n")
    return p
ftr, fva, fte = write("train"), write("val"), write("test")
purged = {}
for name, fq in [("val", fva), ("test", fte)]:
    out = f"{MM}/{name}_hits.m8"
    sh(f"mmseqs easy-search {fq} {ftr} {out} {MM}/tmp_{name} --search-type 3 "
       f"--min-seq-id {ID_THR} -c {COV_THR} --cov-mode 2 --threads 8 -v 1")
    hit = set(pd.read_csv(out, sep="\t", header=None)[0]) if os.path.getsize(out) else set()
    purged[name] = len(hit)
    pool.loc[pool.uid.isin(hit), "split"] = f"{name}_purged"
print("Residual high-identity hits purged:", purged)

# ---- 6. Max identity of each test sequence to any training sequence ----
fte = write("test"); out = f"{MM}/test_maxid.m8"
sh(f"mmseqs easy-search {fte} {ftr} {out} {MM}/tmp_id --search-type 3 -s 7.5 "
   f"--max-seqs 300 --threads 8 -v 1")
if os.path.getsize(out):
    h = pd.read_csv(out, sep="\t", header=None)
    h["cov"] = ((h[7] - h[6]).abs() + 1) / h[0].map(dict(zip(pool.uid, pool.L)))   # strand-independent
    h["eff"] = h[2] * h["cov"].clip(upper=1)              # identity x query coverage
    maxid = h.groupby(0).eff.max()
else:
    maxid = pd.Series(dtype=float)
pool["test_maxid_to_train"] = np.where(pool.split == "test", pool.uid.map(maxid).fillna(0.0), np.nan)

# ---- 7. Report and save ---------------------------------------------------
ct = pd.crosstab(pool.GeneType, pool.split)
print("\nClass counts per split:\n", ct)
t = pool.loc[pool.split == "test", "test_maxid_to_train"]
print("\nTest identity-to-train (identity x coverage):",
      pd.cut(t, [-0.01, 0.0001, 0.5, 0.65, 0.8]).value_counts().sort_index().to_dict())
pool[["NCBIGeneID", "uid", "cluster", "split", "L", "test_maxid_to_train"]].to_csv(
    f"{OUT}/split_homology80.csv", index=False)
json.dump({"fixed_to_train_components": len(big), "id_thr": ID_THR, "cov_thr": COV_THR, "clustering": mode_used, "pool": len(pool),
           "removed_short": n_short, "n_clusters": int(len(cs)), "purged": purged,
           "split_sizes": pool.split.value_counts().to_dict()},
          open(f"{OUT}/Cell02_split_summary.json", "w"), indent=1)
print("\nSaved split_homology80.csv and Cell02_split_summary.json to", OUT)
