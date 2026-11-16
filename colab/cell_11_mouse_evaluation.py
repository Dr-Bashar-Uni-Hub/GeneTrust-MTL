# ======================================================================
# CELL 11 — EXTERNAL VALIDATION ON MOUSE: TensorFlow models  (post hoc)
# Run in a TensorFlow session: Cell 1, then Cell 3, then this cell
# (needs mouse_external.csv from Cell 10 in the Drive folder). ~10–15 min.
# Models trained on the HUMAN 80% partition are applied unchanged (no
# retraining). Also repeats the motif-perturbation test on mouse sequences.
# ======================================================================
import numpy as np, pandas as pd, re, json, time
from scipy.stats import wilcoxon
if not ("transformer" in globals() and "enc" in globals() and isinstance(globals().get("Y", {}).get("test"), dict)):
    raise RuntimeError("Wrong session: run Cell 1 and Cell 3 first (after Runtime > Restart session).")

M = pd.read_csv(f"{OUT}/mouse_external.csv")
M = M[M.GeneType.isin(le.classes_)].reset_index(drop=True)
Xm = np.array([enc(s) for s in M.seq], dtype=np.int32)
ym = le.transform(M.GeneType)
BEST_LR = json.load(open(f"{OUT}/Cell03_best_lr.json")); SEEDS = [1, 7, 21, 42, 100]
print(f"Mouse genes: {len(M):,}", M.GeneType.value_counts().to_dict())

# ---- orientation sanity check: tRNA promoter boxes at canonical positions
def find(seq, pat, lo, hi):
    m = [x.start() for x in re.finditer(f"(?=({pat}))", seq) if lo <= x.start() <= hi]
    return m[0] if m else -1
tr = M[M.GeneType == "tRNA"].seq
a = [find(s, "T[AG]G[CT]..A[AG]..G", 0, 30) for s in tr]; b = [find(s, "GTTC[AG]A.[CT]C", 40, len(s)) for s in tr]
print(f"Orientation check (mouse tRNA, n={len(tr)}): A box found {np.mean([x >= 0 for x in a]):.0%}, "
      f"B box found {np.mean([x >= 0 for x in b]):.0%} (human test set: 90% / 94%)")

# ---- predictions of all models, all seeds -----------------------------
res = {"gene_ids": M.NCBIGeneID.values, "y_true": ym, "classes": le.classes_,
       "maxid": M.maxid_to_human_train.values, "component": M.mouse_component.values}
names = ["MTL_Transformer_PE", "MTL_Transformer", "STL_GeneType", "MTL_BiLSTM", "MTL_CNN"]
for name in names:
    for seed in SEEDS:
        tf.keras.backend.clear_session(); m = MODELS[name][0]()
        m.load_weights(f"{RUNS}/{name}_lr{BEST_LR[name]:g}_s{seed}/model.weights.h5")
        p = m.predict(Xm, batch_size=128, verbose=0); p = p[0] if isinstance(p, list) else p
        res[f"prob__{name}_s{seed}"] = p.astype(np.float32)
    acc = np.mean([(res[f"prob__{name}_s{s}"].argmax(1) == ym).mean() for s in SEEDS])
    print(f"  {name:20s} mouse Gene Type accuracy (5 seeds): {acc:.4f}")

# ---- motif perturbation on mouse tRNA / snoRNA (as Cells 6 and 9) -------
MOT = {"tRNA A box": ("tRNA", "T[AG]G[CT]..A[AG]..G", 11, lambda s: (0, 30)),
       "tRNA B box": ("tRNA", "GTTC[AG]A.[CT]C", 9, lambda s: (40, len(s))),
       "snoRNA C box": ("snoRNA", "[AG]TGATGA", 7, lambda s: (0, 20)),
       "snoRNA D box": ("snoRNA", "CTGA", 4, lambda s: (len(s) - 20, len(s) - 4))}
def shuffle_win(seq, st, w, rng):
    win = seq[st:st + w]
    for _ in range(50):
        new = "".join(rng.permutation(list(win)))
        if new != win: break
    return seq[:st] + new + seq[st + w:]
rng = np.random.default_rng(2026); inst, vs, vi, vk = [], [], [], []
for i in np.where(M.GeneType.isin(["tRNA", "snoRNA"]).values)[0]:
    s = M.seq.iloc[i][:517]; c = M.GeneType.iloc[i]
    found = {k: find(s, p, *r(s)) for k, (cc, p, w, r) in MOT.items() if cc == c}
    occ = [(st, st + MOT[k][2]) for k, st in found.items() if st >= 0]
    for k, st in found.items():
        if st < 0: continue
        w = MOT[k][2]; j = len(inst); inst.append((i, k))
        for _ in range(5): vs.append(shuffle_win(s, st, w, rng)); vi.append(j); vk.append(0)
        cand = [q for q in range(0, len(s) - w + 1) if all(q + w <= a_ or q >= b_ for a_, b_ in occ)]
        for q in rng.choice(cand, size=min(5, len(cand)), replace=False):
            vs.append(shuffle_win(s, int(q), w, rng)); vi.append(j); vk.append(1)
vi, vk = np.array(vi), np.array(vk); rows = np.array([r for r, _ in inst]); mot = np.array([k for _, k in inst])
Xv = np.array([enc(s) for s in vs], dtype=np.int32)
print(f"\nPerturbation (composition-preserving shuffle): {len(inst)} motif instances",
      {k: int((mot == k).sum()) for k in MOT})
res.update({"pert_rows": rows, "pert_motifs": mot})
for name in ["MTL_Transformer_PE", "MTL_Transformer"]:
    for seed in SEEDS:
        tf.keras.backend.clear_session(); m = MODELS[name][0]()
        m.load_weights(f"{RUNS}/{name}_lr{BEST_LR[name]:g}_s{seed}/model.weights.h5")
        po = m.predict(Xm[rows], batch_size=128, verbose=0)[0]; pv = m.predict(Xv, batch_size=128, verbose=0)[0]
        yt = ym[rows]; p0 = po[np.arange(len(rows)), yt]; pvt = pv[np.arange(len(Xv)), yt[vi]]
        ok0 = po.argmax(1) == yt; okv = pv.argmax(1) == yt[vi]
        for kind, lab in [(0, "motif"), (1, "ctrl")]:
            res[f"d{lab}__{name}_s{seed}"] = np.array([pvt[(vi == j) & (vk == kind)].mean() for j in range(len(rows))]) - p0
            res[f"flip{lab}__{name}_s{seed}"] = np.array([(ok0[j] & ~okv[(vi == j) & (vk == kind)]).mean() for j in range(len(rows))])
for name in ["MTL_Transformer_PE", "MTL_Transformer"]:
    dm = np.mean([res[f"dmotif__{name}_s{s}"] for s in SEEDS], 0); dc = np.mean([res[f"dctrl__{name}_s{s}"] for s in SEEDS], 0)
    fm = np.mean([res[f"flipmotif__{name}_s{s}"] for s in SEEDS], 0); fc = np.mean([res[f"flipctrl__{name}_s{s}"] for s in SEEDS], 0)
    for k in MOT:
        q = mot == k
        if q.sum() < 6: print(f"  {name:20s} {k:13s} n={q.sum()} (too few)"); continue
        print(f"  {name:20s} {k:13s} n={q.sum():3d}  Δp motif {np.median(dm[q]):+.3f} vs control {np.median(dc[q]):+.3f}  "
              f"flips {fm[q].mean():.3f} vs {fc[q].mean():.3f}  p={wilcoxon(dm[q], dc[q]).pvalue:.2e}")
np.savez_compressed(f"{OUT}/Cell11_mouse_predictions.npz", **res)
print(f"\nSaved {OUT}/Cell11_mouse_predictions.npz — please upload it with mouse_external.csv and the printed output.")
