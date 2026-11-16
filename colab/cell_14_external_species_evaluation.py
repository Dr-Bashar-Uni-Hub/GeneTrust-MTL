# ======================================================================
# CELL 14 — EXTERNAL VALIDATION ON CHICKEN AND ZEBRAFISH: TF models (post hoc)
# Run in a TensorFlow session: Cell 1, then Cell 3, then this cell (needs
# <species>_external.csv from Cell 13 in the Drive folder). ~10–15 min/species.
# Identical to Cell 11 (mouse): models trained on the HUMAN 80% partition
# are applied unchanged (no retraining), five seeds each, and the
# composition-preserving motif-shuffle test is repeated.
# ======================================================================
SPECIES_TO_EVAL = ["chicken", "zebrafish"]
import os, numpy as np, pandas as pd, re, json
from scipy.stats import wilcoxon
_need = {"OUT (Cell 1)": "OUT" in globals(), "enc (Cell 3)": "enc" in globals(), "le (Cell 3)": "le" in globals(),
         "transformer (Cell 3)": "transformer" in globals(), "MODELS (Cell 3)": "MODELS" in globals(),
         "RUNS (Cell 3)": "RUNS" in globals(),
         "Y['test'] (Cell 3)": isinstance(globals().get("Y"), dict) and isinstance(globals()["Y"].get("test"), dict)}
_missing = [k for k, ok in _need.items() if not ok]
if _missing:
    raise RuntimeError("Not found in this session: " + ", ".join(_missing) + ".\nIn THIS notebook: Runtime > Restart "
                       "session, then run Cell 1, Cell 3 and Cell 14 one after another (no other cells in between). "
                       "If Cell 3 stops with an error, send me that error.")
BEST_LR = json.load(open(f"{OUT}/Cell03_best_lr.json")); SEEDS = [1, 7, 21, 42, 100]
NAMES = ["MTL_Transformer_PE", "MTL_Transformer", "STL_GeneType", "MTL_BiLSTM", "MTL_CNN"]
MOT = {"tRNA A box": ("tRNA", "T[AG]G[CT]..A[AG]..G", 11, lambda s: (0, 30)),
       "tRNA B box": ("tRNA", "GTTC[AG]A.[CT]C", 9, lambda s: (40, len(s))),
       "snoRNA C box": ("snoRNA", "[AG]TGATGA", 7, lambda s: (0, 20)),
       "snoRNA D box": ("snoRNA", "CTGA", 4, lambda s: (len(s) - 20, len(s) - 4))}

def find(seq, pat, lo, hi):
    m = [x.start() for x in re.finditer(f"(?=({pat}))", seq) if lo <= x.start() <= hi]
    return m[0] if m else -1

def shuffle_win(seq, st, w, rng):
    win = seq[st:st + w]
    for _ in range(50):
        new = "".join(rng.permutation(list(win)))
        if new != win: break
    return seq[:st] + new + seq[st + w:]

def load(name, seed):
    tf.keras.backend.clear_session(); m = MODELS[name][0]()
    m.load_weights(f"{RUNS}/{name}_lr{BEST_LR[name]:g}_s{seed}/model.weights.h5"); return m

def evaluate_species(sp):
    path = f"{OUT}/{sp}_external.csv"
    if not os.path.exists(path): print(f"[{sp}] {path} not found — run Cell 13 first; skipped."); return None
    E = pd.read_csv(path)
    absent = sorted(set(E.GeneType) - set(le.classes_))
    if absent: print(f"[{sp}] classes not in the human training label set (dropped, as for mouse):", absent)
    E = E[E.GeneType.isin(le.classes_)].reset_index(drop=True)
    Xe = np.array([enc(s) for s in E.seq], dtype=np.int32); ye = le.transform(E.GeneType)
    print(f"\n[{sp}] genes: {len(E):,}", E.GeneType.value_counts().to_dict(),
          f"| strict80 {int(E.strict80.sum()):,} | strict60 {int(E.strict60.sum()):,}")
    res = {"gene_ids": E.NCBIGeneID.values, "y_true": ye, "classes": le.classes_,
           "maxid": E.maxid_to_human_train.values, "component": E[f"{sp}_component"].values,
           "strict80": E.strict80.values, "strict60": E.strict60.values}
    for name in NAMES:
        for seed in SEEDS:
            p = load(name, seed).predict(Xe, batch_size=128, verbose=0); p = p[0] if isinstance(p, list) else p
            res[f"prob__{name}_s{seed}"] = p.astype(np.float32)
        acc = [(res[f"prob__{name}_s{s}"].argmax(1) == ye).mean() for s in SEEDS]
        print(f"  {name:20s} {sp} Gene Type accuracy: {np.mean(acc):.4f} ± {np.std(acc, ddof=1):.4f}")
    # ---- motif perturbation (composition-preserving shuffle, as Cells 9 and 11)
    rng = np.random.default_rng(2026); inst, vs, vi, vk = [], [], [], []
    for i in np.where(E.GeneType.isin(["tRNA", "snoRNA"]).values)[0]:
        s = E.seq.iloc[i][:517]; c = E.GeneType.iloc[i]
        found = {k: find(s, p, *r(s)) for k, (cc, p, w, r) in MOT.items() if cc == c}
        occ = [(st, st + MOT[k][2]) for k, st in found.items() if st >= 0]
        for k, st in found.items():
            if st < 0: continue
            w = MOT[k][2]; j = len(inst); inst.append((i, k))
            for _ in range(5): vs.append(shuffle_win(s, st, w, rng)); vi.append(j); vk.append(0)
            cand = [q for q in range(0, len(s) - w + 1) if all(q + w <= a_ or q >= b_ for a_, b_ in occ)]
            for q in rng.choice(cand, size=min(5, len(cand)), replace=False):
                vs.append(shuffle_win(s, int(q), w, rng)); vi.append(j); vk.append(1)
    if inst:
        vi, vk = np.array(vi), np.array(vk); rows = np.array([r for r, _ in inst]); mot = np.array([k for _, k in inst])
        Xv = np.array([enc(s) for s in vs], dtype=np.int32)
        print(f"  motif instances: {len(inst)}", {k: int((mot == k).sum()) for k in MOT})
        res.update({"pert_rows": rows, "pert_motifs": mot})
        for name in ["MTL_Transformer_PE", "MTL_Transformer"]:
            for seed in SEEDS:
                m = load(name, seed)
                po = m.predict(Xe[rows], batch_size=128, verbose=0)[0]; pv = m.predict(Xv, batch_size=128, verbose=0)[0]
                yt = ye[rows]; p0 = po[np.arange(len(rows)), yt]; pvt = pv[np.arange(len(Xv)), yt[vi]]
                ok0 = po.argmax(1) == yt; okv = pv.argmax(1) == yt[vi]
                for kind, lab in [(0, "motif"), (1, "ctrl")]:
                    res[f"d{lab}__{name}_s{seed}"] = np.array([pvt[(vi == j) & (vk == kind)].mean() for j in range(len(rows))]) - p0
                    res[f"flip{lab}__{name}_s{seed}"] = np.array([(ok0[j] & ~okv[(vi == j) & (vk == kind)]).mean() for j in range(len(rows))])
        rows_out = []
        for name in ["MTL_Transformer_PE", "MTL_Transformer"]:
            dm = np.mean([res[f"dmotif__{name}_s{s}"] for s in SEEDS], 0); dc = np.mean([res[f"dctrl__{name}_s{s}"] for s in SEEDS], 0)
            fm = np.mean([res[f"flipmotif__{name}_s{s}"] for s in SEEDS], 0); fc = np.mean([res[f"flipctrl__{name}_s{s}"] for s in SEEDS], 0)
            for k in MOT:
                q = mot == k
                if q.sum() < 6: print(f"  {name:20s} {k:13s} n={q.sum()} (too few)"); continue
                p = wilcoxon(dm[q], dc[q]).pvalue
                rows_out.append([sp, name, k, int(q.sum()), np.median(dm[q]), np.median(dc[q]), fm[q].mean(), fc[q].mean(), p])
                print(f"  {name:20s} {k:13s} n={q.sum():4d}  Δp motif {np.median(dm[q]):+.3f} vs control {np.median(dc[q]):+.3f}  "
                      f"flips {fm[q].mean():.3f} vs {fc[q].mean():.3f}  p={p:.2e}")
        pd.DataFrame(rows_out, columns=["species", "model", "motif", "n", "median_dp_motif", "median_dp_ctrl",
                                        "flip_motif", "flip_ctrl", "wilcoxon_p"]).to_csv(f"{OUT}/Cell14_{sp}_motif_shuffle.csv", index=False)
    np.savez_compressed(f"{OUT}/Cell14_{sp}_predictions.npz", **res)
    print(f"[{sp}] saved Cell14_{sp}_predictions.npz")
    return res

EXT_RESULTS = {sp: evaluate_species(sp) for sp in SPECIES_TO_EVAL}

print("\nPlease send me the printed output and, from Drive: Cell14_<species>_predictions.npz, Cell14_<species>_motif_shuffle.csv.")
