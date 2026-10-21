# ======================================================================
# CELL 9 — COMPOSITION-PRESERVING MOTIF PERTURBATION (post hoc; ~10–15 min)
# Optional control for Cell 6: nucleotides inside the motif (and inside each
# control window) are SHUFFLED rather than substituted, so local nucleotide
# composition is preserved and only the order is changed.
# Run Cell 1, then Cell 3 first (Cell 3 skips finished runs in ~2 min).
# For each test tRNA/snoRNA containing a canonical motif, every motif
# nucleotide is substituted by a different random nucleotide (5 replicates).
# Control: the same substitution applied to random windows of equal length
# elsewhere in the same sequence that overlap no detected motif (5 windows).
# Sequence length is unchanged, so the length shortcut cannot explain effects.
# Output: change in probability of the TRUE Gene Type, and prediction flips.
# ======================================================================
import numpy as np, re, json, time
from scipy.stats import wilcoxon

# ---- Session check: needs Cell 1 + Cell 3 (TensorFlow) in THIS session ----
if not ("transformer" in globals() and "enc" in globals() and isinstance(globals().get("Y", {}).get("test"), dict)):
    raise RuntimeError("Wrong session: this cell needs the variables from Cell 1 and Cell 3. "
                       "Do Runtime > Restart session, then run Cell 1, Cell 3, and this cell. "
                       "(Do not run it after Cell 8, which uses PyTorch and different variables.)")

BEST_LR = json.load(open(f"{OUT}/Cell03_best_lr.json"))
SEEDS = [1, 7, 21, 42, 100]
N_REP, N_CTRL = 5, 5
MOT = {  # name: (class, regex, width, allowed start range)
    "tRNA A box":   ("tRNA",   "T[AG]G[CT]..A[AG]..G", 11, lambda s: (0, 30)),
    "tRNA B box":   ("tRNA",   "GTTC[AG]A.[CT]C",      9,  lambda s: (40, len(s))),
    "snoRNA C box": ("snoRNA", "[AG]TGATGA",           7,  lambda s: (0, 20)),
    "snoRNA D box": ("snoRNA", "CTGA",                 4,  lambda s: (len(s) - 20, len(s) - 4)),
}
def find(seq, pat, lo, hi):
    m = [x.start() for x in re.finditer(f"(?=({pat}))", seq) if lo <= x.start() <= hi]
    return m[0] if m else -1
def mutate(seq, st, w, rng):
    # composition-preserving: permute the window's own nucleotides; retry until
    # the window differs from the original (windows of one repeated base cannot change)
    win = seq[st:st + w]
    for _ in range(50):
        new = "".join(rng.permutation(list(win)))
        if new != win: break
    return seq[:st] + new + seq[st + w:]

rng = np.random.default_rng(2026)
te, ytrue = S["test"], Y["test"]["gene_type"]
inst, var_seq, var_inst, var_kind = [], [], [], []
for i in np.where(te.GeneType.isin(["tRNA", "snoRNA"]).values)[0]:
    s = te.seq.iloc[i][:517]; c = te.GeneType.iloc[i]
    found = {m: find(s, p, *r(s)) for m, (k, p, w, r) in MOT.items() if k == c}
    occupied = [(st, st + MOT[m][2]) for m, st in found.items() if st >= 0]
    for m, st in found.items():
        if st < 0: continue
        w = MOT[m][2]; k = len(inst); inst.append((i, m, st))
        for _ in range(N_REP):
            var_seq.append(mutate(s, st, w, rng)); var_inst.append(k); var_kind.append(0)
        cand = [j for j in range(0, len(s) - w + 1)
                if all(j + w <= a or j >= b for a, b in occupied)]
        for j in rng.choice(cand, size=min(N_CTRL, len(cand)), replace=False):
            var_seq.append(mutate(s, int(j), w, rng)); var_inst.append(k); var_kind.append(1)
var_inst, var_kind = np.array(var_inst), np.array(var_kind)
rows = np.array([r for r, _, _ in inst]); motifs = np.array([m for _, m, _ in inst])
Xv = np.array([enc(s) for s in var_seq], dtype=np.int32)
print(f"Motif instances: {len(inst)} | variants: {len(Xv)} |",
      {m: int((motifs == m).sum()) for m in MOT})

res = {"rows": rows, "motifs": motifs, "starts": np.array([s for _, _, s in inst]),
       "gene_ids": te.NCBIGeneID.values[rows], "y_true": ytrue[rows]}
for name in ["MTL_Transformer_PE", "MTL_Transformer"]:
    for seed in SEEDS:
        t0 = time.time(); tf.keras.backend.clear_session()
        m = transformer(pos_mask=(name == "MTL_Transformer_PE"))
        m.load_weights(f"{RUNS}/{name}_lr{BEST_LR[name]:g}_s{seed}/model.weights.h5")
        po = m.predict(X["test"][rows], batch_size=128, verbose=0)[0]
        pv = m.predict(Xv, batch_size=128, verbose=0)[0]
        yt = ytrue[rows]; p0 = po[np.arange(len(rows)), yt]
        pvt = pv[np.arange(len(Xv)), yt[var_inst]]
        ok0 = po.argmax(1) == yt; okv = pv.argmax(1) == yt[var_inst]
        d_m = np.array([pvt[(var_inst == k) & (var_kind == 0)].mean() for k in range(len(rows))]) - p0
        d_c = np.array([pvt[(var_inst == k) & (var_kind == 1)].mean() for k in range(len(rows))]) - p0
        f_m = np.array([(ok0[k] & ~okv[(var_inst == k) & (var_kind == 0)]).mean() for k in range(len(rows))])
        f_c = np.array([(ok0[k] & ~okv[(var_inst == k) & (var_kind == 1)]).mean() for k in range(len(rows))])
        tag = f"{name}_s{seed}"
        res.update({f"p0__{tag}": p0, f"dmotif__{tag}": d_m, f"dctrl__{tag}": d_c,
                    f"flipmotif__{tag}": f_m, f"flipctrl__{tag}": f_c})
        print(f"  {tag:26s} done ({time.time()-t0:.0f}s)")

np.savez_compressed(f"{OUT}/Cell09_motif_shuffle.npz", **res)
print("\nSUMMARY (seed-averaged; Δp = change in probability of the true class)")
for name in ["MTL_Transformer_PE", "MTL_Transformer"]:
    dm = np.mean([res[f"dmotif__{name}_s{s}"] for s in SEEDS], 0)
    dc = np.mean([res[f"dctrl__{name}_s{s}"] for s in SEEDS], 0)
    fm = np.mean([res[f"flipmotif__{name}_s{s}"] for s in SEEDS], 0)
    fc = np.mean([res[f"flipctrl__{name}_s{s}"] for s in SEEDS], 0)
    for mo in MOT:
        k = motifs == mo
        p = wilcoxon(dm[k], dc[k]).pvalue if k.sum() > 5 else np.nan
        print(f"  {name:20s} {mo:13s} n={k.sum():3d}  median Δp motif {np.median(dm[k]):+.3f}  "
              f"control {np.median(dc[k]):+.3f}  flips {fm[k].mean():.3f} vs {fc[k].mean():.3f}  p={p:.2e}")
print(f"\nSaved {OUT}/Cell09_motif_shuffle.npz — please upload this file and the printed output.")
