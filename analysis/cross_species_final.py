"""Final cross-species analysis for the manuscript (Section 3.10, Table 8, Figure 7, Supplementary Tables S21-S23).

Inputs (one folder, default results/cross_species):
  Cell11_mouse_predictions.npz, Cell14_{chicken,zebrafish}_predictions.npz   human-trained TF models (5 seeds)
  runs_fm_ext_multi/DNABERT6_GeneType_s{1,21,42}/preds.npz                   DNABERT (Cell 15, 3 seeds)
  Cell16_{mouse,chicken,zebrafish}_baselines.npz                             length / folding / 1-NN baselines
  Cell14_{chicken,zebrafish}_motif_shuffle.csv                               motif-shuffle tests

Primary cross-species metrics use the classes present in the human homology-aware test set ("shared classes"),
because classes absent from it (rRNA, scRNA) were never validated in-distribution and their composition differs
greatly between species (e.g. 31% 5S rRNA in zebrafish). Accuracy on all genes is also reported.
  python analysis/cross_species_final.py --inp results/cross_species --tables results/tables --fig figures/Figure7.png
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SPECIES = ["mouse", "chicken", "zebrafish"]
MYA = {"mouse": 87, "chicken": 319, "zebrafish": 429}          # TimeTree 5 medians
HUMAN_TEST = ["BIOLOGICAL_REGION", "OTHER", "PROTEIN_CODING", "PSEUDO", "ncRNA", "snRNA", "snoRNA", "tRNA"]
TFM = [("MTL_Transformer_PE", "Transformer+PE"), ("MTL_Transformer", "Transformer (order-blind)"),
       ("STL_GeneType", "Single-task Transformer"), ("MTL_BiLSTM", "MTL BiLSTM"), ("MTL_CNN", "MTL CNN")]
BASE = [("nn1_lookup", "1-NN lookup"), ("length_gc_structure", "Length+GC+structure"), ("length_only", "Length only")]
ORDER = ["DNABERT", "Transformer+PE", "Transformer (order-blind)", "Single-task Transformer", "MTL BiLSTM", "MTL CNN",
         "1-NN lookup", "Length+GC+structure", "Length only"]
CONTRASTS = [("Transformer+PE", "Length only"), ("DNABERT", "Length only"), ("Transformer (order-blind)", "Length only"),
             ("Transformer+PE", "Transformer (order-blind)"), ("Transformer (order-blind)", "Single-task Transformer"),
             ("DNABERT", "Transformer+PE"), ("Transformer+PE", "1-NN lookup"), ("DNABERT", "1-NN lookup")]

def load_species(inp, sp, mouse_strict="results/tables/mouse_strict_rule_based.csv"):
    Z = np.load(f"{inp}/{'Cell11_mouse' if sp == 'mouse' else 'Cell14_' + sp}_predictions.npz", allow_pickle=True)
    y, ids, cls = Z["y_true"], Z["gene_ids"], np.array(Z["classes"])
    if sp == "mouse":
        st = pd.read_csv(mouse_strict).set_index("NCBIGeneID").loc[ids]
        s60 = st.no_rule60_match.values.astype(bool)
    else:
        s60 = Z["strict60"].astype(bool)
    P = {lab: [Z[f"prob__{nm}_s{s}"].argmax(1) for s in [1, 7, 21, 42, 100]] for nm, lab in TFM}
    D = [np.load(f"{inp}/runs_fm_ext_multi/DNABERT6_GeneType_s{s}/preds.npz", allow_pickle=True) for s in [1, 21, 42]]
    assert all((d[f"{sp}_gene_ids"] == ids).all() for d in D), f"DNABERT gene order differs ({sp})"
    P["DNABERT"] = [d[f"{sp}_gene_type"].argmax(1) for d in D]
    B = np.load(f"{inp}/Cell16_{sp}_baselines.npz", allow_pickle=True)
    assert (B["gene_ids"] == ids).all() and list(B["classes"]) == list(cls), f"baseline alignment ({sp})"
    for k, lab in BASE: P[lab] = [B[k]]
    return dict(y=y, cls=cls, comp=Z["component"], s60=s60, shared=np.isin(cls[y], HUMAN_TEST), P=P)

def metrics(d):
    y, sh, s60, rows = d["y"], d["shared"], d["s60"], []
    for m in ORDER:
        P = d["P"][m]
        def stat(f):
            v = np.array([f(p) for p in P]); return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan)
        acc_all = stat(lambda p: (p == y).mean())
        acc_sh = stat(lambda p: (p[sh] == y[sh]).mean())
        bacc = stat(lambda p: np.mean([(p[y == c] == c).mean() for c in np.unique(y[sh])]))
        m60 = sh & s60
        acc60 = stat(lambda p: (p[m60] == y[m60]).mean())
        rows.append([m, *acc_all, *acc_sh, *bacc, *acc60])
    return pd.DataFrame(rows, columns=["model", "acc_all", "acc_all_sd", "acc_shared", "acc_shared_sd", "bacc_shared",
                                       "bacc_shared_sd", "acc_shared_no60", "acc_shared_no60_sd"])

def contrasts(d, sp, n_boot=2000):
    y, comp = d["y"], d["comp"]
    corr = {m: np.mean([p == y for p in d["P"][m]], 0) for m in ORDER}
    uc = np.unique(comp); mem = [np.where(comp == c)[0] for c in uc]; rng = np.random.default_rng(0)
    B = [np.concatenate([mem[j] for j in rng.integers(0, len(uc), len(uc))]) for _ in range(n_boot)]
    out = []
    for sub, m in [("shared classes", d["shared"]), ("shared, no >=60% match", d["shared"] & d["s60"]), ("all genes", np.ones(len(y), bool))]:
        for a, b in CONTRASTS:
            dd = corr[a] - corr[b]; bs = np.array([dd[ix[m[ix]]].mean() for ix in B])
            lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            out.append([sp, sub, f"{a} − {b}", round(100 * dd[m].mean(), 1), round(100 * lo, 1), round(100 * hi, 1), round(p, 4), int(m.sum())])
    return pd.DataFrame(out, columns=["species", "subset", "contrast", "diff_pp", "lo", "hi", "p", "n"])

def rrna_table(d, sp):
    y, cls = d["y"], d["cls"]; r = cls[y] == "rRNA"; rows = []
    if r.sum() == 0: return pd.DataFrame()
    for m in ["DNABERT", "Transformer+PE", "Transformer (order-blind)", "MTL CNN", "1-NN lookup", "Length only"]:
        for k, p in enumerate(d["P"][m]):
            rows.append([sp, m, k, int(r.sum()), round((p[r] == np.where(cls == "rRNA")[0][0]).mean(), 3),
                         round((cls[p[r]] == "PSEUDO").mean(), 3)])
    return pd.DataFrame(rows, columns=["species", "model", "seed_index", "n_rRNA", "recall_rRNA", "frac_predicted_PSEUDO"])

def motif_direct(inp, species=("chicken", "zebrafish")):
    """One-sided paired Wilcoxon: is the order-aware net effect (motif minus control, seed-averaged) more negative
    than the order-blind one? Same test as Supplementary Table S18 (mouse); reported in Supplementary Table S21."""
    from scipy.stats import wilcoxon
    S = [1, 7, 21, 42, 100]; rows = []
    for sp in species:
        Z = np.load(f"{inp}/Cell14_{sp}_predictions.npz", allow_pickle=True); mot = Z["pert_motifs"]
        net = {m: np.mean([Z[f"dmotif__{m}_s{s}"] for s in S], 0) - np.mean([Z[f"dctrl__{m}_s{s}"] for s in S], 0)
               for m in ["MTL_Transformer_PE", "MTL_Transformer"]}
        for k in ["tRNA A box", "tRNA B box", "snoRNA C box", "snoRNA D box"]:
            q = mot == k; a, b = net["MTL_Transformer_PE"][q], net["MTL_Transformer"][q]
            rows.append([sp, k, int(q.sum()), round(float(np.median(a)), 3), round(float(np.median(b)), 3),
                         wilcoxon(a, b, alternative="less").pvalue])
    return pd.DataFrame(rows, columns=["species", "motif", "n", "net_PE_median", "net_OB_median", "p_one_sided_PE_more_negative"])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--inp", default="results/predictions")
    ap.add_argument("--tables", default="results/tables"); ap.add_argument("--mouse_strict", default="results/tables/mouse_strict_rule_based.csv"); ap.add_argument("--fig", default="figures/Figure7.png")
    a = ap.parse_args(); os.makedirs(a.tables, exist_ok=True)
    D = {sp: load_species(a.inp, sp, a.mouse_strict) for sp in SPECIES}
    M = pd.concat([metrics(D[sp]).assign(species=sp) for sp in SPECIES])
    C = pd.concat([contrasts(D[sp], sp) for sp in SPECIES])
    R = pd.concat([rrna_table(D[sp], sp) for sp in SPECIES])
    M.round(4).to_csv(f"{a.tables}/cross_species_metrics.csv", index=False)
    C.to_csv(f"{a.tables}/cross_species_contrasts.csv", index=False); R.to_csv(f"{a.tables}/cross_species_rRNA.csv", index=False)
    comp = pd.DataFrame({sp: pd.Series(D[sp]["cls"][D[sp]["y"]]).value_counts() for sp in SPECIES}).fillna(0).astype(int)
    comp.to_csv(f"{a.tables}/cross_species_composition.csv")
    motif_direct(a.inp).to_csv(f"{a.tables}/motif_direct_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(M.round(3).to_string(index=False)); print(); print(C[C.subset != "all genes"].to_string(index=False))
    print(); print(R.groupby(["species", "model"])[["recall_rRNA", "frac_predicted_PSEUDO"]].agg(["mean", "min", "max"]).round(2).to_string())
    print(); print(comp.to_string())
    # ---- Figure 7 ---------------------------------------------------------------
    show = [("DNABERT", "#7b3294", "o", "-"), ("Transformer+PE", "#1b7837", "s", "-"), ("Transformer (order-blind)", "#5aae61", "^", "--"),
            ("Single-task Transformer", "#a6dba0", "v", ":"), ("1-NN lookup", "#2166ac", "D", "-."), ("Length only", "#b2182b", "x", "-")]
    x = [MYA[s] for s in SPECIES]; lab = [f"{s}\n({MYA[s]} Mya)" for s in SPECIES]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    for m, c, mk, ls in show:
        sub = M[M.model == m].set_index("species").loc[SPECIES]
        ax[0].errorbar(x, sub.acc_shared, yerr=sub.acc_shared_sd.fillna(0), color=c, marker=mk, ls=ls, lw=1.6, ms=6, capsize=3, label=m)
        ax[1].errorbar(x, sub.bacc_shared, yerr=sub.bacc_shared_sd.fillna(0), color=c, marker=mk, ls=ls, lw=1.6, ms=6, capsize=3)
    for m, c, mk, off in [("DNABERT", "#7b3294", "o", -6), ("Transformer+PE", "#1b7837", "s", 6)]:
        cc = C[(C.contrast == f"{m} − Length only") & (C.subset == "shared classes")].set_index("species").loc[SPECIES]
        ax[2].errorbar(np.array(x) + off, cc.diff_pp, yerr=[cc.diff_pp - cc.lo, cc.hi - cc.diff_pp], color=c, marker=mk, ls="-", lw=1.6, ms=6, capsize=3, label=m)
    ax[2].axhline(0, color="#b2182b", lw=1, ls="--")
    T = ["(a) Accuracy (classes in human test set)", "(b) Balanced accuracy (same classes)", "(c) Advantage over length-only (pp, 95% CI)"]
    for k in range(3):
        ax[k].set_xticks(x); ax[k].set_xticklabels(lab); ax[k].set_xlabel("Divergence from human"); ax[k].grid(alpha=0.3)
        ax[k].set_title(T[k], fontsize=10, loc="left")
    ax[0].set_ylabel("Gene Type accuracy"); ax[1].set_ylabel("Balanced accuracy"); ax[2].set_ylabel("Percentage points")
    ax[0].set_ylim(0, 0.8); ax[1].set_ylim(0, 0.8); ax[0].legend(fontsize=7.5, frameon=False, loc="lower left"); ax[2].legend(fontsize=8, frameon=False)
    fig.tight_layout(); os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True); fig.savefig(a.fig, dpi=300); print("saved", a.fig)

if __name__ == "__main__":
    main()
