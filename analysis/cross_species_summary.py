"""Cross-species summary: human held-out test (homology-aware, 80%) and every external species with a
<species>_accuracy.csv (from external_species_analysis.py), ordered by divergence time from human.
Writes cross_species_summary.csv and Figure 7 (accuracy along the divergence gradient).
  python analysis/cross_species_summary.py --tables results/tables --fig figures/Figure7_cross_species.png
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DIVERGENCE_MYA = {"human": 0, "mouse": 87, "chicken": 319, "zebrafish": 429}   # TimeTree 5 median estimates
SHOW = ["DNABERT", "Transformer+PE", "Transformer (order-blind)", "Single-task Transformer", "1-NN lookup", "Length only"]
STYLE = {"DNABERT": ("#7b3294", "o", "-"), "Transformer+PE": ("#1b7837", "s", "-"),
         "Transformer (order-blind)": ("#5aae61", "^", "--"), "Single-task Transformer": ("#a6dba0", "v", ":"),
         "1-NN lookup": ("#2166ac", "D", "-."), "Length only": ("#b2182b", "x", "-")}

def human_row(t):
    """Human homology-aware (80%) test accuracies from the existing tables."""
    s = pd.read_csv(f"{t}/Cell04_summary.csv", header=[0, 1], index_col=0)
    g = lambda m: float(s.loc[m, ("GT_acc", "mean")])
    ab = pd.read_csv(f"{t}/abs_acc.csv").set_index("model").gt_all.str.split(" ").str[0].astype(float)
    d = {"Transformer+PE": g("MTL_Transformer_PE"), "Transformer (order-blind)": g("MTL_Transformer"),
         "Single-task Transformer": g("STL_GeneType"), "MTL BiLSTM": g("MTL_BiLSTM"), "MTL CNN": g("MTL_CNN"),
         "Length only": ab["Length-only GBM"], "1-NN lookup": ab["1-NN 6-mer"]}
    if os.path.exists(f"{t}/Cell08_dnabert_runs.csv"): d["DNABERT"] = pd.read_csv(f"{t}/Cell08_dnabert_runs.csv").GT_acc.mean()
    return d

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--fig", default="figures/Figure7_cross_species.png"); ap.add_argument("--species", nargs="*")
    a = ap.parse_args()
    sp_list = a.species or [s for s in ["mouse", "chicken", "zebrafish"] if os.path.exists(f"{a.tables}/{s}_accuracy.csv")]
    rows = [{"species": "human", "subset": "held-out test (80%)", "model": k, "acc": v} for k, v in human_row(a.tables).items()]
    for sp in sp_list:
        T = pd.read_csv(f"{a.tables}/{sp}_accuracy.csv")
        for _, r in T.iterrows():
            rows.append({"species": sp, "subset": "all", "model": r.model, "acc": r["acc all"], "balanced_acc": r.get("balanced acc all", np.nan)})
            rows.append({"species": sp, "subset": "no>=60% human match", "model": r.model, "acc": r["acc no>=60%"]})
    S = pd.DataFrame(rows); S["divergence_mya"] = S.species.map(DIVERGENCE_MYA)
    W = S.pivot_table(index="model", columns=["species", "subset"], values="acc").round(3)
    W = W[[c for sp in ["human"] + sp_list for c in W.columns if c[0] == sp]]
    print(W.to_string()); W.to_csv(f"{a.tables}/cross_species_summary.csv")
    # ---- gap over length-only (pp), the key shortcut-free transfer quantity -----
    gap = S[S.subset.isin(["all", "held-out test (80%)"])].pivot_table(index="species", columns="model", values="acc")
    gap = (100 * gap.sub(gap["Length only"], axis=0)).round(1).reindex(["human"] + sp_list)
    print("\nAdvantage over length-only baseline (percentage points):"); print(gap.to_string())
    gap.to_csv(f"{a.tables}/cross_species_gap_over_length.csv")
    # ---- Figure 7 ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    order = ["human"] + sp_list; x = [DIVERGENCE_MYA[s] for s in order]
    for m in SHOW:
        c, mk, ls = STYLE[m]
        ya = [S[(S.species == s) & (S.model == m) & S.subset.isin(["all", "held-out test (80%)"])].acc.mean() for s in order]
        ax[0].plot(x, ya, color=c, marker=mk, ls=ls, label=m, lw=1.6, ms=6)
        ys = [np.nan] + [S[(S.species == s) & (S.model == m) & (S.subset == "no>=60% human match")].acc.mean() for s in sp_list]
        ax[1].plot(x, ys, color=c, marker=mk, ls=ls, lw=1.6, ms=6)
    for k, t in enumerate(["(a) All genes (human: homology-aware test set)", "(b) Genes without a ≥60%-identity human training match"]):
        ax[k].set_xticks(x); ax[k].set_xticklabels([f"{s}\n({DIVERGENCE_MYA[s]} Mya)" if s != "human" else "human\n(held-out)" for s in order])
        ax[k].set_xlabel("Divergence from human"); ax[k].set_ylabel("Gene Type accuracy"); ax[k].set_title(t, fontsize=10, loc="left")
        ax[k].set_ylim(0, 1); ax[k].grid(alpha=0.3)
    ax[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout(); os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True); fig.savefig(a.fig, dpi=300)
    print("saved", a.fig)

if __name__ == "__main__":
    main()
