"""External-species analysis (mouse, chicken, zebrafish, ...): accuracy table, cluster-bootstrap
contrasts and per-class F1 for human-trained models applied without retraining.

Generalizes analysis/mouse_analysis.py with relative paths and command-line arguments.
Strict subsets use the partitioning rule (identity >= thr covering >= 80% of EITHER sequence),
as in Table 6. Baselines that need the human training sequences (length-only, folding, 1-NN,
validation-fitted combiners) are computed only when --kaggle_dir is given; DNABERT only when
--dnabert_dir is given. Balanced accuracy (mean per-class recall) is reported in addition to
accuracy because class composition differs between species.

Example (chicken):
  python analysis/external_species_analysis.py --species chicken \
      --ext data/chicken_external.csv --pred results/predictions/Cell14_chicken_predictions.npz \
      --kaggle_dir /path/to/kaggle --split data/split_homology80.csv --runs_dir /path/to/Drive/JBSD_Revision/runs \
      --best_lr results/tables/Cell03_best_lr.json --dnabert_dir /path/to/runs_fm_ext_multi \
      --mfe_human mfe200.pkl --mfe_ext mfe200_chicken.pkl --out results/tables
Mouse (reproduces Table 6): --species mouse --ext data/mouse_external.csv
      --pred results/predictions/Cell11_mouse_predictions.npz --strict_file results/tables/mouse_strict_rule_based.csv
"""
import argparse, itertools, json, os, pickle
import numpy as np, pandas as pd

S5, S3 = [1, 7, 21, 42, 100], [1, 21, 42]
NEURAL = [("MTL_Transformer_PE", "Transformer+PE"), ("MTL_Transformer", "Transformer (order-blind)"),
          ("STL_GeneType", "Single-task Transformer"), ("MTL_BiLSTM", "MTL BiLSTM"), ("MTL_CNN", "MTL CNN")]
CONTRASTS = [("Transformer+PE", "Transformer (order-blind)"), ("Transformer (order-blind)", "Single-task Transformer"),
             ("Transformer (order-blind)", "Length only"), ("Transformer+PE", "Length only"),
             ("Transformer+PE", "Length+GC+structure"), ("Length+GC+structure", "Length only"),
             ("Transformer+PE + biophysical", "Transformer+PE"), ("DNABERT", "Transformer+PE"),
             ("DNABERT + biophysical", "DNABERT"), ("Transformer+PE", "MTL BiLSTM"), ("Transformer+PE", "1-NN lookup")]

def clean(s): return "".join(c for c in str(s).upper() if c in "ACGT")

def load_inputs(a):
    Z = np.load(a.pred, allow_pickle=True)
    cls, y, ids = list(Z["classes"]), Z["y_true"], Z["gene_ids"]
    E = pd.read_csv(a.ext).set_index("NCBIGeneID").loc[ids]
    if a.strict_file:                       # mouse: rule-based flags kept in a separate table
        st = pd.read_csv(a.strict_file).set_index("NCBIGeneID").loc[ids]
        s80, s60 = st.no_rule80_match.values.astype(bool), st.no_rule60_match.values.astype(bool)
    else:                                   # Cell 13 output: strict flags are already rule-based
        s80, s60 = E.strict80.values.astype(bool), E.strict60.values.astype(bool)
    comp_col = f"{a.species}_component"
    comp = E[comp_col].values if comp_col in E else Z["component"]
    return Z, cls, y, ids, E, s80, s60, comp, E.maxid_to_human_train.values

def human_baselines(a, cls, E, ids, C):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize
    import scipy.sparse as sp
    raw = pd.concat([pd.read_csv(f"{a.kaggle_dir}/train.csv"), pd.read_csv(f"{a.kaggle_dir}/test.csv")]).drop_duplicates("NCBIGeneID")
    raw["seq"] = raw.NucleotideSequence.map(clean)
    spl = pd.read_csv(a.split); df = raw.merge(spl[["NCBIGeneID", "split"]], on="NCBIGeneID").set_index("NCBIGeneID")
    tr, va = df[df.split == "train"], df[df.split == "val"]
    ytr = np.array([cls.index(c) for c in tr.GeneType]); yv = np.array([cls.index(c) for c in va.GeneType])
    have_fold = a.mfe_human and a.mfe_ext and os.path.exists(a.mfe_human) and os.path.exists(a.mfe_ext)
    mh = pickle.load(open(a.mfe_human, "rb")) if have_fold else None
    me = pickle.load(open(a.mfe_ext, "rb")) if have_fold else None
    def feats(seqs, keys, mf):
        L = pd.Series(seqs).str.len().values; w = pd.Series(seqs).str[:200]
        F = [np.log1p(np.minimum(L, 517))]
        if mf is not None:
            m = np.array([mf[k] for k in keys])
            F += [w.map(lambda q: (q.count("G") + q.count("C")) / len(q)).values, m[:, 0] / m[:, 1], m[:, 2]]
        return np.c_[tuple(F)]
    Ftr, Fv, Fe = feats(tr.seq.values, tr.index, mh), feats(va.seq.values, va.index, mh), feats(E.seq.values, ids, me)
    G = lambda X, Xt: HistGradientBoostingClassifier(random_state=0).fit(X, ytr).predict(Xt)
    C["Length only"] = [G(Ftr[:, :1], Fe[:, :1])]
    if have_fold: C["Length+GC+structure"] = [G(Ftr, Fe)]
    else: print("  (no ViennaRNA feature files: folding baseline and combiners skipped; see analysis/fold_external.py)")
    idx = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=6))}
    def K(seqs):
        r, c = [], []
        for i, q in enumerate(seqs):
            q = q[:517]
            for j in range(len(q) - 5): r.append(i); c.append(idx[q[j:j + 6]])
        return normalize(sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(seqs), 4096)))
    nn = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute").fit(K(tr.seq.values))
    C["1-NN lookup"] = [ytr[nn.kneighbors(K(E.seq.values))[1][:, 0]]]
    return (Fv, Fe, yv) if have_fold else None

def combiner(val_probs, ext_probs, Fv, Fe, yv):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(np.c_[np.log(val_probs + 1e-6), Fv], yv)
    return clf.classes_[clf.predict_proba(np.c_[np.log(ext_probs + 1e-6), Fe]).argmax(1)]

def main():
    from sklearn.metrics import f1_score, recall_score
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", required=True); ap.add_argument("--ext", required=True); ap.add_argument("--pred", required=True)
    ap.add_argument("--strict_file"); ap.add_argument("--kaggle_dir"); ap.add_argument("--split", default="data/split_homology80.csv")
    ap.add_argument("--runs_dir"); ap.add_argument("--best_lr", default="results/tables/Cell03_best_lr.json")
    ap.add_argument("--dnabert_dir"); ap.add_argument("--mfe_human"); ap.add_argument("--mfe_ext")
    ap.add_argument("--n_boot", type=int, default=2000); ap.add_argument("--out", default="results/tables")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sp = a.species
    Z, cls, y, ids, E, s80, s60, comp, mx = load_inputs(a)
    C = {}
    fold = human_baselines(a, cls, E, ids, C) if a.kaggle_dir else print("  (no --kaggle_dir: human-data baselines skipped)")
    for nm, lab in NEURAL:
        if f"prob__{nm}_s1" in Z: C[lab] = [Z[f"prob__{nm}_s{s}"].argmax(1) for s in S5]
    if fold and a.runs_dir:
        best = json.load(open(a.best_lr)); Fv, Fe, yv = fold
        vp = lambda s: np.load(f"{a.runs_dir}/MTL_Transformer_PE_lr{best['MTL_Transformer_PE']:g}_s{s}/preds.npz")["val_gene_type"]
        C["Transformer+PE + biophysical"] = [combiner(vp(s), Z[f"prob__MTL_Transformer_PE_s{s}"], Fv, Fe, yv) for s in S5]
    if a.dnabert_dir:
        fm = {s: np.load(f"{a.dnabert_dir}/DNABERT6_GeneType_s{s}/preds.npz", allow_pickle=True) for s in S3}
        key = f"{sp}_gene_type"
        assert all((fm[s][f"{sp}_gene_ids"] == ids).all() for s in fm), "DNABERT gene order differs from the TF predictions"
        C["DNABERT"] = [fm[s][key].argmax(1) for s in fm]
        if fold: C["DNABERT + biophysical"] = [combiner(fm[s]["val_gene_type"], fm[s][key], *fold) for s in fm]
    order = ["DNABERT", "Transformer+PE", "Transformer (order-blind)", "Single-task Transformer", "MTL BiLSTM", "MTL CNN",
             "1-NN lookup", "Length+GC+structure", "Length only", "Transformer+PE + biophysical", "DNABERT + biophysical"]
    C = {k: C[k] for k in order if k in C}
    corr = {k: np.mean([p == y for p in v], 0) for k, v in C.items()}
    ALLm, nohom = np.ones(len(y), bool), mx <= 1e-4
    def mF1(v, m): return np.mean([f1_score(y[m], p[m], labels=np.unique(y[m]), average="macro", zero_division=0) for p in v])
    def bacc(v, m): return np.mean([recall_score(y[m], p[m], labels=np.unique(y[m]), average="macro", zero_division=0) for p in v])
    rows = [[k, corr[k].mean(), mF1(v, ALLm), bacc(v, ALLm), corr[k][s80].mean(), corr[k][s60].mean(), corr[k][nohom].mean()]
            for k, v in C.items()]
    T = pd.DataFrame(rows, columns=["model", "acc all", "macroF1 all", "balanced acc all", "acc no>=80% human match",
                                    "acc no>=60%", "acc no detectable homolog"]).round(3)
    T.insert(0, "species", sp)
    print(f"[{sp}] n all {len(y)}, strict80 {s80.sum()}, strict60 {s60.sum()}, no homolog {nohom.sum()}, components {len(np.unique(comp))}")
    print(T.to_string(index=False)); T.to_csv(f"{a.out}/{sp}_accuracy.csv", index=False)
    # ---- cluster bootstrap over within-species homology components ------------
    uc = np.unique(comp); mem = [np.where(comp == c)[0] for c in uc]; rng = np.random.default_rng(0)
    B = [np.concatenate([mem[j] for j in rng.integers(0, len(uc), len(uc))]) for _ in range(a.n_boot)]
    out = []
    for x, z in CONTRASTS:
        if x not in corr or z not in corr: continue
        d = corr[x] - corr[z]
        for sub, m in [("all", ALLm), ("no>=60%", s60)]:
            bs = np.array([d[ix[m[ix]]].mean() for ix in B])
            lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
            out.append([sp, f"{x} − {z}", sub, round(100 * d[m].mean(), 1), round(100 * lo, 1), round(100 * hi, 1), round(p, 3)])
    R = pd.DataFrame(out, columns=["species", "contrast", "subset", "diff_pp", "lo", "hi", "p"])
    print(R.to_string(index=False)); R.to_csv(f"{a.out}/{sp}_contrasts.csv", index=False)
    labs = np.unique(y)
    PC = pd.DataFrame({"class": np.array(cls)[labs], "n": [int((y == l).sum()) for l in labs]})
    for k, v in C.items():
        PC[k] = np.mean([f1_score(y, p, labels=labs, average=None, zero_division=0) for p in v], 0).round(3)
    print(PC.to_string(index=False)); PC.to_csv(f"{a.out}/{sp}_perclass.csv", index=False)
    pickle.dump({"corr": corr, "C": C, "y": y, "s60": s60, "comp": comp}, open(f"{a.out}/{sp}_corr.pkl", "wb"))

if __name__ == "__main__":
    main()
