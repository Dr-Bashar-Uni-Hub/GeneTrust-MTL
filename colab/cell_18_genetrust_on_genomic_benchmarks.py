# ======================================================================
# CELL 18 — THE GENETRUST PROTOCOL ON AN INDEPENDENT BENCHMARK  (standalone)
# Run ALONE in its own notebook (GPU runtime recommended for the CNN; ~1–2 h).
# Applies the same evaluation controls to Genomic Benchmarks (Grešová et al., 2023),
# which is constructed independently of NCBI Gene:
#   1. leakage audit of the provided train/test split (exact and ≥80%-identity matches, both strands)
#   2. homology-aware re-split (connected components at ≥80% identity covering ≥80% of either sequence)
#   3. baselines: constant, length only, GC only, length+GC, 1-NN on canonical 6-mers
#   4. models: 6-mer logistic regression (order-blind) and a CNN on one-hot sequence (order-aware, 3 seeds)
#   5. homology-component bootstrap CIs for model − baseline contrasts
#   6. external species (enhancers only): human-trained models applied to mouse enhancers
# Tasks: human_enhancers_ensembl (variable length: a length shortcut is possible) and
#        human_nontata_promoters (all 251 nt: a negative control in which length cannot help).
# Outputs: Drive MyDrive/JBSD_Revision/Cell18_*.csv/json/npz and a zip to send back.
# ======================================================================
TASKS = ["human_enhancers_ensembl", "human_nontata_promoters"]
EXTERNAL = {"human_enhancers_ensembl": "dummy_mouse_enhancers_ensembl"}
SEEDS, MAXLEN_CAP, SUBSAMPLE = [1, 21, 42], 600, None       # SUBSAMPLE=int to cap sequences per task (quick test)
import os, glob, json, shutil, subprocess, itertools, time
import numpy as np, pandas as pd

try:
    from google.colab import drive; drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
OUT = "/content/drive/MyDrive/JBSD_Revision" if IN_COLAB else os.environ.get("C18_OUT", "/tmp/c18out")
W = "/content/gb" if IN_COLAB else os.environ.get("C18_WORK", "/tmp/gb")
os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True)

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0: print(r.stdout[-1500:], r.stderr[-1500:]); raise RuntimeError(cmd)
    return r.stdout
if IN_COLAB:
    sh("pip -q install genomic-benchmarks")
    if shutil.which("mmseqs") is None:
        sh("wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mm.tgz && tar xzf /tmp/mm.tgz -C /tmp && cp /tmp/mmseqs/bin/mmseqs /usr/local/bin/")
print(sh("mmseqs version").strip())

# ---------------------------------------------------------------- data
COMP = str.maketrans("ACGT", "TGCA")
def clean(s): return "".join(c for c in s.upper() if c in "ACGTN")
def load_task(name):
    d = f"{W}/{name}"
    if not os.path.isdir(d):
        from genomic_benchmarks.loc2seq import download_dataset
        download_dataset(name, version=0, dest_path=W)
    rows = []
    for split in ["train", "test"]:
        for cdir in sorted(glob.glob(f"{d}/{split}/*")):
            c = os.path.basename(cdir)
            for f in glob.glob(f"{cdir}/*.txt"):
                rows.append((f"{split}:{c}:{os.path.basename(f)[:-4]}", split, c, clean(open(f).read().strip())))
    D = pd.DataFrame(rows, columns=["id", "split", "label", "seq"]).sort_values("id").reset_index(drop=True)
    D = D[D.seq.str.len() >= 6].reset_index(drop=True)
    if SUBSAMPLE: D = D.groupby("split", group_keys=False).apply(lambda g: g.sample(min(len(g), SUBSAMPLE), random_state=0)).reset_index(drop=True)
    D["uid"] = [f"q{i}" for i in range(len(D))]
    return D

# ---------------------------------------------------------------- homology graph (both strands searched by MMseqs2)
def write_fa(D, path):
    with open(path, "w") as f:
        for u, s in zip(D.uid, D.seq): f.write(f">{u}\n{s}\n")
M8 = "query,target,fident,qstart,qend,tstart,tend,qlen,tlen"
MM_EXTRA = os.environ.get("MMSEQS_EXTRA", "")       # optional memory flags for small machines; unused in Colab
def search(qfa, tfa, tag, min_id=0.8):
    m8 = f"{W}/{tag}.m8"
    sh(f"mmseqs easy-search {qfa} {tfa} {m8} {W}/tmp_{tag} --search-type 3 --min-seq-id {min_id} -c 0 "
       f"--max-seqs 300 --threads 8 -v 1 --format-output {M8} {MM_EXTRA}")
    if os.path.getsize(m8) == 0: return pd.DataFrame(columns=M8.split(","))
    h = pd.read_csv(m8, sep="\t", header=None, names=M8.split(","))
    qc = ((h.qend - h.qstart).abs() + 1) / h.qlen; tc = ((h.tend - h.tstart).abs() + 1) / h.tlen
    return h[(h["query"] != h["target"]) & (h.fident >= min_id) & ((qc >= 0.8) | (tc >= 0.8))]

def components(D, h):
    import scipy.sparse as sps
    from scipy.sparse.csgraph import connected_components
    ix = {u: i for i, u in enumerate(D.uid)}
    G = sps.coo_matrix((np.ones(len(h)), (h["query"].map(ix).values, h["target"].map(ix).values)), shape=(len(D),) * 2)
    return connected_components(G, directed=False)[1]

def homology_split(D, comp, test_frac, seed=2026):
    rng = np.random.default_rng(seed); cs = rng.permutation(np.unique(comp)); n = len(D); tgt = test_frac * n
    size = pd.Series(comp).value_counts(); test_c, k = set(), 0
    for c in cs:
        if k >= tgt: break
        if size[c] > 0.02 * n: continue                       # very large components stay in training (as in Cell 2)
        test_c.add(c); k += size[c]
    return np.where(np.isin(comp, list(test_c)), "test", "train")

# ---------------------------------------------------------------- features, baselines, models
canon = {}
for p in itertools.product("ACGT", repeat=6):
    k = "".join(p); c = min(k, k.translate(COMP)[::-1]); canon[k] = c
cid = {c: i for i, c in enumerate(sorted(set(canon.values())))}
def kmer_matrix(seqs):
    import scipy.sparse as sps
    r, c = [], []
    for i, s in enumerate(seqs):
        for j in range(len(s) - 5):
            k = canon.get(s[j:j + 6])
            if k is not None: r.append(i); c.append(cid[k])
    X = sps.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(seqs), len(cid)))
    n = np.sqrt(X.multiply(X).sum(1)).A1; n[n == 0] = 1
    return sps.diags(1 / n) @ X
def simple_feats(seqs):
    L = np.array([len(s) for s in seqs], float); gc = np.array([(s.count("G") + s.count("C")) / max(1, len(s)) for s in seqs])
    return np.c_[np.log1p(L)], np.c_[gc], np.c_[np.log1p(L), gc]
def nn1(Ktr, ytr, Kte, chunk=2000):
    Ktr_t = Ktr.T.tocsc().astype(np.float32); out = np.empty(Kte.shape[0], dtype=ytr.dtype)
    for a in range(0, Kte.shape[0], chunk):
        S = (Kte[a:a + chunk] @ Ktr_t).toarray(); out[a:a + chunk] = ytr[S.argmax(1)]
    return out
def baselines(tr, te, ytr):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    Ltr, Gtr, LGtr = simple_feats(tr.seq); Lte, Gte, LGte = simple_feats(te.seq)
    P = {"Constant (majority)": np.full(len(te), np.bincount(ytr).argmax())}
    for nm, a, b in [("Length only", Ltr, Lte), ("GC only", Gtr, Gte), ("Length+GC", LGtr, LGte)]:
        P[nm] = HistGradientBoostingClassifier(random_state=0).fit(a, ytr).predict(b)
    Ktr, Kte = kmer_matrix(tr.seq.tolist()), kmer_matrix(te.seq.tolist())
    P["1-NN 6-mer lookup"] = nn1(Ktr, ytr, Kte)
    P["6-mer logistic regression"] = LogisticRegression(C=1.0, max_iter=2000).fit(Ktr, ytr).predict(Kte)
    return P
def onehot(seqs, L):
    X = np.zeros((len(seqs), L, 4), np.float32); m = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s[:L]):
            k = m.get(ch)
            if k is not None: X[i, j, k] = 1
    return X
def cnn_preds(tr, te, ytr, L, seed, extra=None):
    import tensorflow as tf
    from tensorflow.keras import layers as KL
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    nc = int(ytr.max()) + 1
    m = tf.keras.Sequential([KL.Input((L, 4)), KL.Conv1D(64, 9, activation="relu", padding="same"), KL.MaxPooling1D(3),
                             KL.Conv1D(64, 9, activation="relu", padding="same"), KL.GlobalMaxPooling1D(),
                             KL.Dropout(0.3), KL.Dense(nc, activation="softmax")])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    Xtr = onehot(tr.seq.tolist(), L); rng = np.random.default_rng(seed); v = rng.random(len(Xtr)) < 0.1
    es = tf.keras.callbacks.EarlyStopping("val_loss", patience=3, restore_best_weights=True)
    m.fit(Xtr[~v], ytr[~v], validation_data=(Xtr[v], ytr[v]), epochs=30, batch_size=256, callbacks=[es], verbose=0)
    out = [m.predict(onehot(te.seq.tolist(), L), batch_size=512, verbose=0).argmax(1)]
    if extra is not None: out.append(m.predict(onehot(extra.seq.tolist(), L), batch_size=512, verbose=0).argmax(1))
    return out

def bootstrap(corr_a, corr_b, comp, n=2000, seed=0):
    d = corr_a - corr_b; uc = np.unique(comp); mem = [np.where(comp == c)[0] for c in uc]; rng = np.random.default_rng(seed)
    bs = np.array([d[np.concatenate([mem[j] for j in rng.integers(0, len(uc), len(uc))])].mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); return 100 * d.mean(), 100 * lo, 100 * hi, 2 * min((bs <= 0).mean(), (bs >= 0).mean())

def bacc(y, p): return np.mean([(p[y == c] == c).mean() for c in np.unique(y)])

# ---------------------------------------------------------------- run
summary, contrasts, audit = [], [], {}
for task in TASKS:
    t0 = time.time(); D = load_task(task); classes = sorted(D.label.unique()); D["y"] = D.label.map({c: i for i, c in enumerate(classes)})
    L = int(min(MAXLEN_CAP, D.seq.str.len().max()))
    print(f"\n===== {task}: {len(D):,} sequences {D.groupby(['split', 'label']).size().to_dict()} | length {D.seq.str.len().min()}–{D.seq.str.len().max()}", flush=True)
    # 1. leakage audit of the provided split
    tr0, te0 = D[D.split == "train"], D[D.split == "test"]
    rc = {s.translate(COMP)[::-1] for s in tr0.seq}; ex = set(tr0.seq)
    exact = te0.seq.isin(ex) | te0.seq.isin(rc)
    write_fa(D, f"{W}/{task}.fa"); write_fa(tr0, f"{W}/{task}_tr.fa"); write_fa(te0, f"{W}/{task}_te.fa")
    hq = search(f"{W}/{task}_te.fa", f"{W}/{task}_tr.fa", f"{task}_te_vs_tr")
    near = te0.uid.isin(set(hq["query"]))
    audit[task] = {"n_train": len(tr0), "n_test": len(te0), "test_exact_duplicate_in_train_either_strand": int(exact.sum()),
                   "test_with_ge80pct_identity_match_in_train": int(near.sum()), "frac_near": float(near.mean())}
    print("  leakage audit:", audit[task], flush=True)
    # 2. homology-aware re-split
    hall = search(f"{W}/{task}.fa", f"{W}/{task}.fa", f"{task}_all")
    D["comp"] = components(D, hall); D["hsplit"] = homology_split(D, D.comp.values, len(te0) / len(D))
    assert not set(D.comp[D.hsplit == "train"]) & set(D.comp[D.hsplit == "test"])
    audit[task].update(n_components=int(D.comp.nunique()), largest_component=int(D.comp.value_counts().iloc[0]),
                       homology_test_n=int((D.hsplit == "test").sum()))
    ext = None
    if task in EXTERNAL:
        ext = load_task(EXTERNAL[task]); ext["y"] = ext.label.map({c: i for i, c in enumerate(classes)})
    for split_name, col in [("provided", "split"), ("homology-aware", "hsplit")]:
        tr, te = D[D[col] == "train"].reset_index(drop=True), D[D[col] == "test"].reset_index(drop=True)
        ytr, yte, comp = tr.y.values, te.y.values, te.comp.values
        P = baselines(tr, te, ytr)
        Pext = baselines(tr, ext, ytr) if (ext is not None and split_name == "homology-aware") else None
        cnn = [cnn_preds(tr, te, ytr, L, s, ext if Pext is not None else None) for s in SEEDS]
        P["CNN (one-hot, 3 seeds)"] = [c[0] for c in cnn]
        if Pext is not None: Pext["CNN (one-hot, 3 seeds)"] = [c[1] for c in cnn]
        for setname, PP, Y, cmp_ in [(split_name, P, yte, comp)] + ([("external: " + EXTERNAL[task], Pext, ext.y.values, None)] if Pext else []):
            for nm, p in PP.items():
                ps = p if isinstance(p, list) else [p]
                acc = [(q == Y).mean() for q in ps]; ba = [bacc(Y, q) for q in ps]
                summary.append([task, setname, nm, np.mean(acc), np.std(acc, ddof=1) if len(ps) > 1 else np.nan, np.mean(ba)])
            if cmp_ is not None:
                corr = {nm: np.mean([(q == Y) for q in (p if isinstance(p, list) else [p])], 0) for nm, p in PP.items()}
                for a, b in [("CNN (one-hot, 3 seeds)", "Length only"), ("CNN (one-hot, 3 seeds)", "1-NN 6-mer lookup"),
                             ("6-mer logistic regression", "Length only"), ("CNN (one-hot, 3 seeds)", "6-mer logistic regression")]:
                    contrasts.append([task, setname, f"{a} − {b}", *bootstrap(corr[a], corr[b], cmp_)])
        print(f"  [{split_name}] done ({(time.time() - t0) / 60:.0f} min)", flush=True)
    D[["id", "split", "hsplit", "label", "comp"]].to_csv(f"{OUT}/Cell18_{task}_partition.csv", index=False)

S_ = pd.DataFrame(summary, columns=["task", "evaluation set", "model", "accuracy", "acc_sd_over_seeds", "balanced_accuracy"])
C_ = pd.DataFrame(contrasts, columns=["task", "split", "contrast", "diff_pp", "lo", "hi", "p"]).round(4)
S_.round(4).to_csv(f"{OUT}/Cell18_summary.csv", index=False); C_.to_csv(f"{OUT}/Cell18_contrasts.csv", index=False)
json.dump(audit, open(f"{OUT}/Cell18_leakage_audit.json", "w"), indent=1)
pd.set_option("display.width", 220); print("\n" + S_.round(3).to_string(index=False)); print("\n" + C_.to_string(index=False))
pkg = "/content/Cell18_results" if IN_COLAB else f"{OUT}_pkg"; shutil.rmtree(pkg, ignore_errors=True); os.makedirs(pkg)
for f in glob.glob(f"{OUT}/Cell18_*"): shutil.copy(f, pkg)
zf = shutil.make_archive(pkg, "zip", pkg); print("\nSend me this file:", zf)
try:
    from google.colab import files; files.download(zf)
except Exception:
    pass
