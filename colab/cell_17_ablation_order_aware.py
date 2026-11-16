# ======================================================================
# CELL 17 — ABLATION OF THE ORDER-AWARE MODIFICATION  (standalone)
# Run ALONE in its own notebook (GPU runtime: A100/L4; ~1.5–2.5 h, resumable).
# The order-aware Transformer differs from the original in two switched
# components: a learned positional embedding (PE) and padding masks (key-padding
# masks in attention + masked mean pooling). This cell trains the two missing
# arms of the 2×2 design with the IDENTICAL Cell 3 harness (same data, vocabulary,
# architecture, LR grid selected on validation loss with seed 42, five seeds):
#     neither  = MTL_Transformer        (existing runs, reused)
#     PE only  = MTL_Transformer_PEonly (new)
#     mask only= MTL_Transformer_MASKonly (new)
#     both     = MTL_Transformer_PE     (existing runs, reused)
# Outputs (Drive, MyDrive/JBSD_Revision/): Cell17_ablation_*.csv, Cell17_ablation_predictions.npz,
# and a zip to send back. Needs split_homology80.csv and Cell03_best_lr.json in that folder.
# ======================================================================
import os, glob, json, shutil, zipfile
import numpy as np, pandas as pd
try:
    from google.colab import drive; drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
OUT = "/content/drive/MyDrive/JBSD_Revision" if IN_COLAB else os.environ.get("C17_OUT", "/tmp/rev/c17out")
# ---- locate the human Kaggle data (train.csv/test.csv) without needing Cell 1
SHA = {"train.csv": "158040253e4975790882c59e7c1fd1756a3bab32dc6e2226aba13e54c39cdb8e",
       "test.csv": "ea5cd67d1ea97a36e67ae50b975d14482f20fa5a441af7c3aa279f668ff45d94"}
CKPT_NAME = "Genomic_Transformer_Checkpoint"

def _data_dir_ok(d):
    return all(os.path.exists(f"{d}/{f}") for f in ("train.csv", "test.csv"))

def _search_dirs(roots):
    for r in roots:
        for f in glob.glob(f"{r}/**/train.csv", recursive=True):
            d = os.path.dirname(f)
            if _data_dir_ok(d):
                try:
                    if "NucleotideSequence" in pd.read_csv(f, nrows=1).columns: return d
                except Exception:
                    pass
    return None

def _unpack_checkpoint(work):
    """Find Genomic_Transformer_Checkpoint*.zip (loose, or nested in e.g. Results.zip) and unpack it, as Cell 1 does."""
    roots = ["/content", "/content/drive/MyDrive"] if IN_COLAB else ["/tmp"]
    loose = [z for r in roots for z in glob.glob(f"{r}/**/{CKPT_NAME}*.zip", recursive=True)]
    src = max(loose, key=os.path.getmtime) if loose else None
    if src is None:
        outer = [z for r in roots for z in glob.glob(f"{r}/**/*.zip", recursive=True) if CKPT_NAME not in os.path.basename(z)]
        for z in sorted(outer, key=os.path.getmtime, reverse=True):
            try:
                with zipfile.ZipFile(z) as zf:
                    if any(os.path.basename(n).startswith(CKPT_NAME) and n.endswith(".zip") for n in zf.namelist()) or \
                       any(n.endswith("data/train.csv") for n in zf.namelist()):
                        src = z; break
            except (zipfile.BadZipFile, OSError):
                continue
    if src is None: return None
    print("Unpacking", src)
    os.makedirs(work, exist_ok=True)
    with zipfile.ZipFile(src) as z: z.extractall(work)
    done = set()
    while (d := _search_dirs([work])) is None:
        inner = [z for z in glob.glob(f"{work}/**/{CKPT_NAME}*.zip", recursive=True) if z not in done]
        if not inner: return None
        with zipfile.ZipFile(inner[0]) as z: z.extractall(os.path.dirname(inner[0]))
        done.add(inner[0])
    return d

def find_human_data():
    import hashlib
    d = (_search_dirs(["/content/rev", "/content/rev13", "/tmp/rev", "/tmp/rev13"])          # unpacked earlier (e.g. by Cell 1)
         or _unpack_checkpoint("/content/rev13" if IN_COLAB else "/tmp/rev13")  # checkpoint zip in Drive or /content
         or _search_dirs(["/content/drive/MyDrive"] if IN_COLAB else []))       # loose Kaggle CSVs in Drive
    if d is None and IN_COLAB:
        from google.colab import files
        print("Human data not found. Upload train.csv and test.csv (Kaggle), or the checkpoint zip / Results.zip:")
        for b in files.upload():
            shutil.move(b, f"/content/{b}") if os.path.abspath(b) != f"/content/{b}" else None
        d = _search_dirs(["/content"]) or _unpack_checkpoint("/content/rev13")
    if d is None:
        raise FileNotFoundError("train.csv/test.csv not found. Run Cell 1 first, or put the Kaggle CSVs or the "
                                "checkpoint zip in Google Drive and re-run this cell.")
    for f, h in SHA.items():
        got = hashlib.sha256(open(f"{d}/{f}", "rb").read()).hexdigest()
        print(f"  {f}: SHA-256 {'OK' if got == h else 'MISMATCH (' + got[:16] + '…) — check that this is the original Kaggle file'}")
    return d


DATA = find_human_data()
print("Human data:", DATA)

# ===== Cell 3 harness (verbatim: data, vocabulary, architectures, run()) =====
import os, json, time, glob, itertools
from collections import Counter
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras import layers as L
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score

if "OUT" not in globals():                           # session was restarted
    from google.colab import drive; drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/JBSD_Revision"
    DATA = [os.path.dirname(f) for f in glob.glob("/content/rev/**/data/train.csv", recursive=True)][0]
RUNS = f"{OUT}/runs"; os.makedirs(RUNS, exist_ok=True)
SMALL = os.environ.get("SMALL_TEST") == "1"           # local debugging only
LR_GRID = [1e-4, 3e-4, 1e-3]
MAX_EPOCHS, PATIENCE, BATCH = (2, 1, 64) if SMALL else (150, 10, 128)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
import subprocess, warnings
warnings.filterwarnings("ignore", message=".*does not support masking.*")
GPU = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True,
                     capture_output=True, text=True).stdout.strip() or "none"
print("GPU:", GPU, "| TF sees:", len(tf.config.list_physical_devices("GPU")), "GPU(s)")

# ---- 1. Data on the new split ---------------------------------------------
V = set("ACGT")
def roles(d):
    d = str(d).lower(); r = []
    if "pseudo" in d: r.append("PSEUDO")
    if "regulatory" in d: r.append("REGULATORY")
    if "protein" in d: r.append("PROTEIN_CODING")
    return r or ["OTHER"]
def fam(s):
    s = str(s)
    if s.startswith("IGH"): return "IMMUNOGLOBULIN"
    if s.startswith(("RNU", "SNORD")): return "RNA_GENE"
    if s.startswith("LOC"): return "LOCUS"
    return "OTHER"
raw = pd.concat([pd.read_csv(f"{DATA}/train.csv"), pd.read_csv(f"{DATA}/test.csv")]).drop_duplicates("NCBIGeneID")
spl = pd.read_csv(f"{OUT}/split_homology80.csv")
P = raw.merge(spl[["NCBIGeneID", "split", "cluster", "test_maxid_to_train"]], on="NCBIGeneID")
P["seq"] = P.NucleotideSequence.map(lambda s: "".join(c for c in str(s).upper() if c in V))
P["fam"] = P.Symbol.map(fam); P["roles"] = P.Description.map(roles)
S = {k: P[P.split == k].reset_index(drop=True) for k in ["train", "val", "test"]}
if SMALL:
    S = {k: v.sample(min(len(v), 600), random_state=0).reset_index(drop=True) for k, v in S.items()}
assert set(S["train"].cluster) & set(S["test"].cluster) == set(), "train/test share a component"
assert set(S["train"].NCBIGeneID) & set(S["test"].NCBIGeneID) == set()
assert not set(S["train"].seq) & set(S["test"].seq), "exact duplicate across train/test"

cnt = Counter()
for s in S["train"].seq: cnt.update(s[i:i+6] for i in range(len(s) - 5))
all6 = ["".join(p) for p in itertools.product("ACGT", repeat=6)]
order = [k for k, _ in cnt.most_common()] + sorted(set(all6) - set(cnt))
vocab = {k: i + 1 for i, k in enumerate(order)}        # always 4,096 tokens + pad
def enc(s):
    e = [vocab[s[i:i+6]] for i in range(len(s) - 5)][:512]
    return e + [0] * (512 - len(e))
X = {k: np.array([enc(s) for s in v.seq], dtype=np.int32) for k, v in S.items()}
le = LabelEncoder().fit(S["train"].GeneType)
fe = LabelEncoder().fit(S["train"].fam)
mlb = MultiLabelBinarizer().fit(S["train"].roles)
Y = {k: {"gene_type": le.transform(v.GeneType), "symbol_family": fe.transform(v.fam),
         "functional_roles": mlb.transform(v.roles).astype("float32")} for k, v in S.items()}
NG, NS, NR = len(le.classes_), len(fe.classes_), len(mlb.classes_)
GT_TEST_LABELS = np.unique(Y["test"]["gene_type"])     # macro-F1 over classes present in test
print({k: len(v) for k, v in S.items()}, "| classes:", NG, NS, NR, "| roles:", list(mlb.classes_))
print("Gene Type classes in test:", list(le.classes_[GT_TEST_LABELS]))

# ---- 2. Architectures -----------------------------------------------------
class TransformerEncoder(L.Layer):                      # identical to original
    def __init__(self, d=64, h=4, ff=128, dr=0.2, **kw):
        super().__init__(**kw)
        self.attention = L.MultiHeadAttention(num_heads=h, key_dim=d)
        self.ffn = tf.keras.Sequential([L.Dense(ff, activation="relu"), L.Dense(d)])
        self.layernorm1, self.layernorm2 = L.LayerNormalization(epsilon=1e-6), L.LayerNormalization(epsilon=1e-6)
        self.dropout1, self.dropout2 = L.Dropout(dr), L.Dropout(dr)
    def call(self, x, mask=None, training=False):
        am = None if mask is None else tf.logical_and(mask[:, :, None], mask[:, None, :])
        a = self.dropout1(self.attention(x, x, attention_mask=am), training=training)
        o = self.layernorm1(x + a)
        return self.layernorm2(o + self.dropout2(self.ffn(o), training=training))

class PadMask(L.Layer):
    def call(self, t): return tf.not_equal(t, 0)
class PositionEmbedding(L.Layer):
    def build(self, s): self.pos = self.add_weight(name="pos", shape=(s[1], s[2]), initializer="uniform")
    def call(self, x): return x + self.pos[None]
class MaskedMeanPool(L.Layer):
    def call(self, x, mask):
        m = tf.cast(mask, x.dtype)[..., None]
        return tf.reduce_sum(x * m, 1) / tf.maximum(tf.reduce_sum(m, 1), 1.0)

def heads(z, tasks):
    out = []
    if "gene_type" in tasks: out.append(L.Dense(NG, activation="softmax", name="gene_type")(z))
    if "symbol_family" in tasks: out.append(L.Dense(NS, activation="softmax", name="symbol_family")(z))
    if "functional_roles" in tasks: out.append(L.Dense(NR, activation="sigmoid", name="functional_roles")(z))
    return out
ALL = ("gene_type", "symbol_family", "functional_roles")

def transformer(tasks=ALL, pos_mask=False):
    i = L.Input((512,), dtype="int32"); x = L.Embedding(4097, 64, name="Embedding")(i)
    m = None
    if pos_mask:
        m = PadMask()(i); x = PositionEmbedding(name="PositionEmbedding")(x)
    for _ in range(2): x = TransformerEncoder()(x, mask=m)
    z = MaskedMeanPool()(x, m) if pos_mask else L.GlobalAveragePooling1D()(x)
    return tf.keras.Model(i, heads(L.Dropout(0.2)(z), tasks))
def cnn():
    i = L.Input((512,), dtype="int32"); x = L.Embedding(4097, 64)(i)
    x = L.Conv1D(128, 5, activation="relu", padding="same")(x)
    x = L.Conv1D(128, 3, activation="relu", padding="same")(x)
    return tf.keras.Model(i, heads(L.Dropout(0.2)(L.GlobalMaxPooling1D()(x)), ALL))
def bilstm():
    i = L.Input((512,), dtype="int32"); x = L.Embedding(4097, 64)(i)
    x = L.Bidirectional(L.LSTM(64, dropout=0.2))(x)
    return tf.keras.Model(i, heads(L.Dropout(0.2)(x), ALL))

MODELS = {
    "MTL_Transformer":    (lambda: transformer(), ALL, 429778),
    "MTL_Transformer_PE": (lambda: transformer(pos_mask=True), ALL, 429778 + 512 * 64),
    "STL_GeneType":       (lambda: transformer(("gene_type",)), ("gene_type",), 429258),
    "STL_SymbolFamily":   (lambda: transformer(("symbol_family",)), ("symbol_family",), 428868),
    "STL_FunctionalRoles":(lambda: transformer(("functional_roles",)), ("functional_roles",), 428868),
    "MTL_CNN":            (cnn, ALL, 354898),
    "MTL_BiLSTM":         (bilstm, ALL, 330578),
}
LOSS = {"gene_type": "sparse_categorical_crossentropy", "symbol_family": "sparse_categorical_crossentropy",
        "functional_roles": "binary_crossentropy"}

# ---- 3. One run = train to convergence, save everything -------------------
def metrics(tasks, preds, split="test"):
    y, r = Y[split], {}
    for t, p in zip(tasks, preds):
        if t == "gene_type":
            yp = p.argmax(1)
            r.update(GT_acc=accuracy_score(y[t], yp),
                     GT_mF1=f1_score(y[t], yp, labels=GT_TEST_LABELS, average="macro", zero_division=0))
        elif t == "symbol_family":
            yp = p.argmax(1)
            r.update(SF_acc=accuracy_score(y[t], yp), SF_mF1=f1_score(y[t], yp, average="macro", zero_division=0))
        else:
            yp = (p >= 0.5).astype(int)
            r.update(FR_exact=(y[t] == yp).all(1).mean(),
                     FR_mF1=f1_score(y[t], yp, average="macro", zero_division=0),
                     FR_microF1=f1_score(y[t], yp, average="micro", zero_division=0))
    return r

class Progress(tf.keras.callbacks.Callback):
    def __init__(self, tag): super().__init__(); self.tag = tag
    def on_train_begin(self, logs=None): self.t0 = time.time()
    def on_epoch_end(self, ep, logs=None):
        if (ep + 1) % 10 == 0:
            print(f"      {self.tag}: epoch {ep+1}, val_loss {logs['val_loss']:.4f}, "
                  f"{(time.time()-self.t0)/(ep+1):.1f} s/epoch", flush=True)

def run(name, lr, seed):
    tag = f"{name}_lr{lr:g}_s{seed}"; d = f"{RUNS}/{tag}"
    if os.path.exists(f"{d}/metrics.json"):
        return json.load(open(f"{d}/metrics.json"))
    os.makedirs(d, exist_ok=True)
    build, tasks, _ = MODELS[name]
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    m = build()
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss={t: LOSS[t] for t in tasks})
    ytr = {t: Y["train"][t] for t in tasks}; yva = {t: Y["val"][t] for t in tasks}
    if len(tasks) == 1: ytr, yva = ytr[tasks[0]], yva[tasks[0]]
    es = tf.keras.callbacks.EarlyStopping("val_loss", patience=PATIENCE, restore_best_weights=True)
    t0 = time.time()
    h = m.fit(X["train"], ytr, validation_data=(X["val"], yva), epochs=MAX_EPOCHS,
              batch_size=BATCH, callbacks=[es, Progress(tag)], verbose=0)
    train_s = time.time() - t0
    t0 = time.time(); pt = m.predict(X["test"], batch_size=128, verbose=0); infer_s = time.time() - t0
    pv = m.predict(X["val"], batch_size=128, verbose=0)
    pt, pv = (pt if isinstance(pt, list) else [pt]), (pv if isinstance(pv, list) else [pv])
    vl = np.array(h.history["val_loss"])
    res = {"model": name, "lr": lr, "seed": seed, "gpu": GPU, "params": int(m.count_params()),
           "best_epoch": int(vl.argmin() + 1), "epochs_run": len(vl), "hit_cap": bool(vl.argmin() + 1 == MAX_EPOCHS),
           "best_val_loss": float(vl.min()), "train_s": train_s, "infer_ms_per_seq": 1000 * infer_s / len(X["test"]),
           **metrics(tasks, pt)}
    np.savez_compressed(f"{d}/preds.npz", **{f"test_{t}": p for t, p in zip(tasks, pt)},
                        **{f"val_{t}": p for t, p in zip(tasks, pv)})
    pd.DataFrame(h.history).to_csv(f"{d}/history.csv", index=False)
    m.save_weights(f"{d}/model.weights.h5")
    json.dump(res, open(f"{d}/metrics.json", "w"), indent=1, default=float)
    return res


# ===== Ablation arms ======================================================
def transformer_ab(pe, mask, tasks=ALL):
    """Original Transformer with the two components of the order-aware modification switched independently."""
    i = L.Input((512,), dtype="int32"); x = L.Embedding(4097, 64, name="Embedding")(i)
    m = PadMask()(i) if mask else None
    if pe: x = PositionEmbedding(name="PositionEmbedding")(x)
    for _ in range(2): x = TransformerEncoder()(x, mask=m)
    z = MaskedMeanPool()(x, m) if mask else L.GlobalAveragePooling1D()(x)
    return tf.keras.Model(i, heads(L.Dropout(0.2)(z), tasks))
MODELS["MTL_Transformer_PEonly"] = (lambda: transformer_ab(True, False), ALL, 429778 + 512 * 64)
MODELS["MTL_Transformer_MASKonly"] = (lambda: transformer_ab(False, True), ALL, 429778)
ARMS = [("MTL_Transformer", "neither"), ("MTL_Transformer_PEonly", "PE only"),
        ("MTL_Transformer_MASKonly", "mask only"), ("MTL_Transformer_PE", "both (order-aware)")]
# sanity: the factorized builder reproduces both existing architectures exactly
for (pe, mk), ref in [((False, False), "MTL_Transformer"), ((True, True), "MTL_Transformer_PE")]:
    a, b = transformer_ab(pe, mk), MODELS[ref][0]()
    assert a.count_params() == b.count_params() and [l.__class__.__name__ for l in a.layers] == [l.__class__.__name__ for l in b.layers], ref
for n, _ in ARMS:
    k = MODELS[n][0]().count_params(); assert SMALL or k == MODELS[n][2], (n, k)
print("Architecture checks passed:", {n: MODELS[n][2] for n, _ in ARMS})

# ---- learning rate for the new arms (Cell 3 protocol: grid on validation loss, seed 42) ----
BEST_LR = json.load(open(f"{OUT}/Cell03_best_lr.json"))
for n in ["MTL_Transformer_PEonly", "MTL_Transformer_MASKonly"]:
    if n not in BEST_LR:
        rs = [run(n, lr, 42) for lr in LR_GRID]
        BEST_LR[n] = float(min(rs, key=lambda r: r["best_val_loss"])["lr"])
        print(f"  {n}: LR grid", {r['lr']: round(r['best_val_loss'], 4) for r in rs}, "-> selected", BEST_LR[n])
json.dump(BEST_LR, open(f"{OUT}/Cell17_best_lr_with_ablation.json", "w"), indent=1)

# ---- five seeds per arm (existing runs are reused instantly) ----
SEEDS = [1, 7, 21, 42, 100]; rows = []
for n, lab in ARMS:
    for s in SEEDS:
        r = run(n, BEST_LR[n], s); r["arm"] = lab; rows.append(r)
        print(f"  {lab:20s} seed {s:3d}  GT_acc={r['GT_acc']:.4f}  best_epoch={r['best_epoch']}", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/Cell17_ablation_runs.csv", index=False)
cols = [c for c in ["GT_acc", "GT_mF1", "SF_acc", "SF_mF1", "FR_exact", "FR_mF1"] if c in R]
print("\nHUMAN HOMOLOGY-AWARE TEST SET (mean ± SD over 5 seeds)")
print(R.groupby("arm", sort=False)[cols].agg(lambda x: f"{x.mean():.4f} ± {x.std():.4f}").to_string())

def load(n, s):
    tf.keras.backend.clear_session(); m = MODELS[n][0]()
    m.load_weights(f"{RUNS}/{n}_lr{BEST_LR[n]:g}_s{s}/model.weights.h5"); return m
def gt(p): return p[0] if isinstance(p, list) else p

# ---- order sensitivity: permute the 6-mers of each sequence (non-padding positions) ----
rng = np.random.default_rng(0); idx = rng.choice(len(X["test"]), size=min(300, len(X["test"])), replace=False)
Xp = X["test"][idx].copy()
for r_ in Xp:
    k = int((r_ > 0).sum()); r_[:k] = rng.permutation(r_[:k])
sens = {}
for n, lab in ARMS:
    m = load(n, 1); a = gt(m.predict(X["test"][idx], verbose=0)); b = gt(m.predict(Xp, verbose=0))
    sens[lab] = float(np.abs(a - b).max())
print("\nOrder sensitivity (max change in any Gene Type probability when 6-mers are permuted):", {k: f"{v:.2e}" for k, v in sens.items()})

# ---- per-gene correctness (seed-averaged) and homology-cluster bootstrap on the human test set ----
yt = Y["test"]["gene_type"]; comp = S["test"].cluster.values
P_h = {n: np.stack([np.load(f"{RUNS}/{n}_lr{BEST_LR[n]:g}_s{s}/preds.npz")["test_gene_type"] for s in SEEDS]) for n, _ in ARMS}
corr = {n: (P_h[n].argmax(2) == yt).mean(0) for n, _ in ARMS}
uc = np.unique(comp); mem = [np.where(comp == c)[0] for c in uc]; rb = np.random.default_rng(0)
B = [np.concatenate([mem[j] for j in rb.integers(0, len(uc), len(uc))]) for _ in range(2000)]
lab = dict(ARMS); out = []
for a, b in [("MTL_Transformer_PEonly", "MTL_Transformer"), ("MTL_Transformer_MASKonly", "MTL_Transformer"),
             ("MTL_Transformer_PE", "MTL_Transformer"), ("MTL_Transformer_PE", "MTL_Transformer_PEonly"),
             ("MTL_Transformer_PE", "MTL_Transformer_MASKonly")]:
    d = corr[a] - corr[b]; bs = np.array([d[ix].mean() for ix in B]); lo, hi = np.percentile(bs, [2.5, 97.5])
    out.append([f"{lab[a]} − {lab[b]}", round(100 * d.mean(), 1), round(100 * lo, 1), round(100 * hi, 1),
                round(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 4)])
C = pd.DataFrame(out, columns=["contrast (human test, Gene Type)", "diff_pp", "lo", "hi", "p"])
print("\n" + C.to_string(index=False)); C.to_csv(f"{OUT}/Cell17_ablation_contrasts_human.csv", index=False)

# ---- external species (shared classes = classes present in the human test set) ----
shared = set(le.classes_[GT_TEST_LABELS]); ext = {}; erows = []
for sp in ["mouse", "chicken", "zebrafish"]:
    f = f"{OUT}/{sp}_external.csv"
    if not os.path.exists(f): print(f"  [{sp}] {f} not found; skipped"); continue
    E = pd.read_csv(f); E = E[E.GeneType.isin(shared)].reset_index(drop=True)
    Xe = np.array([enc(s) for s in E.seq], dtype=np.int32); ye = le.transform(E.GeneType)
    ext[f"{sp}_gene_ids"] = E.NCBIGeneID.values; ext[f"{sp}_y"] = ye
    for n, l in ARMS:
        Pe = np.stack([gt(load(n, s).predict(Xe, batch_size=256, verbose=0)) for s in SEEDS]).astype(np.float16)
        ext[f"{sp}__{n}"] = Pe; yp = Pe.argmax(2)
        acc = (yp == ye).mean(1); bacc = np.array([np.mean([(p[ye == c] == c).mean() for c in np.unique(ye)]) for p in yp])
        erows.append([sp, l, acc.mean(), acc.std(ddof=1), bacc.mean(), bacc.std(ddof=1)])
        print(f"  [{sp}] {l:20s} acc {acc.mean():.3f} ± {acc.std(ddof=1):.3f}  balanced {bacc.mean():.3f} ± {bacc.std(ddof=1):.3f}", flush=True)
ER = pd.DataFrame(erows, columns=["species", "arm", "acc_shared", "acc_sd", "bacc_shared", "bacc_sd"]); ER.to_csv(f"{OUT}/Cell17_ablation_external.csv", index=False)
np.savez_compressed(f"{OUT}/Cell17_ablation_predictions.npz", test_gene_ids=S["test"].NCBIGeneID.values, test_y=yt,
                    test_cluster=comp, classes=le.classes_, **{f"human__{n}": P_h[n].astype(np.float16) for n, _ in ARMS}, **ext)
json.dump(sens, open(f"{OUT}/Cell17_order_sensitivity.json", "w"), indent=1)

# ---- package results (no weights) ----
pkg = "/content/Cell17_ablation_results" if IN_COLAB else f"{OUT}_pkg"; shutil.rmtree(pkg, ignore_errors=True); os.makedirs(pkg)
for f in glob.glob(f"{OUT}/Cell17_*"): shutil.copy(f, pkg)
zf = shutil.make_archive(pkg, "zip", pkg); print("\nSend me this file:", zf)
try:
    from google.colab import files; files.download(zf)
except Exception:
    pass
