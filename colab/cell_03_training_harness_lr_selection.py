# ======================================================================
# CELL 3 — TRAINING HARNESS + SMOKE TEST + LEARNING-RATE SELECTION
# Post hoc re-evaluation on the homology-aware split (Cell 2).
# Resumable: every finished run is stored in Drive and skipped on re-run.
# Float32, deterministic ops, LR grid {1e-4, 3e-4, 1e-3}. Progress is
# printed every 10 epochs (logging only; training is unaffected).
# ======================================================================
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

# ---- 4. Smoke test: parameter counts must match the original models -------
print("\nSMOKE TEST")
for name, (build, tasks, expected) in MODELS.items():
    n = build().count_params()
    print(f"  {name:20s} params {n:,}  expected {expected:,}  {'OK' if n == expected else 'MISMATCH'}")
    assert n == expected, name
np.savez_compressed(f"{OUT}/split_labels_test.npz", gene_ids=S["test"].NCBIGeneID.values,
                    gt=Y["test"]["gene_type"], sf=Y["test"]["symbol_family"], fr=Y["test"]["functional_roles"],
                    maxid=S["test"].test_maxid_to_train.values, gt_classes=le.classes_,
                    sf_classes=fe.classes_, fr_classes=mlb.classes_)
print("SMOKE TEST PASSED")

# ---- 5. Learning-rate selection on validation loss (seed 42) ------------
rows = []
for name in MODELS:
    for lr in LR_GRID:
        print(f"  -> {name} lr={lr:g} ...", flush=True)
        r = run(name, lr, 42); rows.append(r)
        print(f"  {name:20s} lr={lr:g}  best_val_loss={r['best_val_loss']:.4f}  "
              f"best_epoch={r['best_epoch']}/{r['epochs_run']}{'  (HIT CAP)' if r['hit_cap'] else ''}")
lrdf = pd.DataFrame(rows); lrdf.to_csv(f"{OUT}/Cell03_lr_search.csv", index=False)
BEST_LR = {n: float(g.loc[g.best_val_loss.idxmin(), "lr"]) for n, g in lrdf.groupby("model")}
json.dump(BEST_LR, open(f"{OUT}/Cell03_best_lr.json", "w"), indent=1)
print("\nSelected learning rates (by validation loss only):", BEST_LR)
