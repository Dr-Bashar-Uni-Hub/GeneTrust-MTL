# ======================================================================
# CELL 1 — REPRODUCTION CHECK + LEAKAGE AUDIT  (no training, ~3 min on A100)
# Re-run of frozen results; post hoc audit. Nothing here changes your models.
# ======================================================================
import os, glob, json, zipfile, itertools, hashlib
from collections import Counter
import numpy as np, pandas as pd, tensorflow as tf
from tensorflow.keras.layers import (Input, Embedding, Dense, Dropout, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D)
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score

# ---- 1. Locate checkpoint ZIP (loose, nested inside another zip, or upload)
CKPT_ZIP = None          # optional: paste the full path here to skip the search
try:
    from google.colab import drive
    drive.mount("/content/drive")
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
WORK = "/content/rev" if IN_COLAB else "/tmp/rev"
os.makedirs(WORK, exist_ok=True)
NAME = "Genomic_Transformer_Checkpoint_"
roots = ["/content", "/content/drive/MyDrive"] if IN_COLAB else ["/tmp"]

def find_loose():
    hits = []
    for r in roots:
        hits += glob.glob(f"{r}/**/{NAME}*.zip", recursive=True)
    return max(hits, key=os.path.getmtime) if hits else None

def find_nested():
    # look inside other zips (e.g. Results.zip) for the checkpoint zip
    outer = []
    for r in roots:
        outer += [z for z in glob.glob(f"{r}/**/*.zip", recursive=True) if NAME not in os.path.basename(z)]
    for z in sorted(outer, key=os.path.getmtime, reverse=True):
        try:
            with zipfile.ZipFile(z) as zf:
                inner = [n for n in zf.namelist() if os.path.basename(n).startswith(NAME) and n.endswith(".zip")]
                if inner:
                    print(f"Found {inner[0]} inside {z}; extracting it...")
                    return zf.extract(inner[0], WORK)
        except zipfile.BadZipFile:
            continue
    return None

if CKPT_ZIP is None:
    CKPT_ZIP = find_loose() or find_nested()
if CKPT_ZIP is None and IN_COLAB:
    from google.colab import files
    print("Checkpoint not found in Drive. Please upload Genomic_Transformer_Checkpoint_*.zip "
          "or Results.zip:")
    up = files.upload()
    roots = ["/content"]
    CKPT_ZIP = find_loose() or find_nested()
if CKPT_ZIP is None:
    raise FileNotFoundError("Checkpoint zip still not found; set CKPT_ZIP to its full path.")
print("Checkpoint source:", CKPT_ZIP)

def locate_root(base):
    """Folder holding data/train.csv and Results/Transformer_Baseline.keras."""
    for d, _, files in os.walk(base):
        if os.path.exists(f"{d}/data/train.csv") and os.path.exists(f"{d}/Results/Transformer_Baseline.keras"):
            return d
    return None

# Extract, then keep unpacking nested checkpoint zips until the folder appears
with zipfile.ZipFile(CKPT_ZIP) as z:
    z.extractall(WORK)
root, done = locate_root(WORK), set()
while root is None:
    inner = [z for z in glob.glob(f"{WORK}/**/{NAME}*.zip", recursive=True) if z not in done]
    if not inner:
        raise FileNotFoundError(f"No checkpoint folder (data/ + Results/) found inside {CKPT_ZIP}")
    print("Extracting nested", os.path.relpath(inner[0], WORK))
    with zipfile.ZipFile(inner[0]) as z:
        z.extractall(os.path.dirname(inner[0]))
    done.add(inner[0]); root = locate_root(WORK)
print("Checkpoint folder:", root)
DATA, RES = f"{root}/data", f"{root}/Results"
OUT = "/content/drive/MyDrive/JBSD_Revision" if IN_COLAB else f"{WORK}/out"
os.makedirs(OUT, exist_ok=True)

# ---- 2. Data + preprocessing exactly as in the original notebook -------
for f in ["train.csv", "validation.csv", "test.csv"]:
    print(f, hashlib.sha256(open(f"{DATA}/{f}", "rb").read()).hexdigest()[:16])
V = set("ACGT")
def roles(d):
    d = str(d).lower(); r = []
    if "pseudo" in d: r.append("PSEUDO")
    if "regulatory" in d: r.append("REGULATORY")
    if "immune" in d: r.append("IMMUNE")
    if "protein" in d: r.append("PROTEIN_CODING")
    return r or ["OTHER"]
def fam(s):
    s = str(s)
    if s.startswith("IGH"): return "IMMUNOGLOBULIN"
    if s.startswith("RNU") or s.startswith("SNORD"): return "RNA_GENE"
    if s.startswith("LOC"): return "LOCUS"
    return "OTHER"
D = {}
for k in ["train", "validation", "test"]:
    df = pd.read_csv(f"{DATA}/{k}.csv")
    df["seq"] = df.NucleotideSequence.map(lambda s: "".join(c for c in str(s).upper() if c in V))
    df["fam"] = df.Symbol.map(fam); df["roles"] = df.Description.map(roles)
    D[k] = df
tr, va, te = D["train"], D["validation"], D["test"]
cnt = Counter()
for s in tr.seq: cnt.update(s[i:i+6] for i in range(len(s) - 5))
vocab = {k: i + 1 for i, (k, _) in enumerate(cnt.most_common())}
def enc(s):
    e = [vocab.get(s[i:i+6], 0) for i in range(len(s) - 5)][:512]
    return e + [0] * (512 - len(e))
Xte = np.array([enc(s) for s in te.seq])
le = LabelEncoder().fit(tr.GeneType); yg = le.transform(te.GeneType)
fe = LabelEncoder().fit(tr.fam);      ys = fe.transform(te.fam)
mlb = MultiLabelBinarizer().fit(tr.roles); yr = mlb.transform(te.roles)

# ---- 3. Rebuild architecture and load the frozen seed-42 weights -------
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim=64, num_heads=4, ff_dim=128, dropout_rate=0.2, **kw):
        super().__init__(**kw)
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm1 = LayerNormalization(epsilon=1e-6); self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate); self.dropout2 = Dropout(dropout_rate)
    def call(self, x, training=False):
        a = self.dropout1(self.attention(x, x), training=training); o = self.layernorm1(x + a)
        return self.layernorm2(o + self.dropout2(self.ffn(o), training=training))
def build_original():
    i = Input((512,)); x = Embedding(4097, 64, name="Embedding")(i)
    for _ in range(2): x = TransformerEncoder()(x)
    x = Dropout(0.2)(GlobalAveragePooling1D()(x))
    return tf.keras.Model(i, [Dense(10, activation="softmax", name="gene_type")(x),
                              Dense(4, activation="softmax", name="symbol_family")(x),
                              Dense(4, activation="sigmoid", name="functional_roles")(x)])
model = build_original()
model.load_weights(f"{RES}/Transformer_Baseline.keras")
p = model.predict(Xte, batch_size=64, verbose=0)
pg, ps, pr = p[0].argmax(1), p[1].argmax(1), (p[2] >= 0.5).astype(int)

# ---- 4. Reproduction check against the frozen manuscript values --------
frozen = {"GT_acc": 0.9288, "GT_mF1": 0.7598, "SF_acc": 0.9287, "SF_mF1": 0.9261,
          "FR_exact": 0.8068, "FR_mF1": 0.8271}
def metrics(m):
    return {"GT_acc": accuracy_score(yg[m], pg[m]), "GT_mF1": f1_score(yg[m], pg[m], average="macro"),
            "SF_acc": accuracy_score(ys[m], ps[m]), "SF_mF1": f1_score(ys[m], ps[m], average="macro"),
            "FR_exact": (yr[m] == pr[m]).all(1).mean(),
            "FR_mF1": f1_score(yr[m], pr[m], average="macro", zero_division=0)}
full = metrics(np.ones(len(te), bool))
print("\nREPRODUCTION CHECK (tolerance 0.0015)")
for k, v in frozen.items():
    print(f"  {k:9s} frozen {v:.4f}  reproduced {full[k]:.4f}  {'PASS' if abs(v-full[k]) < 0.0015 else 'FAIL'}")

# ---- 5. Leakage audit ----------------------------------------------------
key = ["NCBIGeneID", "Symbol", "Description", "GeneType", "NucleotideSequence"]
trk = set(map(tuple, tr[key].values)); trseq = set(tr.seq); trid = set(tr.NCBIGeneID)
leak = {
    "val_rows_identical_to_train": int(sum(tuple(r) in trk for r in va[key].values)), "val_n": len(va),
    "test_rows_identical_to_train": int(sum(tuple(r) in trk for r in te[key].values)), "test_n": len(te),
    "test_seq_exact_dup_in_train": int(te.seq.isin(trseq).sum()),
}
clean = (~te.NCBIGeneID.isin(trid) & ~te.seq.isin(trseq)).values
leak["test_clean_n"] = int(clean.sum())
print("\nLEAKAGE AUDIT:", json.dumps(leak, indent=1))
sub = {"full": full, "overlap": metrics(~clean), "clean": metrics(clean)}
print(pd.DataFrame(sub).T.round(4))

# ---- 6. Order-invariance test (no positional encoding) -----------------
rng = np.random.default_rng(0); S = Xte[rng.choice(len(Xte), 300, replace=False)]; P = S.copy()
for r in P:
    v = np.where(r != 0)[0]; r[v] = r[rng.permutation(v)]
a, b = model.predict(S, verbose=0), model.predict(P, verbose=0)
perm = {n: float(np.abs(a[k] - b[k]).max()) for k, n in enumerate(["GT", "SF", "FR"])}
print("\nMax |Δprob| after shuffling 6-mer order within each sequence:", perm)

json.dump({"reproduction": full, "leakage": leak, "subsets": sub, "order_invariance": perm},
          open(f"{OUT}/Cell01_audit.json", "w"), indent=1, default=float)
np.save(f"{OUT}/test_clean_mask.npy", clean)
print("\nSaved to", OUT, "— please send me the full printed output.")
