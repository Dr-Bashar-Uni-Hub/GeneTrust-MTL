# ======================================================================
# CELL 12 — DNABERT ON THE MOUSE EXTERNAL SET  (post hoc)
# STANDALONE: run ALONE in a FRESH session (Runtime > Disconnect and delete
# runtime). Do not run Cells 1–7 first: TensorFlow reserves the GPU memory.
# Re-fine-tunes DNABERT-6 on the HUMAN 80% partition exactly as Cell 8 (the
# earlier weights were not kept), then predicts the MOUSE external set from
# Cell 10 (mouse_external.csv). 3 seeds. ~20–30 min on an A100. Resumable.
# ======================================================================
CKPT_ZIP = "/content/drive/MyDrive/Genomic_Transformer/Results.zip"   # same as Cell 1
MODEL_NAME = "zhihan1996/DNA_bert_6"
SEEDS, EPOCHS, LR, BATCH, ACCUM = [1, 21, 42], 4, 3e-5, 16, 2   # effective batch 32

import os, glob, json, zipfile, shutil, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate"], check=False)
import numpy as np, pandas as pd, torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding, set_seed)

IN_COLAB, DRIVE_OK = os.path.exists("/content"), False
if IN_COLAB:
    try:
        from google.colab import drive; drive.mount("/content/drive"); DRIVE_OK = True
    except Exception as e:
        print("Google Drive could not be mounted (", str(e)[:60], ") -> switching to UPLOAD mode.")
if DRIVE_OK:
    OUT = "/content/drive/MyDrive/JBSD_Revision"
elif IN_COLAB:
    # ---- Upload mode: no Drive needed ------------------------------------
    OUT = "/content/jbsd_local"; os.makedirs(OUT, exist_ok=True)
    need = {"split_homology80.csv": None, "mouse_external.csv": None, "zip": None}
    for f in glob.glob("/content/*"):
        b = os.path.basename(f)
        if b in need: need[b] = f
        elif b.endswith(".zip") and ("Results" in b or "Genomic_Transformer_Checkpoint" in b): need["zip"] = f
    missing = [k for k, v in need.items() if v is None]
    if missing:
        print("Please upload these files now (select all at once): split_homology80.csv, mouse_external.csv,",
              "and Results.zip (or Genomic_Transformer_Checkpoint_*.zip). Still missing:", missing)
        from google.colab import files
        up = files.upload()
        for b in up:
            src = os.path.join(os.getcwd(), b)
            if os.path.abspath(src) != os.path.join("/content", b): shutil.move(src, os.path.join("/content", b))
            if b.startswith("split_homology80"): need["split_homology80.csv"] = f"/content/{b}"
            elif b.startswith("mouse_external"): need["mouse_external.csv"] = f"/content/{b}"
            elif b.endswith(".zip"): need["zip"] = f"/content/{b}"
    assert all(need.values()), f"Still missing after upload: {[k for k, v in need.items() if v is None]}"
    shutil.copy(need["split_homology80.csv"], f"{OUT}/split_homology80.csv")
    shutil.copy(need["mouse_external.csv"], f"{OUT}/mouse_external.csv")
    CKPT_ZIP = need["zip"]
else:
    OUT = "/tmp/rev/c8out"
WORK = "/content/rev8" if IN_COLAB else "/tmp/rev8"
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

# ---- 1. Locate the original data files (unpack nested zips if needed) ---
def locate_data(base):
    for d, _, files in os.walk(base):
        if {"train.csv", "test.csv"} <= set(files) and os.path.basename(d) == "data":
            return d
    return None
DATA = locate_data(WORK) or (locate_data("/content/rev") if IN_COLAB else None)
if DATA is None:
    with zipfile.ZipFile(CKPT_ZIP) as z: z.extractall(WORK)
    done = set()
    while (DATA := locate_data(WORK)) is None:
        inner = [p for p in glob.glob(f"{WORK}/**/Genomic_Transformer_Checkpoint_*.zip", recursive=True) if p not in done]
        if not inner: raise FileNotFoundError("data/train.csv not found inside CKPT_ZIP")
        with zipfile.ZipFile(inner[0]) as z: z.extractall(os.path.dirname(inner[0]))
        done.add(inner[0])
print("Data:", DATA)

# ---- 2. Same split, cleaning and label encoding as Cell 3 ---------------
V = set("ACGT")
raw = pd.concat([pd.read_csv(f"{DATA}/train.csv"), pd.read_csv(f"{DATA}/test.csv")]).drop_duplicates("NCBIGeneID")
spl = pd.read_csv(f"{OUT}/split_homology80.csv")
P = raw.merge(spl[["NCBIGeneID", "split"]], on="NCBIGeneID")
P["seq"] = P.NucleotideSequence.map(lambda s: "".join(c for c in str(s).upper() if c in V))
S = {k: P[P.split == k].reset_index(drop=True) for k in ["train", "val", "test"]}
le = LabelEncoder().fit(S["train"].GeneType)
Y = {k: le.transform(v.GeneType) for k, v in S.items()}
labs = np.unique(Y["test"])
print({k: len(v) for k, v in S.items()}, "| classes:", list(le.classes_))

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
kmers = lambda s: " ".join(s[i:i+6] for i in range(min(len(s), 515) - 5))   # <=510 6-mers + [CLS]/[SEP]
class DS(torch.utils.data.Dataset):
    def __init__(self, seqs, y):
        self.enc = tok([kmers(s) for s in seqs], truncation=True, max_length=512); self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        d = {k: v[i] for k, v in self.enc.items()}; d["labels"] = int(self.y[i]); return d
ds = {k: DS(S[k].seq.tolist(), Y[k]) for k in S}
MOUSE = pd.read_csv(f"{OUT}/mouse_external.csv"); MOUSE = MOUSE[MOUSE.GeneType.isin(le.classes_)].reset_index(drop=True)
ds["mouse"] = DS(MOUSE.seq.tolist(), le.transform(MOUSE.GeneType))
print("Mouse external set:", len(MOUSE))
collate = DataCollatorWithPadding(tok)

import inspect, math
TA_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)
def args_for(seed, outdir):
    # Version-robust: works with Transformers 4.x and 5.x
    steps = math.ceil(len(ds["train"]) / (BATCH * ACCUM)) * EPOCHS
    kw = dict(output_dir=outdir, save_strategy="epoch", num_train_epochs=EPOCHS, learning_rate=LR,
              per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
              per_device_eval_batch_size=64, warmup_steps=int(0.1 * steps), weight_decay=0.01,
              fp16=torch.cuda.is_available(), load_best_model_at_end=True,
              metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=1,
              report_to="none", seed=seed, logging_steps=200, dataloader_num_workers=2)
    kw["eval_strategy" if "eval_strategy" in TA_PARAMS else "evaluation_strategy"] = "epoch"
    dropped = [k for k in kw if k not in TA_PARAMS]
    if dropped: print("  (TrainingArguments: not supported in this version, skipped:", dropped, ")")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in TA_PARAMS})

# ---- 3. Fine-tune, 3 seeds (resumable) ----------------------------------
RUNS = f"{OUT}/runs_fm_ext"; os.makedirs(RUNS, exist_ok=True)
rows = []
for seed in SEEDS:
    d = f"{RUNS}/DNABERT6_GeneType_s{seed}"
    if os.path.exists(f"{d}/metrics.json"):
        rows.append(json.load(open(f"{d}/metrics.json"))); print("  seed", seed, "already done"); continue
    os.makedirs(d, exist_ok=True); set_seed(seed); t0 = time.time()
    id2label = {i: c for i, c in enumerate(le.classes_)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id2label), id2label=id2label,
        label2id={c: i for i, c in id2label.items()}, ignore_mismatched_sizes=True)
    assert model.classifier.out_features == len(id2label), f"classifier has {model.classifier.out_features} outputs"
    print(f"  classifier outputs: {model.classifier.out_features} (expected {len(id2label)}) OK")
    tmp = f"/content/hf_tmp_s{seed}" if IN_COLAB else f"/tmp/hf_tmp_s{seed}"
    tr = Trainer(model=model, args=args_for(seed, tmp), train_dataset=ds["train"],
                 eval_dataset=ds["val"], data_collator=collate)
    tr.train(); train_s = time.time() - t0
    t1 = time.time(); pt = torch.softmax(torch.tensor(tr.predict(ds["test"]).predictions, dtype=torch.float32), -1).numpy()
    infer_ms = 1000 * (time.time() - t1) / len(ds["test"])
    pv = torch.softmax(torch.tensor(tr.predict(ds["val"]).predictions, dtype=torch.float32), -1).numpy()
    pm = torch.softmax(torch.tensor(tr.predict(ds["mouse"]).predictions, dtype=torch.float32), -1).numpy()
    hist = [h for h in tr.state.log_history if "eval_loss" in h]
    best = min(hist, key=lambda h: h["eval_loss"])
    yp = pt.argmax(1)
    res = {"model": "DNABERT6_GeneType", "seed": seed, "params": int(sum(p.numel() for p in model.parameters())),
           "best_epoch": float(best["epoch"]), "best_val_loss": float(best["eval_loss"]), "train_s": train_s,
           "infer_ms_per_seq": infer_ms, "GT_acc": accuracy_score(Y["test"], yp),
           "GT_mF1": f1_score(Y["test"], yp, labels=labs, average="macro", zero_division=0),
           "mouse_GT_acc": float((pm.argmax(1) == le.transform(MOUSE.GeneType)).mean())}
    np.savez_compressed(f"{d}/preds.npz", test_gene_type=pt, val_gene_type=pv, gene_ids=S["test"].NCBIGeneID.values,
                        mouse_gene_type=pm, mouse_gene_ids=MOUSE.NCBIGeneID.values)
    pd.DataFrame(tr.state.log_history).to_csv(f"{d}/history.csv", index=False)
    json.dump(res, open(f"{d}/metrics.json", "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True); del model, tr; torch.cuda.empty_cache()
    rows.append(res)
    print(f"  seed {seed}: human GT_acc={res['GT_acc']:.4f} | mouse GT_acc={res['mouse_GT_acc']:.4f} | best_epoch={res['best_epoch']} ({train_s/60:.0f} min)")

r = pd.DataFrame(rows); r.to_csv(f"{OUT}/Cell12_dnabert_mouse_runs.csv", index=False)
print(f"\nDNABERT-6 mouse external accuracy {r.mouse_GT_acc.mean():.4f} ± {r.mouse_GT_acc.std():.4f} | human test {r.GT_acc.mean():.4f}")
print(f"(human) acc {r.GT_acc.mean():.4f} ± {r.GT_acc.std():.4f} | "
      f"macro-F1 {r.GT_mF1.mean():.4f} ± {r.GT_mF1.std():.4f} | params {r.params.iloc[0]:,}")
pkg = f"{WORK}/JBSD_mouse_dnabert"; shutil.rmtree(pkg, ignore_errors=True)
shutil.copytree(RUNS, f"{pkg}/runs_fm_ext"); shutil.copy(f"{OUT}/Cell12_dnabert_mouse_runs.csv", pkg)
for f_ in ["mouse_external.csv", "Cell11_mouse_predictions.npz"]:
    if os.path.exists(f"{OUT}/{f_}"): shutil.copy(f"{OUT}/{f_}", pkg)
zipf = shutil.make_archive(pkg, "zip", pkg); print("Upload this file to me:", zipf, "(results are NOT on Drive in upload mode — download it now)" if not DRIVE_OK else "")
try:
    from google.colab import files; files.download(zipf)
except Exception:
    pass
