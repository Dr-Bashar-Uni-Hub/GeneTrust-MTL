# ======================================================================
# CELL 13 — ADDITIONAL EXTERNAL SPECIES: CHICKEN AND ZEBRAFISH  (post hoc)
# CPU only; ~20–40 min per species (mostly NCBI download). Runs in any session:
# it finds the human data itself (unpacked by Cell 1, the checkpoint zip in
# Drive, loose Kaggle CSVs in Drive, or upload). Needs split_homology80.csv
# in MyDrive/JBSD_Revision/.
# Generalizes Cell 10 (mouse) to more distant vertebrates, so that
# cross-species transfer can be measured along a divergence gradient:
#   human -> mouse (~87 Mya) -> chicken (~319 Mya) -> zebrafish (~429 Mya)
# Construction rule is IDENTICAL to Cell 10: live NCBI Gene records whose
# FULL genomic sequence is <= 1,000 nt, same Gene Type classes, random cap
# per class (seed 2026). Homology to the HUMAN TRAINING partition is
# measured with the SAME rule as the partitions (identity >= thr covering
# >= 80% of EITHER sequence), so strict subsets match Table 6 directly.
# Run once per species: set SPECIES below (or loop over both).
# ======================================================================
SPECIES_TO_BUILD = ["chicken", "zebrafish"]        # edit to build one only
MAX_LEN, CAP_PER_CLASS, SEED = 1000, 3000, 2026
TAXA = {"mouse": 10090, "chicken": 9031, "zebrafish": 7955}
DIVERGENCE_MYA = {"mouse": 87, "chicken": 319, "zebrafish": 429}   # TimeTree 5 (Kumar et al., 2022), median estimates

import os, re, json, glob, shutil, subprocess, zipfile
import numpy as np, pandas as pd

try:
    from google.colab import drive; drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
OUT = "/content/drive/MyDrive/JBSD_Revision" if IN_COLAB else "/tmp/rev/c13out"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["BIOLOGICAL_REGION", "OTHER", "PROTEIN_CODING", "PSEUDO", "ncRNA",
           "rRNA", "scRNA", "snRNA", "snoRNA", "tRNA"]
CANON = {c.upper(): c for c in CLASSES}
V = set("ACGT")

def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:]); raise RuntimeError(cmd)
    return r.stdout

# ---- helpers (identical to Cell 10) -------------------------------------
def walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values(): yield from walk(v)
    elif isinstance(o, list):
        for v in o: yield from walk(v)

def parse_report(line):
    """One JSON line from `datasets summary gene ... --as-json-lines`."""
    j = json.loads(line); g = j.get("gene", j)
    gid = g.get("gene_id") or g.get("geneId")
    typ = str(g.get("type", g.get("gene_type", ""))).upper().replace(" ", "_")
    lens = []
    for d in walk(g.get("annotations", g)):
        r = d.get("genomic_range") or d.get("genomicRange") or d.get("range")
        if isinstance(r, dict) and "begin" in r and "end" in r:
            lens.append(abs(int(r["end"]) - int(r["begin"])) + 1)
        elif isinstance(r, list):
            for x in r:
                if isinstance(x, dict) and "begin" in x and "end" in x:
                    lens.append(abs(int(x["end"]) - int(x["begin"])) + 1)
    return {"NCBIGeneID": int(gid) if gid else None, "Symbol": g.get("symbol", ""),
            "Description": g.get("description", ""), "type_raw": typ,
            "GeneType": CANON.get(typ), "glen": min(lens) if lens else None}

def parse_fasta(path):
    seqs, gid, buf = {}, None, []
    for line in open(path):
        if line.startswith(">"):
            if gid is not None and gid not in seqs: seqs[gid] = "".join(buf)
            m = re.search(r"GeneID[=:](\d+)", line); gid = int(m.group(1)) if m else None; buf = []
        else:
            buf.append(line.strip())
    if gid is not None and gid not in seqs: seqs[gid] = "".join(buf)
    return seqs

M8_COLS = "query,target,fident,qstart,qend,tstart,tend,qlen,tlen"
def homology_to_human(m8_path, prefix):
    """From one MMseqs2 search (custom columns) derive, per query gene:
       - maxid: max over hits of identity x query coverage (descriptive score, as Cell 10)
       - rule80 / rule60: True if ANY hit has identity >= thr and covers >= 80% of
         EITHER sequence (the partitioning rule; as analysis/mouse_strict_rule.py)."""
    if not os.path.exists(m8_path) or os.path.getsize(m8_path) == 0:
        return pd.Series(dtype=float), set(), set()
    h = pd.read_csv(m8_path, sep="\t", header=None, names=M8_COLS.split(","))
    h["qcov"] = ((h.qend - h.qstart).abs() + 1) / h.qlen
    h["tcov"] = ((h.tend - h.tstart).abs() + 1) / h.tlen
    h["eff"] = h.fident * h.qcov.clip(upper=1)
    maxid = h.groupby("query").eff.max()
    either = (h.qcov >= 0.8) | (h.tcov >= 0.8)
    m80 = set(h.loc[(h.fident >= 0.8) & either, "query"]); m60 = set(h.loc[(h.fident >= 0.6) & either, "query"])
    to_id = lambda s: {int(q[len(prefix):]) for q in s}
    return maxid.rename(lambda q: int(q[len(prefix):])), to_id(m80), to_id(m60)

def components_within(fa, ids, W, prefix):
    import scipy.sparse as sps
    from scipy.sparse.csgraph import connected_components
    m8b = f"{W}/self.m8"
    sh(f"mmseqs easy-search {fa} {fa} {m8b} {W}/tmp_self --search-type 3 "
       f"--min-seq-id 0.8 -c 0.8 --cov-mode 2 --max-seqs 1000 --threads 8 -v 1")
    ix = {f"{prefix}{g}": i for i, g in enumerate(ids)}
    e = pd.read_csv(m8b, sep="\t", header=None) if os.path.getsize(m8b) else pd.DataFrame({0: [], 1: []})
    e = e[e[0] != e[1]]
    G = sps.coo_matrix((np.ones(len(e)), (e[0].map(ix).values, e[1].map(ix).values)), shape=(len(ids),) * 2)
    return connected_components(G, directed=False)[1]

def trna_orientation(seqs):
    def find(seq, pat, lo, hi):
        m = [x.start() for x in re.finditer(f"(?=({pat}))", seq) if lo <= x.start() <= hi]
        return m[0] if m else -1
    a = np.mean([find(s, "T[AG]G[CT]..A[AG]..G", 0, 30) >= 0 for s in seqs]) if len(seqs) else np.nan
    b = np.mean([find(s, "GTTC[AG]A.[CT]C", 40, len(s)) >= 0 for s in seqs]) if len(seqs) else np.nan
    return a, b

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

def build_species(sp, human_train_fa):
    taxon, W, pfx = TAXA[sp], (f"/content/{sp}" if IN_COLAB else f"/tmp/{sp}"), sp[0]
    os.makedirs(W, exist_ok=True)
    # ---- 1. gene reports --------------------------------------------------
    rep = f"{W}/{sp}_genes.jsonl"
    if not os.path.exists(rep) or os.path.getsize(rep) == 0:
        sh(f"datasets summary gene taxon {taxon} --as-json-lines > {rep}")
    R = pd.DataFrame([parse_report(l) for l in open(rep) if l.strip()])
    n_reports = len(R)
    print(f"\n[{sp}] gene reports: {n_reports:,}; raw types:", R.type_raw.value_counts().head(15).to_dict())
    R = R[R.GeneType.notna() & R.glen.notna() & (R.glen <= MAX_LEN)]
    eligible = R.GeneType.value_counts().to_dict(); n_eligible = len(R)
    print(f"[{sp}] genes <= 1,000 nt by class:", eligible)
    R = (R.groupby("GeneType", group_keys=False)
           .apply(lambda x: x.sample(min(len(x), CAP_PER_CLASS), random_state=SEED)).reset_index(drop=True))
    # ---- 2. sequences -----------------------------------------------------
    ids = f"{W}/ids.txt"; R.NCBIGeneID.astype(str).to_csv(ids, index=False, header=False)
    zf = f"{W}/{sp}_genes.zip"
    if not os.path.exists(zf):
        sh(f"datasets download gene gene-id --inputfile {ids} --include gene --filename {zf}")
    with zipfile.ZipFile(zf) as z: z.extractall(f"{W}/dl")
    seqs = parse_fasta(glob.glob(f"{W}/dl/**/gene.fna", recursive=True)[0])
    R["seq"] = R.NCBIGeneID.map(lambda g: "".join(c for c in seqs.get(g, "").upper() if c in V))
    R["L"] = R.seq.str.len()
    R = R[(R.L >= 6) & (R.L <= MAX_LEN)].reset_index(drop=True)
    print(f"[{sp}] sequences retrieved: {len(R):,}", R.GeneType.value_counts().to_dict())
    oa, ob = trna_orientation(R.loc[R.GeneType == "tRNA", "seq"].tolist())
    print(f"[{sp}] orientation check (tRNA): A box {oa:.0%}, B box {ob:.0%} (human 90%/94%, mouse 90%/84%)")
    # ---- 3. homology to human TRAINING partition (one search, both rules) -
    fa = f"{W}/{sp}.fa"
    with open(fa, "w") as f:
        for g, s in zip(R.NCBIGeneID, R.seq): f.write(f">{pfx}{g}\n{s}\n")
    m8 = f"{W}/{sp}_vs_h.m8"
    sh(f"mmseqs easy-search {fa} {human_train_fa} {m8} {W}/tmp1 --search-type 3 -s 7.5 "
       f"--max-seqs 300 --threads 8 -v 1 --format-output {M8_COLS}")
    maxid, m80, m60 = homology_to_human(m8, pfx)
    R["maxid_to_human_train"] = R.NCBIGeneID.map(maxid).fillna(0.0)
    R["strict80"] = ~R.NCBIGeneID.isin(m80)      # partitioning rule (as Table 6)
    R["strict60"] = ~R.NCBIGeneID.isin(m60)
    # ---- 4. homology components within species (bootstrap units) ---------
    R[f"{sp}_component"] = components_within(fa, R.NCBIGeneID.tolist(), W, pfx)
    R["component"] = R[f"{sp}_component"]
    # ---- 5. report and save ------------------------------------------------
    print(f"\n[{sp}] external set: {len(R):,} genes; {R.component.nunique():,} homology components")
    print(pd.DataFrame({"all": R.GeneType.value_counts(),
                        "no >=80% human-train match": R[R.strict80].GeneType.value_counts(),
                        "no >=60% match": R[R.strict60].GeneType.value_counts()}).fillna(0).astype(int))
    nohom = int((R.maxid_to_human_train <= 1e-4).sum())
    summ = {"species": sp, "taxon": taxon, "divergence_mya": DIVERGENCE_MYA[sp], "gene_reports": n_reports,
            "eligible_le_1000nt": n_eligible, "eligible_by_class": eligible, "n_genes": len(R),
            "by_class": R.GeneType.value_counts().to_dict(), "n_strict80": int(R.strict80.sum()),
            "n_strict60": int(R.strict60.sum()), "n_no_detectable_homolog": nohom,
            "n_components": int(R.component.nunique()), "trna_A_box_found": float(oa), "trna_B_box_found": float(ob),
            "seed": SEED, "cap_per_class": CAP_PER_CLASS, "max_len": MAX_LEN, "datasets_cli": sh("datasets --version").strip()}
    print(json.dumps({k: summ[k] for k in ["n_genes", "n_strict80", "n_strict60", "n_no_detectable_homolog", "n_components"]}))
    R.drop(columns=["type_raw", "component"]).to_csv(f"{OUT}/{sp}_external.csv", index=False)
    json.dump(summ, open(f"{OUT}/Cell13_{sp}_summary.json", "w"), indent=1)
    print(f"[{sp}] saved {OUT}/{sp}_external.csv and Cell13_{sp}_summary.json")

if __name__ == "__main__" and os.environ.get("C13_PARSE_ONLY") != "1":
    # ---- tools -------------------------------------------------------------
    if shutil.which("datasets") is None:
        base = "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64"
        sh(f"wget -q {base}/datasets -O /usr/local/bin/datasets && wget -q {base}/dataformat -O /usr/local/bin/dataformat "
           "&& chmod +x /usr/local/bin/datasets /usr/local/bin/dataformat")
    if shutil.which("mmseqs") is None:
        sh("wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mm.tgz && "
           "tar xzf /tmp/mm.tgz -C /tmp && cp /tmp/mmseqs/bin/mmseqs /usr/local/bin/")
    print(sh("datasets --version").strip(), "|", sh("mmseqs version").strip())
    # ---- human training sequences: found or unpacked automatically ---------
    DATA = find_human_data()
    print("Human data:", DATA)
    if not os.path.exists(f"{OUT}/split_homology80.csv"):
        raise FileNotFoundError(f"{OUT}/split_homology80.csv not found: copy data/split_homology80.csv from the "
                                "GitHub repo into MyDrive/JBSD_Revision/ (or run Cell 2).")
    raw = pd.concat([pd.read_csv(f"{DATA}/train.csv"), pd.read_csv(f"{DATA}/test.csv")]).drop_duplicates("NCBIGeneID")
    spl = pd.read_csv(f"{OUT}/split_homology80.csv")
    htr = raw[raw.NCBIGeneID.isin(spl.loc[spl.split == "train", "NCBIGeneID"])]
    hfa = "/content/human_train.fa" if IN_COLAB else "/tmp/human_train.fa"
    with open(hfa, "w") as f:
        for g, s in zip(htr.NCBIGeneID, htr.NucleotideSequence):
            s = "".join(c for c in str(s).upper() if c in V)
            if len(s) >= 6: f.write(f">h{g}\n{s}\n")
    print(f"Human training sequences: {len(htr):,}")
    for sp in SPECIES_TO_BUILD:
        build_species(sp, hfa)
    print("\nDone. Please send me the printed output and the Cell13_*_summary.json files "
          "(the *_external.csv files stay in Drive for Cells 14–15).")
