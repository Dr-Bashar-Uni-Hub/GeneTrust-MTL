# ======================================================================
# CELL 10 — EXTERNAL VALIDATION SET: MOUSE (Mus musculus) GENES  (post hoc)
# CPU only; ~20–40 min (mostly NCBI download). Can run in any session.
# Same construction rule as the original human dataset: live NCBI Gene
# records whose FULL genomic sequence is <= 1,000 nt, same Gene Type
# classes; random cap per class. Cross-species homology to the human
# TRAINING partition is measured so strict subsets can be evaluated.
# ======================================================================
MAX_LEN, CAP_PER_CLASS, SEED = 1000, 3000, 2026
import os, re, json, glob, gzip, shutil, subprocess, zipfile
import numpy as np, pandas as pd

try:
    from google.colab import drive; drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
OUT = "/content/drive/MyDrive/JBSD_Revision" if IN_COLAB else "/tmp/rev/c10out"
W = "/content/mouse" if IN_COLAB else "/tmp/mouse"
os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True)
CLASSES = ["BIOLOGICAL_REGION", "OTHER", "PROTEIN_CODING", "PSEUDO", "ncRNA",
           "rRNA", "scRNA", "snRNA", "snoRNA", "tRNA"]
CANON = {c.upper(): c for c in CLASSES}

def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-1500:]); raise RuntimeError(cmd)
    return r.stdout

# ---- helpers (format-robust parsing of NCBI Datasets output) ------------
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

if __name__ == "__main__" and os.environ.get("C10_PARSE_ONLY") != "1":
    # ---- 1. NCBI Datasets command-line tools -----------------------------
    if shutil.which("datasets") is None:
        base = "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2/linux-amd64"
        sh(f"wget -q {base}/datasets -O /usr/local/bin/datasets && wget -q {base}/dataformat -O /usr/local/bin/dataformat "
           "&& chmod +x /usr/local/bin/datasets /usr/local/bin/dataformat")
    print(sh("datasets --version").strip())

    # ---- 2. Gene reports for all mouse genes -----------------------------
    rep = f"{W}/mouse_genes.jsonl"
    if not os.path.exists(rep) or os.path.getsize(rep) == 0:
        sh(f"datasets summary gene taxon 10090 --as-json-lines > {rep}")
    R = pd.DataFrame([parse_report(l) for l in open(rep) if l.strip()])
    print(f"Mouse gene reports: {len(R):,}; raw types:", R.type_raw.value_counts().head(15).to_dict())
    R = R[R.GeneType.notna() & R.glen.notna() & (R.glen <= MAX_LEN)]
    print("Genes <= 1,000 nt by class:", R.GeneType.value_counts().to_dict())
    R = (R.groupby("GeneType", group_keys=False)
           .apply(lambda x: x.sample(min(len(x), CAP_PER_CLASS), random_state=SEED)).reset_index(drop=True))

    # ---- 3. Download gene sequences --------------------------------------
    ids = f"{W}/ids.txt"; R.NCBIGeneID.astype(str).to_csv(ids, index=False, header=False)
    zf = f"{W}/mouse_genes.zip"
    if not os.path.exists(zf):
        sh(f"datasets download gene gene-id --inputfile {ids} --include gene --filename {zf}")
    with zipfile.ZipFile(zf) as z: z.extractall(f"{W}/dl")
    fna = glob.glob(f"{W}/dl/**/gene.fna", recursive=True)[0]
    seqs = parse_fasta(fna)
    V = set("ACGT")
    R["seq"] = R.NCBIGeneID.map(lambda g: "".join(c for c in seqs.get(g, "").upper() if c in V))
    R["L"] = R.seq.str.len()
    R = R[(R.L >= 6) & (R.L <= MAX_LEN)].reset_index(drop=True)
    print(f"Sequences retrieved: {len(R):,}", R.GeneType.value_counts().to_dict())

    # ---- 4. Cross-species homology to the HUMAN training partition --------
    if shutil.which("mmseqs") is None:
        sh("wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mm.tgz && "
           "tar xzf /tmp/mm.tgz -C /tmp && cp /tmp/mmseqs/bin/mmseqs /usr/local/bin/")
    DATA = ([os.path.dirname(f) for f in glob.glob("/content/rev*/**/data/train.csv", recursive=True)] or
            [os.path.dirname(f) for f in glob.glob("/tmp/rev*/**/data/train.csv", recursive=True)])[0]
    raw = pd.concat([pd.read_csv(f"{DATA}/train.csv"), pd.read_csv(f"{DATA}/test.csv")]).drop_duplicates("NCBIGeneID")
    spl = pd.read_csv(f"{OUT}/split_homology80.csv")
    htr = raw[raw.NCBIGeneID.isin(spl.loc[spl.split == "train", "NCBIGeneID"])]
    with open(f"{W}/human_train.fa", "w") as f:
        for g, s in zip(htr.NCBIGeneID, htr.NucleotideSequence):
            s = "".join(c for c in str(s).upper() if c in V)
            if len(s) >= 6: f.write(f">h{g}\n{s}\n")
    with open(f"{W}/mouse.fa", "w") as f:
        for g, s in zip(R.NCBIGeneID, R.seq): f.write(f">m{g}\n{s}\n")
    m8 = f"{W}/m_vs_h.m8"
    sh(f"mmseqs easy-search {W}/mouse.fa {W}/human_train.fa {m8} {W}/tmp1 --search-type 3 -s 7.5 "
       f"--max-seqs 300 --threads 8 -v 1")
    L_ = dict(zip("m" + R.NCBIGeneID.astype(str), R.L))
    if os.path.getsize(m8):
        h = pd.read_csv(m8, sep="\t", header=None)
        h["eff"] = h[2] * (((h[7] - h[6]).abs() + 1) / h[0].map(L_)).clip(upper=1)
        best = h.groupby(0).eff.max()
    else:
        best = pd.Series(dtype=float)
    R["maxid_to_human_train"] = ("m" + R.NCBIGeneID.astype(str)).map(best).fillna(0.0)

    # ---- 5. Homology components WITHIN mouse (for the cluster bootstrap) ---
    import scipy.sparse as sps
    from scipy.sparse.csgraph import connected_components
    m8b = f"{W}/m_vs_m.m8"
    sh(f"mmseqs easy-search {W}/mouse.fa {W}/mouse.fa {m8b} {W}/tmp2 --search-type 3 "
       f"--min-seq-id 0.8 -c 0.8 --cov-mode 2 --max-seqs 1000 --threads 8 -v 1")
    ix = {f"m{g}": i for i, g in enumerate(R.NCBIGeneID)}
    e = pd.read_csv(m8b, sep="\t", header=None) if os.path.getsize(m8b) else pd.DataFrame({0: [], 1: []})
    e = e[e[0] != e[1]]
    G = sps.coo_matrix((np.ones(len(e)), (e[0].map(ix).values, e[1].map(ix).values)), shape=(len(R),) * 2)
    R["mouse_component"] = connected_components(G, directed=False)[1]

    # ---- 6. Report and save ----------------------------------------------
    R["strict80"] = R.maxid_to_human_train < 0.8; R["strict60"] = R.maxid_to_human_train < 0.6
    print("\nMouse external set:", len(R), "genes;", R.mouse_component.nunique(), "homology components")
    print(pd.DataFrame({"all": R.GeneType.value_counts(), "no >=80% human-train match": R[R.strict80].GeneType.value_counts(),
                        "no >=60% match": R[R.strict60].GeneType.value_counts()}).fillna(0).astype(int))
    print("Identity x coverage to human training:", pd.cut(R.maxid_to_human_train, [-0.01, 1e-4, 0.5, 0.6, 0.8, 1.0]).value_counts().sort_index().to_dict())
    R.drop(columns=["type_raw"]).to_csv(f"{OUT}/mouse_external.csv", index=False)
    print(f"\nSaved {OUT}/mouse_external.csv — please send me the printed output (the CSV stays in Drive for Cells 11–12).")
