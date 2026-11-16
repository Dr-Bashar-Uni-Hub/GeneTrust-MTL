# ======================================================================
# CELL 19 — WHAT ARE THE DUPLICATED TEST SEQUENCES?  (standalone, CPU is enough; ~10–20 min)
# Follow-up to Cell 18. For each Genomic Benchmarks task, characterizes the provided test
# sequences that also occur in the training set (exact, either strand) or have a ≥80%-identity
# match (≥80% coverage of either sequence, MMseqs2, both strands):
#   - their length, compared with all test sequences (short sequences can match by chance)
#   - whether the matching training sequences carry the SAME label, the OTHER label, or both
#   - duplicates within the training set that carry conflicting labels
# Data are read exactly as in Cell 18 (sequences of ≥6 nt).
# Outputs: Drive MyDrive/JBSD_Revision/Cell19_duplicate_check.json and .csv, plus a zip to send back.
# ======================================================================
TASKS = ["human_enhancers_ensembl", "human_nontata_promoters"]
import os, glob, json, shutil, subprocess
import numpy as np, pandas as pd
from collections import defaultdict

try:
    from google.colab import drive; drive.mount("/content/drive"); IN_COLAB = True
except ImportError:
    IN_COLAB = False
OUT = "/content/drive/MyDrive/JBSD_Revision" if IN_COLAB else os.environ.get("C19_OUT", "/tmp/c19out")
W = "/content/gb" if IN_COLAB else os.environ.get("C19_WORK", "/tmp/gb")
os.makedirs(OUT, exist_ok=True); os.makedirs(W, exist_ok=True)
MM_EXTRA = os.environ.get("MMSEQS_EXTRA", "")

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0: print(r.stdout[-1500:], r.stderr[-1500:]); raise RuntimeError(cmd)
    return r.stdout
if IN_COLAB:
    sh("pip -q install genomic-benchmarks")
    if shutil.which("mmseqs") is None:
        sh("wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz -O /tmp/mm.tgz && tar xzf /tmp/mm.tgz -C /tmp && cp /tmp/mmseqs/bin/mmseqs /usr/local/bin/")

COMP = str.maketrans("ACGT", "TGCA")
def rc(s): return s.translate(COMP)[::-1]
def clean(s): return "".join(c for c in s.upper() if c in "ACGTN")
def load_task(name):                                    # identical to Cell 18
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
    D["uid"] = [f"q{i}" for i in range(len(D))]; D["L"] = D.seq.str.len()
    return D

def lenstats(L):
    L = np.asarray(L)
    if len(L) == 0: return {}
    return {"n": int(len(L)), "median": float(np.median(L)), "p10": float(np.percentile(L, 10)),
            "p90": float(np.percentile(L, 90)), "frac_lt_50nt": float((L < 50).mean()), "frac_lt_100nt": float((L < 100).mean())}

def agreement(test_labels, match_label_sets):
    """For each test sequence: training matches all share its label / all have the other label / mixed."""
    same = other = mixed = 0
    for lab, s in zip(test_labels, match_label_sets):
        if s == {lab}: same += 1
        elif lab not in s: other += 1
        else: mixed += 1
    n = max(1, same + other + mixed)
    return {"same_label": same, "other_label_only": other, "both_labels": mixed,
            "frac_same_label": same / n, "frac_other_label_only": other / n, "frac_both_labels": mixed / n}

res, rows = {}, []
for task in TASKS:
    D = load_task(task); tr, te = D[D.split == "train"], D[D.split == "test"]
    print(f"\n===== {task}: train {len(tr):,}, test {len(te):,}; test length median {te.L.median():.0f} "
          f"(range {te.L.min()}–{te.L.max()})", flush=True)
    # exact matches, either strand: map canonical sequence -> training labels
    key = lambda s: min(s, rc(s))
    tr_labels = defaultdict(set)
    for s, l in zip(tr.seq, tr.label): tr_labels[key(s)].add(l)
    te_key = te.seq.map(key); is_dup = te_key.isin(tr_labels.keys()).values
    dup = te[is_dup]
    A = agreement(dup.label, [tr_labels[key(s)] for s in dup.seq])
    conflict_train = sum(1 for v in tr_labels.values() if len(v) > 1)
    r = {"n_test": len(te), "exact_duplicates": int(is_dup.sum()), "frac_exact": float(is_dup.mean()),
         "length_all_test": lenstats(te.L), "length_exact_duplicates": lenstats(dup.L),
         "length_nonduplicates": lenstats(te[~is_dup].L), "exact_label_agreement": A,
         "distinct_training_sequences_with_both_labels": int(conflict_train),
         "exact_duplicates_ge_100nt": int((dup.L >= 100).sum())}
    # near matches (>=80% identity covering >=80% of either sequence), as in Cell 18
    for name, sub in [("tr", tr), ("te", te)]:
        with open(f"{W}/{task}_{name}.fa", "w") as f:
            for u, s in zip(sub.uid, sub.seq): f.write(f">{u}\n{s}\n")
    M8 = "query,target,fident,qstart,qend,tstart,tend,qlen,tlen"; m8 = f"{W}/{task}_c19.m8"
    sh(f"mmseqs easy-search {W}/{task}_te.fa {W}/{task}_tr.fa {m8} {W}/tmp_c19_{task} --search-type 3 --min-seq-id 0.8 -c 0 "
       f"--max-seqs 300 --threads 8 -v 1 --format-output {M8} {MM_EXTRA}")
    if os.path.getsize(m8):
        h = pd.read_csv(m8, sep="\t", header=None, names=M8.split(","))
        qc = ((h.qend - h.qstart).abs() + 1) / h.qlen; tc = ((h.tend - h.tstart).abs() + 1) / h.tlen
        h = h[(h.fident >= 0.8) & ((qc >= 0.8) | (tc >= 0.8))]
    else:
        h = pd.DataFrame(columns=M8.split(","))
    lab_of = dict(zip(D.uid, D.label)); L_of = dict(zip(D.uid, D.L))
    near_sets = h.groupby("query")["target"].apply(lambda t: {lab_of[x] for x in t}).to_dict()
    near_ids = list(near_sets)
    r.update({"near_matches_ge80": len(near_ids), "frac_near": len(near_ids) / len(te),
              "length_near_matches": lenstats([L_of[q] for q in near_ids]),
              "near_label_agreement": agreement([lab_of[q] for q in near_ids], [near_sets[q] for q in near_ids])})
    res[task] = r
    print(f"  exact duplicates: {r['exact_duplicates']:,} ({100 * r['frac_exact']:.1f}%); of these ≥100 nt: {r['exact_duplicates_ge_100nt']:,}")
    print(f"  length (median): all test {r['length_all_test']['median']:.0f} | exact duplicates "
          f"{r['length_exact_duplicates'].get('median', float('nan')):.0f} | non-duplicates {r['length_nonduplicates']['median']:.0f}")
    print(f"  duplicates shorter than 50 nt: {100 * r['length_exact_duplicates'].get('frac_lt_50nt', 0):.1f}%")
    print(f"  exact-duplicate labels: same {A['same_label']:,} | other only {A['other_label_only']:,} | both {A['both_labels']:,}")
    print(f"  training sequences occurring with both labels: {conflict_train:,}")
    NA = r["near_label_agreement"]
    print(f"  ≥80% matches: {len(near_ids):,} ({100 * r['frac_near']:.1f}%); labels same {NA['same_label']:,} | other only "
          f"{NA['other_label_only']:,} | both {NA['both_labels']:,}", flush=True)
    rows.append([task, r["n_test"], r["exact_duplicates"], round(100 * r["frac_exact"], 1), r["exact_duplicates_ge_100nt"],
                 r["length_all_test"]["median"], r["length_exact_duplicates"].get("median"),
                 round(100 * r["length_exact_duplicates"].get("frac_lt_50nt", 0), 1),
                 A["same_label"], A["other_label_only"], A["both_labels"], conflict_train,
                 r["near_matches_ge80"], round(100 * r["frac_near"], 1), NA["same_label"], NA["other_label_only"], NA["both_labels"]])

json.dump(res, open(f"{OUT}/Cell19_duplicate_check.json", "w"), indent=1)
pd.DataFrame(rows, columns=["task", "n_test", "exact_dup", "pct_exact", "exact_dup_ge100nt", "median_len_test", "median_len_dup",
                            "pct_dup_lt50nt", "dup_same_label", "dup_other_label", "dup_both_labels", "train_seqs_both_labels",
                            "near80", "pct_near80", "near_same_label", "near_other_label", "near_both_labels"]
             ).to_csv(f"{OUT}/Cell19_duplicate_check.csv", index=False)
pkg = "/content/Cell19_results" if IN_COLAB else f"{OUT}_pkg"; shutil.rmtree(pkg, ignore_errors=True); os.makedirs(pkg)
for f in glob.glob(f"{OUT}/Cell19_*"): shutil.copy(f, pkg)
zf = shutil.make_archive(pkg, "zip", pkg); print("\nSend me this file:", zf)
try:
    from google.colab import files; files.download(zf)
except Exception:
    pass
