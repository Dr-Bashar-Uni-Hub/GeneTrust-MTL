# ======================================================================
# CELL 4 — FIVE-SEED RUNS AT THE SELECTED LEARNING RATES  (post hoc)
# Requires Cell 3 in this session (re-run Cell 3 after a disconnect: its
# finished runs are skipped instantly). ~2.5–3.5 h on A100, resumable.
# ======================================================================
import json, shutil
SEEDS = [1, 7, 21, 42, 100]
BEST_LR = json.load(open(f"{OUT}/Cell03_best_lr.json"))
rows = []
for name in MODELS:
    for seed in SEEDS:
        r = run(name, BEST_LR[name], seed); rows.append(r)
        keys = [k for k in ("GT_acc", "SF_acc", "FR_exact") if k in r]
        print(f"  {name:20s} seed {seed:3d}  " + "  ".join(f"{k}={r[k]:.4f}" for k in keys)
              + f"  best_epoch={r['best_epoch']}{'  (HIT CAP)' if r['hit_cap'] else ''}")
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/Cell04_seed_runs.csv", index=False)

# ---- Summary: mean ± SD over seeds ---------------------------------------
mcols = [c for c in ["GT_acc", "GT_mF1", "SF_acc", "SF_mF1", "FR_exact", "FR_mF1", "FR_microF1"] if c in res]
summ = res.groupby("model")[mcols + ["best_epoch", "train_s", "infer_ms_per_seq"]].agg(["mean", "std"])
summ.to_csv(f"{OUT}/Cell04_summary.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print("\nMEAN ± SD OVER 5 SEEDS (homology-aware test set)")
print(res.groupby("model")[mcols].agg(lambda x: f"{x.mean():.4f} ± {x.std():.4f}"))
print("\nRuns that stopped at the epoch cap:", int(res.hit_cap.sum()), "of", len(res))

# ---- Paired MTL − STL differences (same seed) ----------------------------
pairs = [("STL_GeneType", ["GT_acc", "GT_mF1"]), ("STL_SymbolFamily", ["SF_acc", "SF_mF1"]),
         ("STL_FunctionalRoles", ["FR_exact", "FR_mF1"])]
mtl = res[res.model == "MTL_Transformer"].set_index("seed")
out = []
for stl, cols in pairs:
    s = res[res.model == stl].set_index("seed")
    for c in cols:
        d = (mtl[c] - s[c]).dropna()
        out.append({"metric": c, "mean_MTL_minus_STL": d.mean(), "sd": d.std(),
                    "n_seeds_MTL_better": int((d > 0).sum()), "n": len(d)})
dd = pd.DataFrame(out); dd.to_csv(f"{OUT}/Cell04_MTL_minus_STL.csv", index=False)
print("\nPAIRED MTL − STL (per seed):\n", dd.round(4).to_string(index=False))

# ---- Package results for upload (predictions + metrics; no weights) ------
pkg = "/content/JBSD_Revision_results" if os.path.exists("/content") else f"{OUT}_pkg"
shutil.rmtree(pkg, ignore_errors=True)
shutil.copytree(OUT, pkg, ignore=shutil.ignore_patterns("*.weights.h5"))
zipf = shutil.make_archive(pkg, "zip", pkg)
print("\nUpload this file to me:", zipf, f"({os.path.getsize(zipf)/1e6:.1f} MB)")
try:
    from google.colab import files; files.download(zipf)
except Exception:
    pass
