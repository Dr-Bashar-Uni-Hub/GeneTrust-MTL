# ======================================================================
# CELL 5 — INTEGRATED GRADIENTS ON tRNA / snoRNA TEST SEQUENCES (post hoc)
# Run Cell 1, then Cell 3 first (Cell 3 skips all finished runs in ~2 min).
# Models: 5 seeds of the order-aware MTL Transformer (PE), 5 seeds of the
# original order-blind MTL Transformer, and an untrained PE model as a
# model-randomisation control. Target = probability of the TRUE Gene Type.
# Motif statistics are computed afterwards from the saved attributions.
# ~5–10 min on a T4.
# ======================================================================
import numpy as np, tensorflow as tf, json, time

IG_STEPS = 32
BEST_LR = json.load(open(f"{OUT}/Cell03_best_lr.json"))
SEEDS = [1, 7, 21, 42, 100]
sel = np.where(S["test"].GeneType.isin(["tRNA", "snoRNA"]).values)[0]
Xs, ys = X["test"][sel], Y["test"]["gene_type"][sel]
print(f"Sequences: {len(sel)}  ({dict(S['test'].GeneType.iloc[sel].value_counts())})")

def ig_forward(model, pos_mask):
    """Same network, but taking the token-embedding tensor as a second input."""
    tok = tf.keras.Input((512,), dtype="int32"); emb = tf.keras.Input((512, 64))
    lay = {type(l).__name__: [] for l in model.layers}
    for l in model.layers: lay[type(l).__name__].append(l)
    x, m = emb, None
    if pos_mask:
        m = lay["PadMask"][0](tok); x = lay["PositionEmbedding"][0](x)
    for enc in lay["TransformerEncoder"]: x = enc(x, mask=m)
    z = lay["MaskedMeanPool"][0](x, m) if pos_mask else lay["GlobalAveragePooling1D"][0](x)
    z = lay["Dropout"][-1](z)
    out = model.get_layer("gene_type")(z)
    if pos_mask:
        f = tf.keras.Model([tok, emb], out)
    else:                                   # order-blind model: tokens not needed
        g = tf.keras.Model(emb, out)
        f = lambda inputs, training=False: g(inputs[1], training=training)
    return f, model.get_layer("Embedding")

def integrated_gradients(model, pos_mask):
    f, E = ig_forward(model, pos_mask)
    alphas = tf.reshape(tf.linspace(0.0, 1.0, IG_STEPS + 1), (-1, 1, 1))
    att, prob, base = np.zeros(Xs.shape, np.float32), np.zeros(len(Xs), np.float32), np.zeros(len(Xs), np.float32)
    for k in range(len(Xs)):
        tok = tf.constant(np.repeat(Xs[k:k+1], IG_STEPS + 1, 0))
        e = E(Xs[k:k+1])                                  # (1,512,64)
        nm = tf.constant((Xs[k:k+1] != 0)[..., None], tf.float32)
        # zero baseline for real tokens only; padding keeps its embedding
        # (an all-zero input makes LayerNorm gradients blow up in the
        # order-blind model; padding is masked anyway in the PE model)
        with tf.GradientTape() as tape:
            z = alphas * e * nm + e * (1 - nm); tape.watch(z)
            p = f([tok, z], training=False)[:, ys[k]]
        g = tape.gradient(p, z).numpy()
        avg = (g[:-1] + g[1:]).mean(0) / 2                # trapezoidal rule
        a = (avg * e[0].numpy()).sum(-1) * (Xs[k] != 0)  # signed, per real token
        att[k] = a; prob[k] = float(p[-1]); base[k] = float(p[0])
    return att, prob, base

def load(name, seed):
    tf.keras.backend.clear_session()
    pos = name == "MTL_Transformer_PE"
    m = transformer(pos_mask=pos)
    if seed is not None:
        m.load_weights(f"{RUNS}/{name}_lr{BEST_LR[name]:g}_s{seed}/model.weights.h5")
    return m, pos

jobs = [("MTL_Transformer_PE", s) for s in SEEDS] + [("MTL_Transformer", s) for s in SEEDS] \
       + [("MTL_Transformer_PE", None)]                   # None = untrained control
# keep finished models on re-run (also picks up results from an earlier attempt)
_prev = globals().get("res")
res = globals().setdefault("IG_RESULTS", {k: v for k, v in _prev.items()
                           if k.split("__")[0] in ("att", "prob", "base")} if isinstance(_prev, dict) else {})
for name, seed in jobs:
    tag = f"{name}_s{seed}" if seed is not None else f"{name}_untrained"
    if f"att__{tag}" in res:
        print(f"  {tag:28s} already done, skipped"); continue
    t0 = time.time()
    if seed is None: tf.keras.utils.set_random_seed(0)
    m, pos = load(name, seed)
    att, prob, base = integrated_gradients(m, pos)
    res[f"att__{tag}"], res[f"prob__{tag}"] = att, prob
    err = np.abs(att.sum(1) - (prob - base)).max()     # IG completeness check
    res[f"base__{tag}"] = base
    print(f"  {tag:28s} mean p(true)={prob.mean():.3f}  completeness max error={err:.4f}  {time.time()-t0:.0f}s")

np.savez_compressed(f"{OUT}/Cell05_IG_structuredRNA.npz", gene_ids=S["test"].NCBIGeneID.values[sel],
                    tokens=Xs, y_true=ys, gt_classes=le.classes_, **res)
print("Saved", f"{OUT}/Cell05_IG_structuredRNA.npz — please upload this file.")
