"""
train.py -- runs the whole experiment:  build data -> train source model ->
            train target model WITHOUT distillation -> train target model WITH distillation
            -> evaluate all three on the same test set -> save everything for the app.

Run:   python train.py              (one run, saves artifacts/ for the app)
       python train.py --seeds 5    (repeat on 5 random worlds, print mean +/- std)
"""
import argparse, os, time
import numpy as np
from data import make_world
from models import EmbeddingRanker
from metrics import evaluate

# ---- settings (all in one place so they are easy to explain / change) ----
SOURCE_DIM, TARGET_DIM = 16, 4        # embedding sizes: large vs compact
SOURCE_EPOCHS, SOURCE_L2 = 12, 1e-5
PLAIN_EPOCHS, PLAIN_L2 = 30, 1e-3     # baseline settings (tuned on a separate random world)
DISTILL_EPOCHS, DISTILL_L2 = 10, 1e-5
ALPHA = 0.7                           # distillation weight (0 = no distillation, 1 = soft labels only)
EXTRA_UNLABELED = 20                  # extra target-domain pairs WITHOUT real labels, per labeled pair
ART_DIR = "artifacts"


def train_source(w, seed=0):
    m = EmbeddingRanker(w["n_users"], w["n_items"], SOURCE_DIM, seed)
    return m.fit(w["src_u"], w["src_i"], w["src_y"], epochs=SOURCE_EPOCHS, lr=0.02, l2=SOURCE_L2, seed=seed)


def make_soft_labels(source, w, seed=0):
    """Ask the (frozen) source model to predict click probability for target-domain pairs.
    This is the 'zero-shot' step: the source model never trains on target data."""
    r = np.random.default_rng(seed + 123)
    n_extra = EXTRA_UNLABELED * len(w["tgt_u"])
    xu = r.integers(0, w["n_users"], n_extra)
    xi = r.integers(0, w["n_songs"], n_extra)
    soft_lab = source.predict(w["tgt_u"], w["tgt_i"])
    soft_extra = source.predict(xu, xi)
    return soft_lab, (xu, xi, soft_extra)


def train_target(w, soft_lab, extra, alpha, seed=0):
    m = EmbeddingRanker(w["n_users"], w["n_items"], TARGET_DIM, seed)
    if alpha == 0:                                       # plain training, real labels only
        return m.fit(w["tgt_u"], w["tgt_i"], w["tgt_y"], epochs=PLAIN_EPOCHS, l2=PLAIN_L2,
                     batch=256, lr=0.01, seed=seed)
    xu, xi, xs = extra
    u = np.concatenate([w["tgt_u"], xu]); i = np.concatenate([w["tgt_i"], xi])
    # extra pairs have no real label: use their soft label as 'y' too, so blend = soft label
    y = np.concatenate([w["tgt_y"], xs]); soft = np.concatenate([soft_lab, xs])
    return m.fit(u, i, y, soft=soft, alpha=alpha, epochs=DISTILL_EPOCHS, l2=DISTILL_L2,
                 batch=1024, lr=0.02, seed=seed)


def run(seed=0, verbose=True):
    w = make_world(seed)
    t0 = time.time()
    source = train_source(w, seed)
    soft_lab, extra = make_soft_labels(source, w, seed)
    plain = train_target(w, soft_lab, extra, 0.0, seed)
    distilled = train_target(w, soft_lab, extra, ALPHA, seed)
    res = {"Target model (no distillation)": evaluate(plain.predict, w),
           "Target model (distilled)": evaluate(distilled.predict, w),
           "Source model (large, reference)": evaluate(source.predict, w)}
    if verbose:
        print(f"source interactions: {len(w['src_y']):,} | target train: {len(w['tgt_y']):,} "
              f"| ratio {len(w['src_y'])/len(w['tgt_y']):.0f}x | positives in target train: {int(w['tgt_y'].sum())}")
        print(f"params: source {source.n_params():,} | target {plain.n_params():,}")
        print(f"trained in {time.time()-t0:.1f}s")
    return w, source, plain, distilled, soft_lab, extra, res


def save(w, source, plain, distilled, soft_lab, extra, res):
    os.makedirs(ART_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(ART_DIR, "world.npz"), **{k: v for k, v in w.items()})
    np.savez_compressed(os.path.join(ART_DIR, "models.npz"),
        **{f"{n}_{a}": getattr(m, a) for n, m in [("source", source), ("plain", plain), ("distilled", distilled)]
           for a in ["P", "Q", "bu", "bi"]},
        **{f"{n}_b0": np.array(m.b0) for n, m in [("source", source), ("plain", plain), ("distilled", distilled)]},
        soft_lab=soft_lab, xu=extra[0], xi=extra[1], xs=extra[2])
    import json
    json.dump(res, open(os.path.join(ART_DIR, "metrics.json"), "w"), indent=2)


def print_table(res):
    cols = list(next(iter(res.values())).keys())
    print(f"{'Model':36s}" + "".join(f"{c:>13s}" for c in cols))
    for name, r in res.items():
        print(f"{name:36s}" + "".join(f"{r[c]:13.3f}" for c in cols))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    out = run(0)
    save(*out); print_table(out[-1])
    if a.seeds > 1:
        allres = [out[-1]] + [run(s, verbose=False)[-1] for s in range(1, a.seeds)]
        print(f"\nMean +/- std over {a.seeds} random worlds:")
        cols = list(allres[0][next(iter(allres[0]))].keys())
        print(f"{'Model':36s}" + "".join(f"{c:>15s}" for c in cols))
        for name in allres[0]:
            row = "".join(f"{np.mean([r[name][c] for r in allres]):9.3f}±{np.std([r[name][c] for r in allres]):.3f}" for c in cols)
            print(f"{name:36s}{row}")
    print(f"\nSaved artifacts to ./{ART_DIR}/  ->  now run: streamlit run app.py")


def load_artifacts():
    """Used by the app: rebuild the world and the three models from the saved files."""
    if not os.path.exists(os.path.join(ART_DIR, "models.npz")):
        save(*run(0))                                    # first launch: train automatically (~5 s)
    w = dict(np.load(os.path.join(ART_DIR, "world.npz"), allow_pickle=True))
    z = np.load(os.path.join(ART_DIR, "models.npz"))
    def build(name, dim):
        m = EmbeddingRanker(w["n_users"].item(), w["n_items"].item(), dim)
        m.P, m.Q, m.bu, m.bi, m.b0 = z[f"{name}_P"], z[f"{name}_Q"], z[f"{name}_bu"], z[f"{name}_bi"], float(z[f"{name}_b0"])
        return m
    models = {"Target model (no distillation)": build("plain", TARGET_DIM),
              "Target model (distilled)": build("distilled", TARGET_DIM),
              "Source model (large, reference)": build("source", SOURCE_DIM)}
    return w, models, z["soft_lab"], (z["xu"], z["xi"], z["xs"])
