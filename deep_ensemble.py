"""
Large-scale model search for Australian Powerball (7 from 35 + PB 1 from 20).

Built to use a lot of hardware (e.g. DGX Spark: GPU + 20 ARM cores) to answer:
"Does any model, however large, predict the next draw better than chance?"
If one does, its probabilities are passed to top_predictions.py.

Model families (each with a hyper-parameter search):
  freq         decayed ball frequencies (half-life x prior grid)
  markov       previous-draw -> next-draw transition model
  logreg       logistic regression on per-ball history features
  gbm          gradient boosted trees on per-ball history features
  gru          GRU deep ensemble on the raw draw sequence      (GPU)
  transformer  Transformer deep ensemble on the raw draw sequence (GPU)

Protocol (no look-ahead anywhere):
  draws [0, warmup)               history only, never scored
  draws [warmup, T - holdout)     VALIDATION: walk-forward predictions used to
                                  pick the best configs and fit stacking weights
  draws [T - holdout, T)          HOLDOUT: walk-forward predictions, never used
                                  for any choice; this is the honest score
Every prediction for draw s uses a model trained only on draws < s.

Score: log-likelihood gain over the uniform model, in nats per draw
(0 = exactly as good as chance, > 0 = better).

--null-runs K repeats the full pipeline (same search, same selection) on K
synthetic fair-draw datasets. The real holdout gain is then ranked against
what pure chance produces after the same amount of searching: this is the
correct p-value when thousands of models are tried.

The final model probabilities are used only if the holdout gain is positive
and significant (or --force-model is given); otherwise uniform is written.

Usage:
    python deep_ensemble.py --quick                         # smoke test (minutes)
    python deep_ensemble.py                                 # full search
    python deep_ensemble.py --configs 64 --seeds 8 --null-runs 20 --gpu-workers 6
    python top_predictions.py --probs model_probs_draw_1585.json --tickets 20
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from top_predictions import DATA_FILE, N_MAIN, N_PB, PICK, load_draws

EPS = 1e-5
U_MAIN_LL = PICK * math.log(PICK / N_MAIN) + (N_MAIN - PICK) * math.log(1 - PICK / N_MAIN)
U_PB_LL = math.log(1 / N_PB)
FEATURE_START = 20  # first draw used as a training row for tabular models
CPU_FAMILIES = ("freq", "markov", "logreg", "gbm")
GPU_FAMILIES = ("gru", "transformer")


# --------------------------------------------------------------------------- utils
def onehots(main, pb):
    t = len(main)
    y = np.zeros((t, N_MAIN), np.float32)
    y[np.arange(t)[:, None], main.astype(int) - 1] = 1
    p = np.zeros((t, N_PB), np.float32)
    p[np.arange(t), pb.astype(int) - 1] = 1
    return y, p


def normalize_main(p):
    """Per-ball inclusion probabilities that sum to 7 and lie in (0, 1)."""
    p = np.clip(np.asarray(p, np.float64), EPS, None)
    for _ in range(20):
        p = p * PICK / p.sum(-1, keepdims=True)
        p = np.clip(p, EPS, 1 - EPS)
    return p


def normalize_pb(q):
    q = np.clip(np.asarray(q, np.float64), EPS, None)
    return q / q.sum(-1, keepdims=True)


def main_ll(p, y):
    """Per-draw log-likelihood of the drawn set under per-ball marginals."""
    return (y * np.log(p) + (1 - y) * np.log(1 - p)).sum(-1)


def pb_ll(q, p):
    return (p * np.log(q)).sum(-1)


def walk_forward(fit, predict, eval_idx, refit):
    """Refit every `refit` draws on all earlier draws, predict the next chunk."""
    pm, qp = [], []
    for start in range(0, len(eval_idx), refit):
        chunk = eval_idx[start:start + refit]
        model = fit(int(chunk[0]))
        a, b = predict(model, chunk)
        pm.append(a)
        qp.append(b)
    return np.concatenate(pm), np.concatenate(qp)


# --------------------------------------------------------------------------- closed-form families
def freq_all(spec, y, p):
    """Exponentially decayed frequency with a uniform Dirichlet prior.
    Row s predicts draw s from draws < s (row T = next, unseen draw)."""
    h, a = spec["halflife"], spec["prior"]
    d = 1.0 if h is None else 0.5 ** (1 / h)
    t = len(y)
    cm, cp, w = np.zeros(N_MAIN), np.zeros(N_PB), 0.0
    pm, qp = np.empty((t + 1, N_MAIN)), np.empty((t + 1, N_PB))
    for s in range(t + 1):
        pm[s] = (cm + a * PICK / N_MAIN) / (w + a)
        qp[s] = (cp + a / N_PB) / (w + a)
        if s < t:
            cm, cp, w = d * cm + y[s], d * cp + p[s], d * w + 1
    return pm, qp


def markov_all(spec, y, p):
    """P(ball i next | balls in previous draw), smoothed toward uniform."""
    a = spec["prior"]
    t = len(y)
    m, nm = np.zeros((N_MAIN, N_MAIN)), np.zeros(N_MAIN)
    q, nq = np.zeros((N_PB, N_PB)), np.zeros(N_PB)
    pm = np.full((t + 1, N_MAIN), PICK / N_MAIN)
    qp = np.full((t + 1, N_PB), 1 / N_PB)
    for s in range(1, t + 1):
        prev = y[s - 1] > 0
        pm[s] = ((m[prev] + a * PICK / N_MAIN) / (nm[prev][:, None] + a)).mean(0)
        j = int(p[s - 1].argmax())
        qp[s] = (q[j] + a / N_PB) / (nq[j] + a)
        if s < t:  # add transition (s-1 -> s) after predicting s
            m[prev] += y[s]
            nm[prev] += 1
            q[j] += p[s]
            nq[j] += 1
    return pm, qp


# --------------------------------------------------------------------------- tabular families
def ball_features(x):
    """Per-ball history features. Row s (0..T) uses only rows < s of x."""
    t, k = x.shape
    c = np.vstack([np.zeros((1, k)), np.cumsum(x, 0)])
    idx = np.arange(t + 1)
    feats = []
    for win in (1, 3, 5, 10, 20, 50, 100):
        lo = np.maximum(idx - win, 0)
        feats.append((c - c[lo]) / np.maximum(idx - lo, 1)[:, None])
    feats.append(c / np.maximum(idx, 1)[:, None])
    for h in (5, 20, 60):
        d = 0.5 ** (1 / h)
        e = np.zeros((t + 1, k))
        for s in range(1, t + 1):
            e[s] = d * e[s - 1] + (1 - d) * x[s - 1]
        feats.append(e)
    gap = np.full((t + 1, k), 100.0)
    last = np.full(k, -1)
    for s in range(1, t + 1):
        last = np.where(x[s - 1] > 0, s - 1, last)
        gap[s] = np.where(last >= 0, np.minimum(s - 1 - last, 100), 100)
    feats.append(gap)
    feats.append(np.broadcast_to(np.arange(k, dtype=float), (t + 1, k)))
    return np.stack(feats, -1).astype(np.float32)  # (T+1, K, nf)


def make_classifier(family, spec):
    if family == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=spec["max_iter"], learning_rate=spec["lr"],
            max_leaf_nodes=spec["leaves"], l2_regularization=spec["l2"],
            min_samples_leaf=spec["min_leaf"], early_stopping=False,
            random_state=0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=spec["C"], max_iter=2000))


def tabular_walk(family, spec, y, p, eval_idx, refit, want_final):
    fm_, fp_ = ball_features(y), ball_features(p)
    nf = fm_.shape[-1]

    def fit(t):
        rows = slice(FEATURE_START, t)
        cm = make_classifier(family, spec).fit(fm_[rows].reshape(-1, nf), y[rows].ravel())
        cp = make_classifier(family, spec).fit(fp_[rows].reshape(-1, nf), p[rows].ravel())
        return cm, cp

    def predict(models, ss):
        cm, cp = models
        a = cm.predict_proba(fm_[ss].reshape(-1, nf))[:, 1].reshape(len(ss), N_MAIN)
        b = cp.predict_proba(fp_[ss].reshape(-1, nf))[:, 1].reshape(len(ss), N_PB)
        return a, b

    pm, qp = walk_forward(fit, predict, eval_idx, refit)
    final = predict(fit(len(y)), np.array([len(y)])) if want_final else None
    return pm, qp, final


# --------------------------------------------------------------------------- sequence families (torch)
def build_net(family, spec):
    import torch
    from torch import nn

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            d, L, drop = spec["d"], spec["L"], spec["dropout"]
            self.inp = nn.Linear(N_MAIN + N_PB, d)
            if family == "gru":
                self.core = nn.GRU(d, d, spec["layers"], batch_first=True,
                                   dropout=drop if spec["layers"] > 1 else 0.0)
            else:
                self.pos = nn.Parameter(torch.zeros(1, L, d))
                layer = nn.TransformerEncoderLayer(d, spec["heads"], 4 * d, drop,
                                                   batch_first=True, norm_first=True)
                self.core = nn.TransformerEncoder(layer, spec["layers"])
            self.drop = nn.Dropout(drop)
            self.head_m = nn.Linear(d, N_MAIN)
            self.head_p = nn.Linear(d, N_PB)
            # Start exactly at the uniform prediction
            nn.init.zeros_(self.head_m.weight)
            nn.init.constant_(self.head_m.bias, math.log(PICK / (N_MAIN - PICK)))
            nn.init.zeros_(self.head_p.weight)
            nn.init.zeros_(self.head_p.bias)

        def forward(self, x):
            h = self.inp(x)
            if family == "gru":
                h, _ = self.core(h)
            else:
                h = self.core(h + self.pos)
            h = self.drop(h[:, -1])
            return self.head_m(h), self.head_p(h)

    return SeqNet()


def train_seq_member(family, spec, wins, ym, pt, train_s, device, seed):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    L = spec["L"]
    nv = max(8, int(len(train_s) * 0.15))
    s_tr = torch.as_tensor(train_s[:-nv], device=device)
    s_va = torch.as_tensor(train_s[-nv:], device=device)
    net = build_net(family, spec).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=spec["lr"], weight_decay=spec["wd"])

    def loss_of(s):
        lm, lp = net(wins[s - L])
        return (F.binary_cross_entropy_with_logits(lm, ym[s], reduction="none").sum(-1).mean()
                + F.cross_entropy(lp, pt[s]))

    best, best_state, bad = float("inf"), None, 0
    for ep in range(spec["epochs"]):
        net.train()
        perm = s_tr[torch.randperm(len(s_tr), device=device)]
        for b in range(0, len(perm), spec["batch"]):
            loss = loss_of(perm[b:b + spec["batch"]])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == spec["epochs"] - 1:
            net.eval()
            with torch.no_grad():
                v = float(loss_of(s_va))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= spec["patience"]:
                    break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def seq_walk(family, spec, y, p, eval_idx, refit, want_final, n_seeds, seed_base):
    import torch
    from numpy.lib.stride_tricks import sliding_window_view

    device = "cuda" if torch.cuda.is_available() else "cpu"
    L = spec["L"]
    z = np.concatenate([y, p], 1)
    # wins[j] = draws j .. j+L-1, used to predict draw s = j + L
    wins = np.ascontiguousarray(sliding_window_view(z, L, axis=0).transpose(0, 2, 1))
    wins = torch.as_tensor(wins, device=device)
    ym = torch.as_tensor(y, device=device)
    pt = torch.as_tensor(p.argmax(1), device=device, dtype=torch.long)

    def fit(t):
        train_s = np.arange(L, t)
        return [train_seq_member(family, spec, wins, ym, pt, train_s, device,
                                 seed_base + 7919 * t + k) for k in range(n_seeds)]

    def predict(nets, ss):
        j = torch.as_tensor(np.asarray(ss) - L, device=device)
        a = b = 0
        with torch.no_grad():
            for net in nets:
                lm, lp = net(wins[j])
                a = a + torch.sigmoid(lm)
                b = b + torch.softmax(lp, -1)
        return (a / len(nets)).cpu().numpy(), (b / len(nets)).cpu().numpy()

    pm, qp = walk_forward(fit, predict, eval_idx, refit)
    final = predict(fit(len(y)), np.array([len(y)])) if want_final else None
    return pm, qp, final


# --------------------------------------------------------------------------- task runner
def _worker_init(threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        import torch
        torch.set_num_threads(threads)
        torch.set_float32_matmul_precision("high")
    except ImportError:
        pass


def run_task(task):
    fam, name, spec = task["family"], task["name"], task["spec"]
    y, p = onehots(task["main"], task["pb"])
    eval_idx, refit, want_final = task["eval_idx"], task["refit"], task["want_final"]
    t0 = time.time()
    if fam in ("freq", "markov"):
        pm_all, qp_all = (freq_all if fam == "freq" else markov_all)(spec, y, p)
        pm, qp = pm_all[eval_idx], qp_all[eval_idx]
        final = (pm_all[len(y):], qp_all[len(y):])
    elif fam in ("logreg", "gbm"):
        pm, qp, final = tabular_walk(fam, spec, y, p, eval_idx, refit, want_final)
    else:
        pm, qp, final = seq_walk(fam, spec, y, p, eval_idx, refit, want_final,
                                 task["seeds"], task["seed_base"])
    out = dict(family=fam, name=name, spec=spec, pm=normalize_main(pm),
               qp=normalize_pb(qp), secs=time.time() - t0)
    if want_final and final is not None:
        out["fm"], out["fq"] = normalize_main(final[0])[0], normalize_pb(final[1])[0]
    return out


def build_specs(args):
    rng = np.random.default_rng(12345)  # fixed: identical search for real and null runs
    specs = []
    fams = set(args.families)
    if "freq" in fams:
        for h in (5, 10, 20, 50, 100, 200, None):
            for a in (1, 7, 35, 100, 500):
                specs.append(("freq", f"freq_h{h}_a{a}", dict(halflife=h, prior=a)))
    if "markov" in fams:
        for a in (1, 10, 50, 200, 1000):
            specs.append(("markov", f"markov_a{a}", dict(prior=a)))
    if "logreg" in fams:
        for c in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            specs.append(("logreg", f"logreg_C{c}", dict(C=c)))
    for fam in ("gbm", "gru", "transformer"):
        if fam not in fams:
            continue
        for i in range(args.configs):
            if fam == "gbm":
                s = dict(max_iter=int(rng.choice([50, 100, 200, 400])),
                         lr=float(rng.choice([0.01, 0.03, 0.1])),
                         leaves=int(rng.choice([4, 8, 16, 31])),
                         l2=float(rng.choice([0.0, 1.0, 10.0])),
                         min_leaf=int(rng.choice([20, 50, 200, 500])))
            else:
                s = dict(L=int(rng.choice([8, 16, 32])),
                         d=int(rng.choice([32, 64, 128])),
                         layers=int(rng.choice([1, 2, 3])),
                         heads=4,
                         dropout=float(rng.choice([0.1, 0.3, 0.5])),
                         lr=float(rng.choice([3e-4, 1e-3, 3e-3])),
                         wd=float(rng.choice([1e-4, 1e-2, 1e-1])),
                         epochs=int(rng.choice([100, 200, 400])),
                         batch=int(rng.choice([32, 64, 512])),
                         patience=6)
                if args.quick:
                    s["epochs"] = 10
            specs.append((fam, f"{fam}_{i:03d}", s))
    return specs


def run_pipeline(main, pb, specs, args, want_final, cpu_pool, gpu_pool):
    t = len(main)
    eval_idx = np.arange(args.warmup, t)
    base = dict(main=main, pb=pb, eval_idx=eval_idx, refit=args.refit,
                want_final=want_final, seeds=args.seeds)
    cpu_tasks, gpu_tasks = [], []
    for i, (fam, name, spec) in enumerate(specs):
        task = dict(base, family=fam, name=name, spec=spec, seed_base=1000 * i)
        (gpu_tasks if fam in GPU_FAMILIES else cpu_tasks).append(task)
    jobs = []
    if cpu_tasks:
        jobs.append(cpu_pool.map_async(run_task, cpu_tasks, chunksize=1))
    if gpu_tasks:
        jobs.append(gpu_pool.map_async(run_task, gpu_tasks, chunksize=1))
    results = [r for j in jobs for r in j.get()]
    return sorted(results, key=lambda r: r["name"])


# --------------------------------------------------------------------------- selection / stacking
def em_weights(comps, target, binary, iters=500):
    """Mixture weights maximising validation log-likelihood (EM, monotone)."""
    c = np.stack(comps)
    like = np.where(target[None] > 0, c, 1 - c) if binary else (c * target[None]).sum(-1, keepdims=True)
    a = np.full(len(comps), 1 / len(comps))
    for _ in range(iters):
        num = a[:, None, None] * like
        a = (num / num.sum(0, keepdims=True)).mean(axis=(1, 2))
    return a


def analyse(results, main, pb, args):
    y, p = onehots(main, pb)
    eval_idx = np.arange(args.warmup, len(main))
    n_val = len(eval_idx) - args.holdout
    val, hold = slice(0, n_val), slice(n_val, None)
    out = {}
    for part, key, fkey, ll, unif, tgt, uval in (
            ("main", "pm", "fm", main_ll, U_MAIN_LL, y[eval_idx], PICK / N_MAIN),
            ("pb", "qp", "fq", pb_ll, U_PB_LL, p[eval_idx], 1 / N_PB)):
        gain_val = {r["name"]: float((ll(r[key][val], tgt[val]) - unif).mean()) for r in results}
        gain_hold = {r["name"]: float((ll(r[key][hold], tgt[hold]) - unif).mean()) for r in results}
        chosen = []
        for fam in sorted({r["family"] for r in results}):
            rs = sorted((r for r in results if r["family"] == fam), key=lambda r: -gain_val[r["name"]])
            chosen += rs[:args.top_k]
        uniform = np.full_like(results[0][key], uval)
        comps = [uniform] + [r[key] for r in chosen]
        w = em_weights([c[val] for c in comps], tgt[val], binary=(part == "main"))
        mix = sum(wi * c for wi, c in zip(w, comps))
        g = ll(mix[hold], tgt[hold]) - unif
        se = g.std(ddof=1) / math.sqrt(len(g))
        z = g.mean() / se if se > 0 else 0.0
        best = max(results, key=lambda r: gain_val[r["name"]])
        fam_best = {}
        for r in results:
            f = r["family"]
            if f not in fam_best or gain_val[r["name"]] > gain_val[fam_best[f]]:
                fam_best[f] = r["name"]
        final = None
        if all(fkey in r for r in chosen):
            final = sum(wi * c for wi, c in zip(w, [np.full(len(uniform[0]), uval)]
                                                + [r[fkey] for r in chosen]))
        out[part] = dict(
            holdout_gain=float(g.mean()), holdout_se=float(se), z=float(z),
            p_ttest=float(0.5 * math.erfc(z / math.sqrt(2))),
            weights={"uniform": float(w[0]),
                     **{r["name"]: float(wi) for r, wi in zip(chosen, w[1:])}},
            best_single=dict(name=best["name"], val_gain=gain_val[best["name"]],
                             holdout_gain=gain_hold[best["name"]]),
            family_best={f: dict(name=n, val_gain=gain_val[n], holdout_gain=gain_hold[n])
                         for f, n in fam_best.items()},
            final=None if final is None else final.tolist())
    return out


def print_report(rep, n_val, n_hold):
    for part in ("main", "pb"):
        r = rep[part]
        title = "MAIN BALLS (7 of 35)" if part == "main" else "POWERBALL (1 of 20)"
        print(f"\n--- {title}: gain vs uniform, nats per draw "
              f"(validation {n_val} draws, holdout {n_hold} draws) ---")
        print(f"{'family':<13}{'best config':<22}{'validation':>12}{'holdout':>12}")
        for f, d in sorted(r["family_best"].items()):
            print(f"{f:<13}{d['name']:<22}{d['val_gain']:>12.4f}{d['holdout_gain']:>12.4f}")
        top = sorted(r["weights"].items(), key=lambda kv: -kv[1])[:6]
        print("Stacking weights: " + ", ".join(f"{k}={v:.2f}" for k, v in top))
        print(f"STACKED ENSEMBLE holdout gain = {r['holdout_gain']:+.4f} "
              f"+/- {r['holdout_se']:.4f}  (z = {r['z']:.2f}, one-sided p = {r['p_ttest']:.3f})")
        if "p_null" in r:
            print(f"Null calibration: p = {r['p_null']:.3f} "
                  f"(real gain vs {r['n_null']} fair-draw re-runs of the whole search)")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_FILE)
    ap.add_argument("--families", nargs="+", default=list(CPU_FAMILIES + GPU_FAMILIES),
                    choices=list(CPU_FAMILIES + GPU_FAMILIES))
    ap.add_argument("--configs", type=int, default=24,
                    help="random configs per searched family (gbm, gru, transformer)")
    ap.add_argument("--seeds", type=int, default=5, help="deep-ensemble members per net config")
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--refit", type=int, default=10, help="retrain every N draws")
    ap.add_argument("--top-k", type=int, default=3, help="configs per family entering the stack")
    ap.add_argument("--null-runs", type=int, default=0,
                    help="repeat the whole search on N synthetic fair datasets")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--force-model", action="store_true",
                    help="write model probabilities even if the holdout test fails")
    ap.add_argument("--cpu-workers", type=int, default=None)
    ap.add_argument("--gpu-workers", type=int, default=4,
                    help="processes sharing the GPU for the neural families")
    ap.add_argument("--quick", action="store_true", help="tiny smoke-test settings")
    args = ap.parse_args()

    if args.quick:
        args.configs, args.seeds, args.refit, args.null_runs = 2, 1, 50, 0
    try:
        import torch
        print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"
              + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    except ImportError:
        print("torch not installed: skipping gru/transformer families.")
        args.families = [f for f in args.families if f not in GPU_FAMILIES]
    try:
        import sklearn  # noqa: F401
    except ImportError:
        print("scikit-learn not installed: skipping logreg/gbm families.")
        args.families = [f for f in args.families if f not in ("logreg", "gbm")]

    draw_no, dates, main_d, pb_d = load_draws(args.data)
    t = len(main_d)
    n_val = t - args.warmup - args.holdout
    if n_val < 30:
        raise SystemExit(f"Not enough draws ({t}) for warmup {args.warmup} + holdout {args.holdout}.")
    specs = build_specs(args)
    n_gpu = sum(f in GPU_FAMILIES for f, _, _ in specs)
    cpu_workers = args.cpu_workers or max(1, (os.cpu_count() or 2) - (args.gpu_workers if n_gpu else 0))
    print(f"{t} draws (#{draw_no[0]}..#{draw_no[-1]}): warmup {args.warmup}, "
          f"validation {n_val}, holdout {args.holdout}, refit every {args.refit}")
    print(f"{len(specs)} model configs ({n_gpu} neural x {args.seeds} seeds), "
          f"{cpu_workers} CPU workers, {args.gpu_workers if n_gpu else 0} GPU workers, "
          f"{args.null_runs} null runs")

    ctx = mp.get_context("spawn")  # CUDA-safe
    cpu_pool = ctx.Pool(cpu_workers, initializer=_worker_init, initargs=(1,))
    gpu_pool = ctx.Pool(args.gpu_workers, initializer=_worker_init, initargs=(2,)) if n_gpu else None
    try:
        t0 = time.time()
        results = run_pipeline(main_d, pb_d, specs, args, True, cpu_pool, gpu_pool)
        print(f"Real-data search finished in {(time.time() - t0) / 60:.1f} min.")
        rep = analyse(results, main_d, pb_d, args)

        null_gains = {"main": [], "pb": []}
        rng = np.random.default_rng(2024)
        for k in range(args.null_runs):
            t1 = time.time()
            syn_main = np.sort(np.argsort(rng.random((t, N_MAIN)), 1)[:, :PICK] + 1, 1).astype(np.int8)
            syn_pb = rng.integers(1, N_PB + 1, t).astype(np.int8)
            nres = run_pipeline(syn_main, syn_pb, specs, args, False, cpu_pool, gpu_pool)
            nrep = analyse(nres, syn_main, syn_pb, args)
            for part in null_gains:
                null_gains[part].append(nrep[part]["holdout_gain"])
            print(f"null run {k + 1}/{args.null_runs}: main {nrep['main']['holdout_gain']:+.4f}, "
                  f"pb {nrep['pb']['holdout_gain']:+.4f}  ({(time.time() - t1) / 60:.1f} min)")
    finally:
        cpu_pool.close()
        if gpu_pool:
            gpu_pool.close()

    for part, g in null_gains.items():
        if g:
            rep[part]["p_null"] = (1 + sum(x >= rep[part]["holdout_gain"] for x in g)) / (1 + len(g))
            rep[part]["n_null"] = len(g)
            rep[part]["null_gains"] = g
    print_report(rep, n_val, args.holdout)

    probs = {"draw": int(draw_no[-1] + 1), "based_on_draw": int(draw_no[-1])}
    print()
    for part, uval, n in (("main", PICK / N_MAIN, N_MAIN), ("pb", 1 / N_PB, N_PB)):
        r = rep[part]
        p_sig = r.get("p_null", r["p_ttest"])
        accepted = r["holdout_gain"] > 0 and p_sig < args.alpha
        use = (accepted or args.force_model) and r["final"] is not None
        probs[part] = r["final"] if use else [uval] * n
        probs[f"{part}_source"] = "model" if use else "uniform"
        verdict = "ACCEPTED" if accepted else "REJECTED (not better than chance)"
        print(f"{part}: {verdict}; writing {'model' if use else 'uniform'} probabilities"
              + (" (forced)" if use and not accepted else ""))

    out_probs = f"model_probs_draw_{probs['draw']}.json"
    out_rep = f"ensemble_report_draw_{probs['draw']}.json"
    with open(out_probs, "w") as f:
        json.dump(probs, f, indent=1)
    with open(out_rep, "w") as f:
        json.dump(dict(args=vars(args), report=rep,
                       timings={r["name"]: r["secs"] for r in results}), f, indent=1)
    print(f"Saved {out_probs} and {out_rep}")
    print(f"Next: python top_predictions.py --probs {out_probs} --tickets 20")


if __name__ == "__main__":
    main()
