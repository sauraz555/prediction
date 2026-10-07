"""
Top-N ticket generator for Australian Powerball (7 from 35 + Powerball 1 from 20).

Nothing can raise the chance that a given combination is drawn: if the draw is
fair, every combination has probability 1 in 134,490,400. What a model *can*
legitimately optimise is everything around that:

  1. Bias detection   - test whether any ball is drawn more often than chance
                        (chi-square with Monte Carlo p-value). The measured
                        bias is shrunk toward uniform (James-Stein style), so
                        if the data looks fair, the model treats it as fair.
  2. Prize sharing    - the jackpot is split between all winners. Combinations
                        that many people play (birthdays, sequences, last
                        week's numbers, "hot" numbers) pay less when they win.
                        Ranking by low popularity maximises expected payout.
  3. Coverage         - a set of tickets that overlap little covers more
                        number patterns, which maximises the chance that at
                        least one ticket wins *some* division.
  4. Honest backtest  - walk-forward test against the exact hypergeometric odds,
                        so you can see the model does not beat chance on hits.

All 6,724,520 main combinations are enumerated and scored exactly (no sampling).

Usage:
    python top_predictions.py                       # 20 tickets
    python top_predictions.py --tickets 50 --max-overlap 2
    python top_predictions.py --backtest 200        # walk-forward check
    python top_predictions.py --window 150          # bias test on recent draws only
"""
import argparse
import itertools
import math
import multiprocessing
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

N_MAIN, PICK, N_PB = 35, 7, 20
FIRST_7_35_DRAW = 1144  # 19 April 2018: game changed to 7/35 + PB 20
DATA_FILE = "powerball_15_years_results.csv"
CACHE_FILE = "combos_7of35.npy"

# Prize divisions: (main matches, powerball matched)
DIVISIONS = [(7, True), (7, False), (6, True), (6, False), (5, True),
             (4, True), (5, False), (3, True), (2, True)]

# Heuristic popularity weights, in log "co-winner multiplier" units.
# Based on published studies of player choice (birthdays, patterns, recency).
POP_WEIGHTS = {
    "birthday": 0.10,      # per number <= 31
    "month": 0.10,         # extra per number <= 12
    "consecutive": 0.15,   # per adjacent pair (e.g. 14, 15)
    "progression": 2.00,   # all gaps equal (1-2-3..., 5-10-15...)
    "few_gaps": 0.40,      # only 2 distinct gap sizes (grid / diagonal patterns)
    "last_draw": 0.25,     # per number repeated from the latest draw
    "hot": 0.10,           # per number from the published "hot" list
    "lucky": 0.05,         # per multiple of 7
    "same_digit": 0.30,    # 3+ numbers sharing the last digit
}
PB_POP = {"low": 0.10, "lucky": 0.10, "last_draw": 0.30}


# --------------------------------------------------------------------------- data
def load_draws(path=DATA_FILE):
    """Return (draw_numbers, dates, main[N,7], pb[N]) for the current 7/35 format,
    oldest first."""
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        try:
            nums = sorted(int(x) for x in str(r["Winning Numbers"]).split(","))
            pb = int(r["Powerball"])
        except ValueError:
            continue
        if (int(r["Draw Number"]) >= FIRST_7_35_DRAW and len(nums) == PICK
                and max(nums) <= N_MAIN and 1 <= pb <= N_PB):
            rows.append((int(r["Draw Number"]), r["Date"], nums, pb))
    rows.sort(key=lambda x: x[0])
    draw_no = np.array([r[0] for r in rows])
    gaps = np.diff(draw_no)
    if (gaps != 1).any():
        print(f"WARNING: {(gaps != 1).sum()} gap(s) in draw numbers - data incomplete.")
    return (draw_no, [r[1] for r in rows],
            np.array([r[2] for r in rows], dtype=np.int8),
            np.array([r[3] for r in rows], dtype=np.int8))


def to_mask(nums):
    m = 0
    for n in nums:
        m |= 1 << (int(n) - 1)
    return np.uint64(m)


def all_combos():
    """All C(35,7) combinations as int8 [M,7], cached on disk."""
    if os.path.exists(CACHE_FILE):
        return np.load(CACHE_FILE)
    m = math.comb(N_MAIN, PICK)
    flat = np.fromiter(itertools.chain.from_iterable(
        itertools.combinations(range(1, N_MAIN + 1), PICK)), dtype=np.int8, count=m * PICK)
    combos = flat.reshape(m, PICK)
    np.save(CACHE_FILE, combos)
    return combos


def combo_masks(combos):
    masks = np.zeros(len(combos), dtype=np.uint64)
    for j in range(PICK):
        masks |= np.left_shift(np.uint64(1), (combos[:, j] - 1).astype(np.uint64))
    return masks


def popcount(x):
    return np.bitwise_count(x).astype(np.int8)


# --------------------------------------------------------------------------- bias
def chi2_stat(counts, expected):
    return float(((counts - expected) ** 2 / expected).sum())


def fit_bias(main, pb):
    """Shrunk per-ball draw probabilities.

    Under a fair draw E[X^2] is exactly 35*(1-7/35)=28 for main balls (sampling
    without replacement) and 19 for the Powerball, so the shrinkage factor
    lambda = max(0, 1 - E0/X^2) is ~0 unless the data shows real excess variance.
    """
    n = len(main)
    mc = np.bincount(main.ravel(), minlength=N_MAIN + 1)[1:].astype(float)
    pc = np.bincount(pb, minlength=N_PB + 1)[1:].astype(float)
    p0 = PICK / N_MAIN
    x2_m = chi2_stat(mc, n * p0)
    x2_p = chi2_stat(pc, n / N_PB)
    lam_m = max(0.0, 1 - N_MAIN * (1 - p0) / x2_m) if x2_m > 0 else 0.0
    lam_p = max(0.0, 1 - (N_PB - 1) / x2_p) if x2_p > 0 else 0.0
    w = 1 / N_MAIN + lam_m * (mc / mc.sum() - 1 / N_MAIN)
    v = 1 / N_PB + lam_p * (pc / pc.sum() - 1 / N_PB)
    return dict(n=n, main_counts=mc, pb_counts=pc, x2_main=x2_m, x2_pb=x2_p,
                lam_main=lam_m, lam_pb=lam_p, w=w / w.sum(), v=v / v.sum())


def mc_pvalues(n, x2_m, x2_p, reps=20000, seed=0):
    """Monte Carlo p-values for the chi-square stats under a perfectly fair draw."""
    rng = np.random.default_rng(seed)
    p0 = PICK / N_MAIN
    ge_m = ge_p = 0
    for start in range(0, reps, 1000):
        k = min(1000, reps - start)
        keys = rng.random((k, n, N_MAIN))
        picks = np.argpartition(keys, PICK, axis=2)[:, :, :PICK]
        flat = (picks.reshape(k, -1) + N_MAIN * np.arange(k)[:, None]).ravel()
        mc = np.bincount(flat, minlength=k * N_MAIN).reshape(k, N_MAIN)
        ge_m += (((mc - n * p0) ** 2 / (n * p0)).sum(1) >= x2_m).sum()
        pc = np.stack([np.bincount(r, minlength=N_PB) for r in rng.integers(0, N_PB, (k, n))])
        ge_p += (((pc - n / N_PB) ** 2 / (n / N_PB)).sum(1) >= x2_p).sum()
    return (ge_m + 1) / (reps + 1), (ge_p + 1) / (reps + 1)


# --------------------------------------------------------------------------- scoring
class Scorer:
    def __init__(self, combos):
        self.combos = combos
        self.masks = combo_masks(combos)
        c = combos.astype(np.int16)
        gaps = np.diff(c, axis=1)
        n_distinct_gaps = np.ones(len(c), dtype=np.int8)
        sg = np.sort(gaps, axis=1)
        n_distinct_gaps += (np.diff(sg, axis=1) != 0).sum(1).astype(np.int8)
        last_digit = c % 10
        digit_counts = np.stack([(last_digit == d).sum(1) for d in range(10)], 1)
        # Static (history-independent) popularity part
        self.static_pop = (
            POP_WEIGHTS["birthday"] * (c <= 31).sum(1)
            + POP_WEIGHTS["month"] * (c <= 12).sum(1)
            + POP_WEIGHTS["consecutive"] * (gaps == 1).sum(1)
            + POP_WEIGHTS["progression"] * (n_distinct_gaps == 1)
            + POP_WEIGHTS["few_gaps"] * (n_distinct_gaps == 2)
            + POP_WEIGHTS["lucky"] * (c % 7 == 0).sum(1)
            + POP_WEIGHTS["same_digit"] * (digit_counts.max(1) >= 3)
        ).astype(np.float32)

    def score(self, main_hist, bias, hot_window=20):
        """Return (score, popularity, log-prob ratio) for every combination."""
        last_mask = to_mask(main_hist[-1])
        recent = np.bincount(main_hist[-hot_window:].ravel(), minlength=N_MAIN + 1)[1:]
        hot_mask = to_mask(np.argsort(-recent, kind="stable")[:PICK] + 1)
        pop = (self.static_pop
               + POP_WEIGHTS["last_draw"] * popcount(self.masks & last_mask)
               + POP_WEIGHTS["hot"] * popcount(self.masks & hot_mask)).astype(np.float32)
        logw = np.log(bias["w"] * N_MAIN).astype(np.float32)  # log ratio vs uniform
        lp = logw[self.combos.astype(np.intp) - 1].sum(1)
        return lp - pop, pop, lp

    @staticmethod
    def pb_ranking(pb_hist, bias):
        pbs = np.arange(1, N_PB + 1)
        pop = (PB_POP["low"] * (pbs <= 12) + PB_POP["lucky"] * (pbs % 7 == 0)
               + PB_POP["last_draw"] * (pbs == pb_hist[-1]))
        s = np.log(bias["v"] * N_PB) - pop
        return pbs[np.lexsort((pbs, -s))]


def select_tickets(scorer, score, pb_rank, k, max_overlap, balance, rng):
    """Greedy pick of the best-scoring combinations subject to
    (a) any two tickets share at most `max_overlap` main numbers and
    (b) optionally, no number is used much more often than others (coverage)."""
    tiebreak = rng.random(len(score))
    order = np.lexsort((tiebreak, -score))
    cap = math.ceil(k * PICK / N_MAIN) + 1 if balance else k
    usage = np.zeros(N_MAIN + 1, dtype=int)
    chosen, chosen_masks = [], np.zeros(0, dtype=np.uint64)
    for start in range(0, len(order), 200_000):
        block = order[start:start + 200_000]
        for idx in block:
            m = scorer.masks[idx]
            if chosen and popcount(chosen_masks & m).max() > max_overlap:
                continue
            nums = scorer.combos[idx]
            if (usage[nums] >= cap).any():
                continue
            usage[nums] += 1
            chosen.append(idx)
            chosen_masks = np.append(chosen_masks, m)
            if len(chosen) == k:
                break
        if len(chosen) == k:
            break
    if len(chosen) < k:
        print(f"WARNING: only {len(chosen)} tickets satisfy the constraints "
              f"(relax --max-overlap or use --no-balance).")
    pbs = [int(pb_rank[i % N_PB]) for i in range(len(chosen))]
    return np.array(chosen), pbs


# --------------------------------------------------------------------------- odds
def division_probs():
    """Exact per-ticket probability of each prize division."""
    total = math.comb(N_MAIN, PICK)
    out = {}
    for m, pbm in DIVISIONS:
        p_main = math.comb(PICK, m) * math.comb(N_MAIN - PICK, PICK - m) / total
        out[(m, pbm)] = p_main * (1 / N_PB if pbm else (N_PB - 1) / N_PB)
    return out


def div_name(i):
    m, pbm = DIVISIONS[i]
    return f"Div {i + 1} ({m}{'+PB' if pbm else ''})"


def count_divisions(tickets, pbs, draw, draw_pb):
    dm = to_mask(draw)
    hits = np.zeros(len(DIVISIONS), dtype=int)
    for t, p in zip(tickets, pbs):
        m = int(np.bitwise_count(to_mask(t) & dm))
        key = (m, p == draw_pb)
        if key in DIVISIONS:
            hits[DIVISIONS.index(key)] += 1
    return hits


# --------------------------------------------------------------------------- backtest
_BT = {}


def _bt_init(combos):
    _BT["scorer"] = Scorer(combos)


def _bt_step(args):
    t, main, pb, k, max_overlap, balance = args
    scorer = _BT["scorer"]
    bias = fit_bias(main[:t], pb[:t])
    score, _, _ = scorer.score(main[:t], bias)
    idx, pbs = select_tickets(scorer, score, Scorer.pb_ranking(pb[:t], bias),
                              k, max_overlap, balance, np.random.default_rng(t))
    return count_divisions(scorer.combos[idx], pbs, main[t], pb[t])


def backtest(combos, main, pb, steps, k, max_overlap, balance, workers):
    first = len(main) - steps
    jobs = [(t, main, pb, k, max_overlap, balance) for t in range(first, len(main))]
    print(f"\nWalk-forward backtest: {steps} draws x {k} tickets, {workers} workers ...")
    t0 = time.time()
    with multiprocessing.Pool(workers, initializer=_bt_init, initargs=(combos,)) as pool:
        hits = np.sum(pool.map(_bt_step, jobs), axis=0)
    probs = division_probs()
    n_tickets = steps * k
    print(f"Done in {time.time() - t0:.1f} s. Tickets evaluated: {n_tickets}")
    print(f"{'Division':<16}{'model hits':>12}{'expected (chance)':>20}{'z':>8}")
    for i, d in enumerate(DIVISIONS):
        e = probs[d] * n_tickets
        z = (hits[i] - e) / math.sqrt(e) if e > 0 else 0
        print(f"{div_name(i):<16}{hits[i]:>12}{e:>20.2f}{z:>8.2f}")
    e_any = sum(probs.values()) * n_tickets
    print(f"{'Any prize':<16}{hits.sum():>12}{e_any:>20.2f}"
          f"{(hits.sum() - e_any) / math.sqrt(e_any):>8.2f}")
    print("|z| < 2 everywhere means the hit rate is consistent with pure chance.")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_FILE)
    ap.add_argument("--tickets", type=int, default=20)
    ap.add_argument("--max-overlap", type=int, default=3,
                    help="max main numbers any two tickets may share")
    ap.add_argument("--no-balance", action="store_true",
                    help="allow some numbers to appear on many more tickets than others")
    ap.add_argument("--window", type=int, default=0,
                    help="use only the last N draws for the bias test (0 = all)")
    ap.add_argument("--mc-reps", type=int, default=20000)
    ap.add_argument("--backtest", type=int, default=0, help="walk-forward over last N draws")
    ap.add_argument("--workers", type=int, default=multiprocessing.cpu_count())
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args()

    draw_no, dates, main_d, pb_d = load_draws(args.data)
    last_date = datetime.strptime(dates[-1], "%d %B, %Y")
    print(f"Loaded {len(main_d)} draws in the 7/35 format "
          f"(#{draw_no[0]} .. #{draw_no[-1]}, latest {dates[-1]}).")
    nxt = last_date + timedelta(days=7)
    if datetime.now() - last_date > timedelta(days=8):
        print(f"WARNING: data ends {dates[-1]}; run extract_powerball.py to refresh "
              f"before predicting.")

    hist_m = main_d[-args.window:] if args.window else main_d
    hist_p = pb_d[-args.window:] if args.window else pb_d
    bias = fit_bias(hist_m, hist_p)
    p_m, p_p = mc_pvalues(bias["n"], bias["x2_main"], bias["x2_pb"], args.mc_reps)
    print(f"\n--- Bias test on {bias['n']} draws ---")
    print(f"Main balls: X^2 = {bias['x2_main']:.1f} (fair-draw mean 28.0), "
          f"p = {p_m:.3f}, shrinkage lambda = {bias['lam_main']:.3f}")
    print(f"Powerball : X^2 = {bias['x2_pb']:.1f} (fair-draw mean 19.0), "
          f"p = {p_p:.3f}, shrinkage lambda = {bias['lam_pb']:.3f}")
    if p_m > 0.05 and p_p > 0.05:
        print("No statistically significant bias: draws are consistent with a fair machine.")
    print("Max per-ball edge after shrinkage: main "
          f"{(bias['w'].max() * N_MAIN - 1) * 100:+.2f}%, "
          f"PB {(bias['v'].max() * N_PB - 1) * 100:+.2f}% vs. uniform")

    t0 = time.time()
    combos = all_combos()
    scorer = Scorer(combos)
    score, pop, lp = scorer.score(main_d, bias)
    pb_rank = Scorer.pb_ranking(pb_d, bias)
    rng = np.random.default_rng(args.seed)
    idx, pbs = select_tickets(scorer, score, pb_rank, args.tickets,
                              args.max_overlap, not args.no_balance, rng)
    print(f"\nScored all {len(combos):,} combinations in {time.time() - t0:.1f} s.")

    rows = []
    print("\n" + "=" * 78)
    print(f"TOP {len(idx)} TICKETS for draw #{draw_no[-1] + 1} (~{nxt:%d %b %Y})")
    print("ranked by: shrunk bias likelihood + low popularity (bigger share if it wins)")
    print("=" * 78)
    for r, (i, p) in enumerate(zip(idx, pbs), 1):
        nums = [int(x) for x in combos[i]]
        share = math.exp(-float(pop[i]) + float(pop.mean()))
        print(f"{r:02d} | {' '.join(f'{n:2d}' for n in nums)} | PB {p:2d} "
              f"| popularity {pop[i]:.2f} | est. share x{share:.2f}")
        rows.append(dict(rank=r, numbers=" ".join(map(str, nums)), powerball=p,
                         popularity=round(float(pop[i]), 3),
                         bias_logratio=round(float(lp[i]), 5),
                         est_share_multiplier=round(share, 2)))
    covered = len(set(n for row in rows for n in map(int, row["numbers"].split())))
    print(f"\nCoverage: {covered}/35 main numbers, {len(set(pbs))}/20 Powerballs.")

    probs = division_probs()
    p_any = 1 - (1 - sum(probs.values())) ** len(idx)
    print(f"P(at least one prize, approx.) = {p_any:.1%}; "
          f"P(jackpot) = {len(idx)} in {math.comb(35, 7) * 20:,}.")

    out = args.out or f"top_predictions_draw_{draw_no[-1] + 1}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved to {out}")

    if args.backtest:
        backtest(combos, main_d, pb_d, args.backtest, args.tickets,
                 args.max_overlap, not args.no_balance, args.workers)


if __name__ == "__main__":
    main()
