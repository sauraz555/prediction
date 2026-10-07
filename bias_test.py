"""
Pre-draw dry-run bias test (emulates equipment certification).

Regulators run dry draws with certified ball sets and check the results are
uniform. This script does the same in the simulated machine, for one ball set
whose balls differ in weight by up to +/- tolerance:

  1. Uniformity  - chi-square test of how often each number is drawn.
  2. Weight bias - correlation between a ball's weight and how often it is drawn,
                   and the draw rate of the heaviest third vs the lightest third.

Two sets are tested:
  * CERTIFIED  - tolerance from draw_machine.py (default +/- 0.03 g)
  * EXAGGERATED - a deliberately faulty set (default +/- 3 g), as a positive
                  control: it shows whether the test can detect weight bias at all.

Usage:
    python bias_test.py                        # 400 dry-run draws per set
    python bias_test.py --draws 1000 --machine-config my_machine.json
    python bias_test.py --workers 8            # limit CPU cores used
"""
import argparse
import multiprocessing
import os
import random
import time
from collections import Counter

from draw_machine import ball_set_masses, load_config, make_pool, simulate_drum

CHI2_CRIT_34DF_5PCT = 48.60  # chi-square critical value, 34 degrees of freedom, p = 0.05


def dry_run(args):
    cfg, masses = args
    rng = random.Random(os.urandom(16))
    return simulate_drum(cfg, rng, cfg.main_balls, cfg.main_draws, masses=masses)


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5 if vx and vy else 0.0


def test_set(cfg, label, tolerance, draws, pool):
    masses = ball_set_masses(cfg, cfg.main_balls, set_id=0, tolerance_g=tolerance)
    t0 = time.time()
    results = pool.map(dry_run, [(cfg, masses)] * draws)
    counts = Counter(n for draw in results for n in draw)

    n_balls = cfg.main_balls
    freq = [counts.get(b, 0) for b in range(1, n_balls + 1)]
    expected = draws * cfg.main_draws / n_balls
    chi2 = sum((f - expected) ** 2 / expected for f in freq)
    r = pearson(masses, freq)

    order = sorted(range(n_balls), key=lambda i: masses[i])
    third = n_balls // 3
    light = sum(freq[i] for i in order[:third]) / (third * draws)
    heavy = sum(freq[i] for i in order[-third:]) / (third * draws)
    fair = cfg.main_draws / n_balls

    print(f"\n{label}: weights {cfg.ball_mass_g:g} g +/- {tolerance:g} g, "
          f"{draws} dry-run draws ({time.time() - t0:.0f} s)")
    verdict = "PASS (uniform)" if chi2 < CHI2_CRIT_34DF_5PCT else "FAIL (non-uniform)"
    print(f"  Uniformity chi-square:        {chi2:6.1f}  (critical {CHI2_CRIT_34DF_5PCT} at 5%) -> {verdict}")
    print(f"  Weight vs frequency corr. r:  {r:+6.3f}  (|r| > 0.33 is significant at 5% for 35 balls)")
    print(f"  Draw rate per draw - heaviest third: {heavy:.4f}, lightest third: {light:.4f}, "
          f"fair: {fair:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=400, help="dry-run draws per ball set")
    parser.add_argument("--exaggerated", type=float, default=3.0,
                        help="weight tolerance (g) for the faulty positive-control set")
    parser.add_argument("--workers", type=int, help="CPU cores to use (default: all)")
    parser.add_argument("--machine-config", help="JSON file overriding machine parameters")
    args = parser.parse_args()
    cfg = load_config(args.machine_config)

    pool, workers = make_pool(args.workers)
    print(f"Dry-run bias test on {workers} of {multiprocessing.cpu_count()} cores, low priority "
          f"({cfg.mix_type} mix, {cfg.main_balls} balls, {cfg.main_draws} drawn per draw)")
    with pool:
        test_set(cfg, "CERTIFIED SET", cfg.ball_mass_tolerance_g, args.draws, pool)
        test_set(cfg, "EXAGGERATED (FAULTY) SET", args.exaggerated, args.draws, pool)


if __name__ == "__main__":
    main()
