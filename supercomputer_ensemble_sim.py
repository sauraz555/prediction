"""
Ensemble physics simulation of a gravity-mix lottery draw (Australian Powerball format).

Each "universe" is an independent 2D rigid-body simulation with microscopic
perturbations to the initial ball positions. Both the main draw (7 from 35) and
the Powerball (1 from 20) are drawn from simulated drums.

Physical parameters (ball size, chamber size, paddle speed, draw timing, ...)
come from draw_machine.py, where each is marked VERIFIED or ASSUMED.

NOTE: This is a simplified 2D model and several parameters are assumed. Even a
perfectly calibrated model has no predictive power for real draws. Every
combination it produces is as likely (1 in 134,490,400) as any other.

Usage:
    python supercomputer_ensemble_sim.py                # 24 combinations
    python supercomputer_ensemble_sim.py --count 40     # 40 combinations
    python supercomputer_ensemble_sim.py --machine-config my_machine.json
"""
import argparse
import multiprocessing
import os
import random
import time

from draw_machine import describe, load_config, simulate_drum

# Combinations already generated in the previous run (excluded from new output)
PREVIOUS_COMBOS = {
    (3, 13, 15, 20, 26, 28, 31), (1, 10, 13, 17, 24, 34, 35),
    (15, 23, 30, 32, 33, 34, 35), (4, 6, 15, 23, 27, 28, 29),
    (2, 11, 15, 19, 28, 32, 33), (6, 8, 13, 18, 26, 29, 34),
    (2, 3, 6, 10, 22, 27, 33), (11, 12, 13, 19, 23, 26, 30),
    (9, 13, 17, 27, 28, 31, 35), (2, 5, 9, 12, 21, 32, 34),
    (2, 8, 11, 17, 23, 28, 33), (5, 9, 19, 21, 22, 26, 30),
    (8, 13, 20, 23, 28, 29, 32), (3, 5, 13, 25, 29, 33, 34),
    (6, 10, 16, 22, 26, 28, 35), (4, 10, 24, 29, 30, 34, 35),
}


def simulate_universe(args):
    """One full draw: 7 main balls from 35, and 1 Powerball from 20 (separate drum)."""
    _universe_id, cfg = args
    rng = random.Random(os.urandom(16))  # independent OS-entropy seed per universe
    main = simulate_drum(cfg, rng, cfg.main_balls, cfg.main_draws)
    pb = simulate_drum(cfg, rng, cfg.powerball_balls, cfg.powerball_draws)[0]
    return tuple(sorted(main)), pb


def run_ensemble(count, cfg):
    print("Draw machine parameters (VERIFIED = published source, ASSUMED = placeholder):")
    print(describe(cfg))
    print(f"\nSimulated machine time per draw: {cfg.simulated_seconds():.0f} s "
          f"({cfg.mix_type} mix)\n")
    cores = multiprocessing.cpu_count()
    print(f"Running ensemble on {cores} CPU cores, target: {count} new combinations...")
    start = time.time()

    results = []
    seen = set(PREVIOUS_COMBOS)
    with multiprocessing.Pool(processes=cores) as pool:
        while len(results) < count:
            batch = pool.map(simulate_universe,
                             [(i, cfg) for i in range(count - len(results))])
            for combo, pb in batch:
                if combo not in seen:
                    seen.add(combo)
                    results.append((combo, pb))

    print(f"Done in {time.time() - start:.2f} s.\n")
    print("=" * 66)
    print(f"{count} simulated draws (unranked; each is equally likely)")
    print("=" * 66)
    for i, (combo, pb) in enumerate(results, 1):
        print(f"{i:02d} | Main: {list(combo)} | Powerball: {pb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=24, help="number of combinations")
    parser.add_argument("--machine-config", help="JSON file overriding draw machine "
                        "parameters (see draw_machine.py)")
    args = parser.parse_args()
    run_ensemble(args.count, load_config(args.machine_config))
