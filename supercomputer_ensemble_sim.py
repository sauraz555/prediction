"""
Ensemble physics simulation of a gravity-mix lottery draw (Australian Powerball format).

Each "universe" is an independent 2D rigid-body simulation with microscopic
perturbations to the initial ball positions. Both the main draw (7 from 35) and
the Powerball (1 from 20) are drawn from simulated drums.

NOTE: This is a simplified 2D model with assumed parameters (drum size, paddle
speed, elasticity). It is NOT calibrated to the real Smartplay Halogen II, and
its outputs have no predictive power for real draws. Every combination it
produces is as likely (1 in 134,490,400) as any other.

Usage:
    python supercomputer_ensemble_sim.py                # 24 combinations
    python supercomputer_ensemble_sim.py --count 40     # 40 combinations
"""
import argparse
import math
import multiprocessing
import os
import random
import time

import pymunk

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

CHAMBER_RADIUS = 300
CENTER = (400, 400)
BALL_RADIUS = 25
BALL_MASS = 50.0
DT = 1.0 / 1000.0  # 1 ms physics step


def build_drum(space):
    """Static circular chamber plus two counter-rotating mixing paddles."""
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = CENTER
    segments = []
    n = 100
    for i in range(n):
        a1 = (i / n) * 2 * math.pi
        a2 = ((i + 1) / n) * 2 * math.pi
        p1 = (CHAMBER_RADIUS * math.cos(a1), CHAMBER_RADIUS * math.sin(a1))
        p2 = (CHAMBER_RADIUS * math.cos(a2), CHAMBER_RADIUS * math.sin(a2))
        seg = pymunk.Segment(body, p1, p2, 5)
        seg.elasticity = 0.9
        seg.friction = 0.1
        segments.append(seg)
    space.add(body, *segments)

    for dx, omega in ((-50, 8.0), (50, -8.0)):
        paddle = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        paddle.position = (CENTER[0] + dx, CENTER[1] + 150)
        paddle.angular_velocity = omega
        shape = pymunk.Segment(paddle, (-80, 0), (80, 0), 10)
        shape.elasticity = 1.0
        shape.friction = 0.5
        space.add(paddle, shape)


def simulate_drum(rng, num_balls, num_draws, mix_steps=3000, between_steps=500):
    """Simulate one drum: load balls with micro-noise, mix, then draw sequentially."""
    space = pymunk.Space()
    space.gravity = (0, 981)
    build_drum(space)

    inertia = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS, (0, 0))
    balls = []
    for number in range(1, num_balls + 1):
        idx = number - 1
        row, col = divmod(idx, 6)
        base_x = CENTER[0] - 150 + col * 60
        base_y = CENTER[1] - 200 + row * 60
        body = pymunk.Body(BALL_MASS, inertia)
        # Microscopic perturbation (1e-4 units) -> divergence via chaos
        body.position = (base_x + rng.uniform(-1e-4, 1e-4),
                         base_y + rng.uniform(-1e-4, 1e-4))
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = 0.95
        shape.friction = 0.2
        space.add(body, shape)
        balls.append((body, shape, number))

    for _ in range(mix_steps):
        space.step(DT)

    drawn = []
    for _ in range(num_draws):
        for _ in range(between_steps):
            space.step(DT)
        # "Gate" at the bottom: capture the lowest ball (largest y)
        lowest = max(balls, key=lambda b: b[0].position.y)
        drawn.append(lowest[2])
        space.remove(lowest[0], lowest[1])
        balls.remove(lowest)
    return drawn


def simulate_universe(_universe_id):
    """One full draw: 7 main balls from 35, and 1 Powerball from 20 (separate drum)."""
    rng = random.Random(os.urandom(16))  # independent OS-entropy seed per universe
    main = simulate_drum(rng, num_balls=35, num_draws=7)
    pb = simulate_drum(rng, num_balls=20, num_draws=1)[0]
    return tuple(sorted(main)), pb


def run_ensemble(count):
    cores = multiprocessing.cpu_count()
    print(f"Running ensemble on {cores} CPU cores, target: {count} new combinations...")
    start = time.time()

    results = []
    seen = set(PREVIOUS_COMBOS)
    with multiprocessing.Pool(processes=cores) as pool:
        while len(results) < count:
            batch = pool.map(simulate_universe, range(count - len(results)))
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
    args = parser.parse_args()
    run_ensemble(args.count)
