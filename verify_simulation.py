"""
Verification harness for the lottery physics simulation.

Shows that the simulation really computes physics, rather than just
returning random numbers:

  Test 1 - Work done: counts physics steps and actual collisions in one draw,
           and times it on a single CPU core.
  Test 2 - Determinism: the same initial conditions must give exactly the same
           draw every time. A random-number picker could not do this.
  Test 3 - Chaos: two runs that start identically except for one ball moved
           by 0.0001 units. Tracks how far apart the balls get over time.
  Test 4 - Sanity: no ball escapes the drum during mixing.

Run:  python verify_simulation.py
"""
import math
import random
import time

import pymunk

from supercomputer_ensemble_sim import (
    BALL_MASS, BALL_RADIUS, CENTER, CHAMBER_RADIUS, DT, build_drum,
)

MIX_STEPS = 3000
BETWEEN_STEPS = 500


def make_space(seed, num_balls=35, perturb_ball=None, perturb=0.0):
    """Build a drum and load the balls. Same seed gives the same start.
    Optionally shift one ball by `perturb` units in x."""
    rng = random.Random(seed)
    space = pymunk.Space()
    space.gravity = (0, 981)
    build_drum(space)
    inertia = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS, (0, 0))
    balls = []
    for number in range(1, num_balls + 1):
        row, col = divmod(number - 1, 6)
        x = CENTER[0] - 150 + col * 60 + rng.uniform(-1e-4, 1e-4)
        y = CENTER[1] - 200 + row * 60 + rng.uniform(-1e-4, 1e-4)
        if number == perturb_ball:
            x += perturb
        body = pymunk.Body(BALL_MASS, inertia)
        body.position = (x, y)
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.elasticity = 0.95
        shape.friction = 0.2
        space.add(body, shape)
        balls.append((body, shape, number))
    return space, balls


def run_draw(space, balls, num_draws=7, count_collisions=False):
    """Mix, then draw balls one at a time. Optionally count new contacts."""
    ball_shapes = {b[1] for b in balls}
    stats = {"steps": 0, "ball_ball": 0.0, "ball_wall_or_paddle": 0, "escapes": 0}

    def tally(arb):
        if not arb.is_first_contact:
            return
        a, b = arb.shapes
        if a in ball_shapes and b in ball_shapes:
            stats["ball_ball"] += 0.5  # seen once from each ball
        else:
            stats["ball_wall_or_paddle"] += 1

    def step():
        space.step(DT)
        stats["steps"] += 1
        if count_collisions:
            for body, _, _ in balls:
                body.each_arbiter(tally)
                dx = body.position.x - CENTER[0]
                dy = body.position.y - CENTER[1]
                if math.hypot(dx, dy) > CHAMBER_RADIUS + BALL_RADIUS:
                    stats["escapes"] += 1

    for _ in range(MIX_STEPS):
        step()
    drawn = []
    for _ in range(num_draws):
        for _ in range(BETWEEN_STEPS):
            step()
        lowest = max(balls, key=lambda b: b[0].position.y)
        drawn.append(lowest[2])
        space.remove(lowest[0], lowest[1])
        balls.remove(lowest)
        ball_shapes.discard(lowest[1])
    return drawn, stats


def test_work_done():
    print("TEST 1: Work done in ONE draw (single CPU core, instrumented)")
    t0 = time.perf_counter()
    space, balls = make_space(seed=12345)
    drawn, s = run_draw(space, balls, count_collisions=True)
    elapsed = time.perf_counter() - t0
    print(f"  Physics steps (1 ms each):    {s['steps']:,}")
    print(f"  Ball-to-ball collisions:      {int(s['ball_ball']):,}")
    print(f"  Ball-to-wall/paddle impacts:  {s['ball_wall_or_paddle']:,}")
    print(f"  Escaped-ball events:          {s['escapes']}")
    print(f"  Draw order:                   {drawn}")
    print(f"  Time with instrumentation:    {elapsed:.2f} s")

    t0 = time.perf_counter()
    space, balls = make_space(seed=12345)
    run_draw(space, balls)
    raw = time.perf_counter() - t0
    print(f"  Time without instrumentation: {raw:.2f} s  "
          f"({s['steps'] / raw:,.0f} steps/sec, Chipmunk2D C engine)\n")
    return s


def test_determinism():
    print("TEST 2: Determinism (same start -> same result, 3 runs)")
    results = []
    for _ in range(3):
        space, balls = make_space(seed=777)
        drawn, _ = run_draw(space, balls)
        results.append(drawn)
        print(f"  Run: {drawn}")
    print(f"  Identical: {all(r == results[0] for r in results)}\n")


def test_chaos():
    print("TEST 3: Chaos - ball #1 moved by 0.0001 units in universe B")
    sa, ba = make_space(seed=999)
    sb, bb = make_space(seed=999, perturb_ball=1, perturb=1e-4)
    print(f"  {'time (s)':>8} | {'max separation of any ball (units; ball diameter = 50)':>55}")
    for step in range(MIX_STEPS + 1):
        if step % 250 == 0:
            sep = max(
                math.dist(a[0].position, b[0].position) for a, b in zip(ba, bb)
            )
            print(f"  {step * DT:>8.2f} | {sep:>55.6g}")
        sa.step(DT)
        sb.step(DT)
    da, _ = run_draw(sa, ba)
    db, _ = run_draw(sb, bb)
    print(f"  Universe A draw: {da}")
    print(f"  Universe B draw: {db}")
    print(f"  Numbers in common: {len(set(da) & set(db))} of 7\n")


if __name__ == "__main__":
    stats = test_work_done()
    test_determinism()
    test_chaos()
    print("TEST 4: Sanity")
    print(f"  Escaped-ball events during Test 1: {stats['escapes']} (should be 0)")
