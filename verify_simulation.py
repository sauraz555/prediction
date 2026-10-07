"""
Verification harness for the lottery physics simulation.

Shows that the simulation really computes physics, rather than just
returning random numbers:

  Test 1 - Work done: counts physics steps and actual collisions in one draw,
           and times it on a single CPU core.
  Test 2 - Determinism: the same initial conditions must give exactly the same
           draw every time. A random-number picker could not do this.
  Test 3 - Chaos: two runs that start identically except for one ball moved
           by 0.0001 mm. Tracks how far apart the balls get over time.
  Test 4 - Sanity: no ball escapes the drum during mixing.

Run:  python verify_simulation.py [machine_config.json]
"""
import math
import random
import sys
import time

from draw_machine import apply_air_jet, load_balls, load_config, new_space, run_draw as machine_run_draw

CFG = load_config(sys.argv[1] if len(sys.argv) > 1 else None)


def make_space(seed, num_balls=35, perturb_ball=None, perturb=0.0):
    """Build a drum and load the balls. Same seed gives the same start.
    Optionally shift one ball by `perturb` mm in x."""
    space = new_space(CFG)
    balls = load_balls(space, CFG, num_balls, random.Random(seed),
                       perturb_ball=perturb_ball, perturb=perturb)
    return space, balls


def run_draw(space, balls, num_draws=7, count_collisions=False):
    """Mix, then draw balls one at a time. Optionally count new contacts."""
    ball_shapes = {b[1] for b in balls}
    stats = {"steps": 0, "ball_ball": 0.0, "ball_wall_or_paddle": 0, "escapes": 0}
    limit = CFG.chamber_radius_mm + CFG.ball_radius_mm

    def tally(arb):
        if not arb.is_first_contact:
            return
        a, b = arb.shapes
        if a in ball_shapes and b in ball_shapes:
            stats["ball_ball"] += 0.5  # seen once from each ball
        else:
            stats["ball_wall_or_paddle"] += 1

    def on_step():
        stats["steps"] += 1
        if count_collisions:
            for body, _, _ in balls:
                body.each_arbiter(tally)
                if math.hypot(*body.position) > limit:
                    stats["escapes"] += 1

    drawn = machine_run_draw(space, balls, CFG, num_draws, on_step=on_step)
    return drawn, stats


def test_work_done():
    print("TEST 1: Work done in ONE draw (single CPU core, instrumented)")
    t0 = time.perf_counter()
    space, balls = make_space(seed=12345)
    drawn, s = run_draw(space, balls, count_collisions=True)
    elapsed = time.perf_counter() - t0
    print(f"  Physics steps ({CFG.dt_s * 1000:g} ms each):    {s['steps']:,}")
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
    print("TEST 3: Chaos - ball #1 moved by 0.0001 mm in universe B")
    sa, ba = make_space(seed=999)
    sb, bb = make_space(seed=999, perturb_ball=1, perturb=1e-4)
    print(f"  {'time (s)':>8} | {f'max separation of any ball (mm; ball diameter = {CFG.ball_diameter_mm:g})':>55}")
    report_every = max(1, CFG.mix_steps // 20)
    for step in range(CFG.mix_steps + 1):
        if step % report_every == 0:
            sep = max(
                math.dist(a[0].position, b[0].position) for a, b in zip(ba, bb)
            )
            print(f"  {step * CFG.dt_s:>8.2f} | {sep:>55.6g}")
        for space, balls in ((sa, ba), (sb, bb)):
            if CFG.mix_type == "air":
                apply_air_jet(balls, CFG)
            space.step(CFG.dt_s)
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
