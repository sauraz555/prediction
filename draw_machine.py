"""
Physical model of the Australian Powerball draw machine (Smartplay Halogen II).

All parameters live in one place, in real units (millimetres, grams, seconds),
and each one carries its provenance:

  VERIFIED - taken from a published source (cited in SOURCES below)
  ASSUMED  - no public figure found; a plausible placeholder you should replace
             with researched values (see README / --machine-config)

Override any parameter with a JSON file, e.g. machine_config.json:

    {"chamber_diameter_mm": 580, "interval_between_balls_s": 9.5}

and pass it with:  python supercomputer_ensemble_sim.py --machine-config machine_config.json

NOTE: this is a 2D cross-section of a 3D gravity-mix sphere. Even with perfect
parameters, a real draw is chaotic and cannot be predicted by this model.
"""
import json
import math
from dataclasses import asdict, dataclass, fields

import pymunk

SOURCES = {
    "smartplay": "Smartplay International, Halogen lottery draw machines "
                 "(https://www.smartplay.com/lottery-products/lottery-draw-machines)",
    "wiki": "Powerball (Australia), Wikipedia "
            "(https://en.wikipedia.org/wiki/Powerball_(Australia))",
    "thelott_balls": "The Lott, 'On The Ball: The Lottery Balls' "
                     "(https://www.thelott.com/real-winners/the-lott/on-the-ball-the-lottery-balls)",
    "thelott_rules": "The Lott, Powerball how to play "
                     "(https://www.thelott.com/powerball/how-to-play)",
}

# name -> (status, note). Kept next to the dataclass so the two stay in sync.
PROVENANCE = {
    "main_balls":               ("VERIFIED", "35 balls in main barrel [thelott_rules]"),
    "main_draws":               ("VERIFIED", "7 main numbers drawn [thelott_rules]"),
    "powerball_balls":          ("VERIFIED", "20 balls in separate Powerball barrel [thelott_rules]"),
    "powerball_draws":          ("VERIFIED", "1 Powerball drawn [thelott_rules]"),
    "ball_diameter_mm":         ("VERIFIED", "50 mm foam balls on Halogen machines [smartplay]; "
                                             "machine = 2x Halogen II since 2018 [wiki]"),
    "ball_mass_g":              ("ASSUMED", "not published; The Lott only says each ball is weighed "
                                            "to 0.001 g by NMI [thelott_balls]. ~10 g estimated "
                                            "from 50 mm polymer foam (~0.15 g/cm^3)"),
    "ball_elasticity":          ("ASSUMED", "polymer foam restitution estimate"),
    "ball_friction":            ("ASSUMED", "foam-on-foam / foam-on-acrylic estimate"),
    "chamber_diameter_mm":      ("ASSUMED", "not published; ~600 mm rough guess, replace with a measured value"),
    "wall_elasticity":          ("ASSUMED", "acrylic chamber wall estimate"),
    "wall_friction":            ("ASSUMED", "acrylic chamber wall estimate"),
    "mix_type":                 ("VERIFIED", "'gravity' (paddle arms): Smartplay lists Halogen/Halogen II as its "
                                             "gravity-mix line [smartplay]; 'air' = blower-jet alternative"),
    "paddle_rpm":               ("ASSUMED", "not published; gravity-mix arms, ~75 rpm estimate"),
    "paddle_length_mm":         ("ASSUMED", "mixing arm length, scaled to chamber"),
    "air_jet_accel_mm_s2":      ("ASSUMED", "air mode only: upward blower acceleration (~2.5 g)"),
    "initial_mix_s":            ("ASSUMED", "not published; mixing time before first ball"),
    "interval_between_balls_s": ("ASSUMED", "not published; time from one ball captured to next"),
    "gravity_mm_s2":            ("VERIFIED", "standard gravity 9.81 m/s^2"),
    "dt_s":                     ("MODEL", "physics time step (numerical setting, not physical)"),
}


@dataclass(frozen=True)
class MachineConfig:
    # Game format
    main_balls: int = 35
    main_draws: int = 7
    powerball_balls: int = 20
    powerball_draws: int = 1
    # Balls
    ball_diameter_mm: float = 50.0
    ball_mass_g: float = 10.0
    ball_elasticity: float = 0.80
    ball_friction: float = 0.40
    # Chamber (container)
    chamber_diameter_mm: float = 600.0
    wall_elasticity: float = 0.70
    wall_friction: float = 0.20
    # Mixing ("gravity" = paddle arms, "air" = blower jet, no paddles)
    mix_type: str = "gravity"
    paddle_rpm: float = 75.0
    paddle_length_mm: float = 160.0
    air_jet_accel_mm_s2: float = 24500.0
    # Draw timing
    initial_mix_s: float = 20.0
    interval_between_balls_s: float = 6.0
    # Physics
    gravity_mm_s2: float = 9810.0
    dt_s: float = 1.0 / 1000.0

    @property
    def ball_radius_mm(self):
        return self.ball_diameter_mm / 2

    @property
    def chamber_radius_mm(self):
        return self.chamber_diameter_mm / 2

    @property
    def mix_steps(self):
        return round(self.initial_mix_s / self.dt_s)

    @property
    def between_steps(self):
        return round(self.interval_between_balls_s / self.dt_s)

    def simulated_seconds(self):
        """Simulated machine time for one full draw (both barrels)."""
        main = self.initial_mix_s + self.main_draws * self.interval_between_balls_s
        pb = self.initial_mix_s + self.powerball_draws * self.interval_between_balls_s
        return main + pb


def load_config(path=None):
    """Defaults, optionally overridden by a JSON file of {field: value}."""
    if path is None:
        return MachineConfig()
    with open(path) as f:
        overrides = json.load(f)
    known = {f.name for f in fields(MachineConfig)}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"Unknown machine parameters: {sorted(unknown)}")
    cfg = MachineConfig(**overrides)
    if cfg.mix_type not in ("gravity", "air"):
        raise ValueError("mix_type must be 'gravity' or 'air'")
    return cfg


def describe(cfg):
    """Human-readable parameter table with provenance."""
    defaults = MachineConfig()
    lines = [f"{'parameter':<26} {'value':>10}  {'status':<9} note"]
    for name, value in asdict(cfg).items():
        status, note = PROVENANCE[name]
        if value != getattr(defaults, name):
            status = "OVERRIDE"
        shown = f"{value:g}" if isinstance(value, float) else str(value)
        lines.append(f"{name:<26} {shown:>10}  {status:<9} {note}")
    return "\n".join(lines)


# --- Physics ---------------------------------------------------------------
# Coordinates in mm with the chamber centre at the origin; y increases downward
# (gravity is +y), so the bottom gate is at the largest y.

def build_drum(space, cfg):
    """Static circular chamber, plus two counter-rotating mixing arms near the
    base in gravity mode (air mode has no arms; see apply_air_jet)."""
    r = cfg.chamber_radius_mm
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    n = 120
    segments = []
    for i in range(n):
        a1 = (i / n) * 2 * math.pi
        a2 = ((i + 1) / n) * 2 * math.pi
        seg = pymunk.Segment(body, (r * math.cos(a1), r * math.sin(a1)),
                             (r * math.cos(a2), r * math.sin(a2)), 5)
        seg.elasticity = cfg.wall_elasticity
        seg.friction = cfg.wall_friction
        segments.append(seg)
    space.add(body, *segments)
    if cfg.mix_type == "air":
        return

    omega = cfg.paddle_rpm * 2 * math.pi / 60
    half = cfg.paddle_length_mm / 2
    for dx, w in ((-r / 6, omega), (r / 6, -omega)):
        paddle = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        paddle.position = (dx, r / 2)
        paddle.angular_velocity = w
        shape = pymunk.Segment(paddle, (-half, 0), (half, 0), 10)
        shape.elasticity = cfg.ball_elasticity
        shape.friction = 0.5
        space.add(paddle, shape)


def loading_positions(cfg, num_balls):
    """Deterministic rest positions for loading: a lattice inside the chamber,
    filled from the top down (balls are released from above the mixing arms)."""
    d = cfg.ball_diameter_mm * 1.05
    limit = cfg.chamber_radius_mm - cfg.ball_diameter_mm
    k = int(limit // d)
    pts = [(i * d, j * d) for j in range(-k, k + 1) for i in range(-k, k + 1)
           if math.hypot(i * d, j * d) <= limit]
    pts.sort(key=lambda p: (p[1], p[0]))
    if len(pts) < num_balls:
        raise ValueError(f"{num_balls} balls of {cfg.ball_diameter_mm} mm do not fit "
                         f"in a {cfg.chamber_diameter_mm} mm chamber")
    return pts[:num_balls]


def load_balls(space, cfg, num_balls, rng, jitter=1e-4, perturb_ball=None, perturb=0.0):
    """Add numbered balls with microscopic random offsets (initial-condition noise)."""
    r = cfg.ball_radius_mm
    inertia = pymunk.moment_for_circle(cfg.ball_mass_g, 0, r, (0, 0))
    balls = []
    for number, (x, y) in enumerate(loading_positions(cfg, num_balls), start=1):
        x += rng.uniform(-jitter, jitter)
        y += rng.uniform(-jitter, jitter)
        if number == perturb_ball:
            x += perturb
        body = pymunk.Body(cfg.ball_mass_g, inertia)
        body.position = (x, y)
        shape = pymunk.Circle(body, r)
        shape.elasticity = cfg.ball_elasticity
        shape.friction = cfg.ball_friction
        space.add(body, shape)
        balls.append((body, shape, number))
    return balls


def new_space(cfg):
    space = pymunk.Space()
    space.gravity = (0, cfg.gravity_mm_s2)
    build_drum(space, cfg)
    return space


def apply_air_jet(balls, cfg):
    """Air mode: a blower in the lower-central chamber pushes balls upward."""
    r = cfg.chamber_radius_mm
    for body, _, _ in balls:
        x, y = body.position
        if y > 0 and abs(x) < r / 2:
            body.apply_force_at_local_point((0, -body.mass * cfg.air_jet_accel_mm_s2))


def run_draw(space, balls, cfg, num_draws, on_step=None):
    """Mix for initial_mix_s, then capture one ball every interval_between_balls_s.
    The gate captures the lowest ball (largest y)."""
    air = cfg.mix_type == "air"

    def step():
        if air:
            apply_air_jet(balls, cfg)
        space.step(cfg.dt_s)
        if on_step:
            on_step()

    for _ in range(cfg.mix_steps):
        step()
    drawn = []
    for _ in range(num_draws):
        for _ in range(cfg.between_steps):
            step()
        lowest = max(balls, key=lambda b: b[0].position.y)
        drawn.append(lowest[2])
        space.remove(lowest[0], lowest[1])
        balls.remove(lowest)
    return drawn


def simulate_drum(cfg, rng, num_balls, num_draws):
    space = new_space(cfg)
    balls = load_balls(space, cfg, num_balls, rng)
    return run_draw(space, balls, cfg, num_draws)
