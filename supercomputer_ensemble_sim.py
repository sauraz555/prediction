import math
import random
import pymunk
import multiprocessing
import collections
import time

def simulate_universe(universe_id):
    """
    Simulates a single 'chaotic universe' physics run.
    Each universe introduces microscopic perturbations to the initial states.
    """
    # 1. Setup Space
    space = pymunk.Space()
    space.gravity = (0, 981)
    
    CHAMBER_RADIUS = 300
    CENTER = (400, 400)
    
    # 2. Chamber
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = CENTER
    segments = []
    num_segments = 100
    for i in range(num_segments):
        angle1 = (i / num_segments) * 2 * math.pi
        angle2 = ((i + 1) / num_segments) * 2 * math.pi
        p1 = (CHAMBER_RADIUS * math.cos(angle1), CHAMBER_RADIUS * math.sin(angle1))
        p2 = (CHAMBER_RADIUS * math.cos(angle2), CHAMBER_RADIUS * math.sin(angle2))
        segment = pymunk.Segment(body, p1, p2, 5)
        segment.elasticity = 0.9
        segment.friction = 0.1
        segments.append(segment)
    space.add(body, *segments)
    
    # 3. Paddles
    paddle1_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    paddle1_body.position = (CENTER[0] - 50, CENTER[1] + 150)
    paddle1_body.angular_velocity = 8.0
    paddle1_shape = pymunk.Segment(paddle1_body, (-80, 0), (80, 0), 10)
    paddle1_shape.elasticity = 1.0
    
    paddle2_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    paddle2_body.position = (CENTER[0] + 50, CENTER[1] + 150)
    paddle2_body.angular_velocity = -8.0
    paddle2_shape = pymunk.Segment(paddle2_body, (-80, 0), (80, 0), 10)
    
    space.add(paddle1_body, paddle1_shape, paddle2_body, paddle2_shape)
    
    # 4. Balls with Microscopic Perturbations (The Butterfly Effect)
    balls = []
    mass = 50.0
    ball_radius = 25
    inertia = pymunk.moment_for_circle(mass, 0, ball_radius, (0, 0))
    
    # Seed random uniquely for this CPU core/universe
    random.seed(time.time() * (universe_id + 1))
    
    for i in range(1, 36):
        ball_body = pymunk.Body(mass, inertia)
        
        # Base position grid
        row = i // 6
        col = i % 6
        base_x = CENTER[0] - 150 + (col * 60)
        base_y = CENTER[1] - 150 + (row * 60)
        
        # PERTURBATION: Add microscopic noise (10^-4 pixels)
        # This is where chaos theory diverges the universes
        noise_x = random.uniform(-0.0001, 0.0001)
        noise_y = random.uniform(-0.0001, 0.0001)
        
        ball_body.position = (base_x + noise_x, base_y + noise_y)
        
        shape = pymunk.Circle(ball_body, ball_radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.2
        space.add(ball_body, shape)
        balls.append((ball_body, i))
        
    # 5. Execute Simulation
    dt = 1.0 / 1000.0
    # 3 seconds of high-speed mixing
    for _ in range(3000):
        space.step(dt)
        
    drawn_numbers = []
    
    # Draw 7 balls
    for _ in range(7):
        for _ in range(500): # 0.5s mix between draws
            space.step(dt)
            
        lowest_ball = None
        max_y = -float('inf')
        for b_body, number in balls:
            if b_body.position.y > max_y:
                max_y = b_body.position.y
                lowest_ball = (b_body, number)
                
        drawn_numbers.append(lowest_ball[1])
        space.remove(lowest_ball[0], list(lowest_ball[0].shapes)[0])
        balls.remove(lowest_ball)
        
    return tuple(sorted(drawn_numbers))

def run_ensemble(total_universes=64):
    """
    Runs the physics engine across all available CPU cores.
    """
    cores = multiprocessing.cpu_count()
    print(f"--- SUPERCOMPUTER ENSEMBLE SIMULATION INITIALIZED ---")
    print(f"Detected {cores} CPU Cores. Spawning parallel universes...")
    
    start_time = time.time()
    
    # Multiprocessing Pool
    with multiprocessing.Pool(processes=cores) as pool:
        # Map the simulate function across N universes
        results = pool.map(simulate_universe, range(total_universes))
        
    elapsed = time.time() - start_time
    print(f"Simulated {total_universes} chaotic divergent timelines in {elapsed:.2f} seconds.\n")
    
    # Aggregate Results
    # Count frequency of each full combination
    combination_counts = collections.Counter(results)
    
    print("===================================================================")
    print(f"Top 16 Emergent Physic-Calculated Combinations from the Multiverse:")
    print("===================================================================")
    
    # We want 16 distinct combinations. If there are duplicates, we rank by frequency.
    # In a truly chaotic system, almost all combinations will have a frequency of 1,
    # proving the uniform distribution of chaos.
    top_16 = combination_counts.most_common(16)
    
    for i, (combo, count) in enumerate(top_16, 1):
        # Generate a random powerball for completeness
        pb = random.randint(1, 20)
        print(f"Timeline Cluster {i:02d} | White Balls: {list(combo)} | Powerball: {pb}")

if __name__ == "__main__":
    # For a real supercomputer, you would scale total_universes to 10,000,000+
    # For this desktop execution, we will simulate 64 parallel universes.
    run_ensemble(total_universes=64)
