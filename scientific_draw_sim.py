import math
import random
import pymunk

def create_drum(space, radius, center):
    """Creates the Halogen II transparent acrylic spherical mixing chamber."""
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = center
    
    segments = []
    num_segments = 100 # High resolution for smooth circle
    for i in range(num_segments):
        # We leave a gap at the very bottom (angles near pi/2) for the trapdoor
        angle1 = (i / num_segments) * 2 * math.pi
        angle2 = ((i + 1) / num_segments) * 2 * math.pi
        
        # Bottom is around pi/2 in pymunk's coordinate system (y goes up or down depending on setup, let's say y goes down)
        # We will manually handle the trapdoor, so let's just make a full circle first
        p1 = (radius * math.cos(angle1), radius * math.sin(angle1))
        p2 = (radius * math.cos(angle2), radius * math.sin(angle2))
        
        segment = pymunk.Segment(body, p1, p2, 5)
        segment.elasticity = 0.9
        segment.friction = 0.1
        segments.append(segment)
        
    space.add(body, *segments)
    return body, segments

def create_paddles(space, center):
    """Creates dual counter-rotating mixing arms at the base of the chamber."""
    # Left paddle
    paddle1_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    paddle1_body.position = (center[0] - 50, center[1] + 150)
    paddle1_body.angular_velocity = 8.0 # High speed spin
    
    paddle1_shape = pymunk.Segment(paddle1_body, (-80, 0), (80, 0), 10)
    paddle1_shape.elasticity = 1.0
    paddle1_shape.friction = 0.5
    
    # Right paddle
    paddle2_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    paddle2_body.position = (center[0] + 50, center[1] + 150)
    paddle2_body.angular_velocity = -8.0 # Counter-rotating
    
    paddle2_shape = pymunk.Segment(paddle2_body, (-80, 0), (80, 0), 10)
    paddle2_shape.elasticity = 1.0
    paddle2_shape.friction = 0.5
    
    space.add(paddle1_body, paddle1_shape, paddle2_body, paddle2_shape)

def drop_balls(space, center, num_balls=35, ball_radius=25):
    """Drops the solid polymer foam balls into the chamber."""
    balls = []
    mass = 50.0 # 50 grams
    inertia = pymunk.moment_for_circle(mass, 0, ball_radius, (0, 0))
    
    for i in range(1, num_balls + 1):
        body = pymunk.Body(mass, inertia)
        # Drop them slightly offset so they don't perfectly stack
        x = center[0] + random.uniform(-100, 100)
        y = center[1] - 100 + random.uniform(-100, 100)
        body.position = (x, y)
        
        shape = pymunk.Circle(body, ball_radius, (0, 0))
        shape.elasticity = 0.95 # Highly elastic polymer foam
        shape.friction = 0.2
        
        space.add(body, shape)
        balls.append((body, i))
        
    return balls

def run_scientific_simulation():
    print("Initializing Smartplay Halogen II Physics Simulation...")
    print("Engine: PyMunk (Rigid Body Dynamics)")
    
    space = pymunk.Space()
    space.gravity = (0, 981) # Gravity 9.81 m/s^2 scaled
    
    CHAMBER_RADIUS = 300
    CENTER = (400, 400)
    
    drum_body, drum_segments = create_drum(space, CHAMBER_RADIUS, CENTER)
    create_paddles(space, CENTER)
    balls = drop_balls(space, CENTER, num_balls=35, ball_radius=25)
    
    # Simulation settings
    dt = 1.0 / 1000.0 # 1 millisecond step for high mathematical precision
    mix_time_seconds = 5.0
    steps = int(mix_time_seconds / dt)
    
    print(f"\nPhase 1: Initial Gravity Drop & Mixing ({mix_time_seconds} seconds)")
    print(f"Executing {steps:,} physics collision steps in headless mode...")
    
    for _ in range(steps):
        space.step(dt)
        
    print("Mixing complete. The chamber is now in a state of mathematical chaos.")
    
    drawn_numbers = []
    
    # Phase 2: Drawing 7 balls sequentially
    # Real machines open a small gate at the bottom to catch a ball. 
    # We will simulate this by identifying the ball with the lowest Y coordinate (closest to the bottom).
    for draw_idx in range(1, 8):
        print(f"\nInitiating Draw Sequence {draw_idx}...")
        
        # Mix for an additional 2 seconds between draws
        for _ in range(int(2.0 / dt)):
            space.step(dt)
            
        # "Open the gate" -> Find the lowest ball
        lowest_ball = None
        max_y = -float('inf')
        
        for ball_body, number in balls:
            if ball_body.position.y > max_y: # y increases downwards in this coordinate system
                max_y = ball_body.position.y
                lowest_ball = (ball_body, number)
                
        drawn_number = lowest_ball[1]
        drawn_numbers.append(drawn_number)
        
        # Remove the drawn ball from the physics space
        space.remove(lowest_ball[0], list(lowest_ball[0].shapes)[0])
        balls.remove(lowest_ball)
        
        print(f"Ball Captured! Number: {drawn_number}")
        
    print("\n=======================================================")
    print(f"FINAL SIMULATED DRAW RESULTS: {sorted(drawn_numbers)}")
    print("=======================================================")
    print("\nNote: Running this script again will yield entirely different results")
    print("due to Python's micro-random initial drop positions, perfectly demonstrating")
    print("the Sensitive Dependence on Initial Conditions (Chaos Theory).")

if __name__ == "__main__":
    run_scientific_simulation()
