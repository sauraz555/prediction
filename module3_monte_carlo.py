import numpy as np
import time

def run_monte_carlo(n_simulations=10_000_000):
    """
    Simulates N Powerball draws and compares a single 'ticket' against each draw
    to empirically verify the theoretical odds.
    """
    print(f"Starting Monte Carlo simulation with {n_simulations:,} draws...")
    start_time = time.time()
    
    target_white = set([1, 2, 3, 4, 5])
    target_pb = 6
    
    jackpots = 0
    match_5 = 0
    match_4_pb = 0
    match_4 = 0
    match_3_pb = 0
    match_3 = 0
    match_2_pb = 0
    match_1_pb = 0
    match_0_pb = 0
    
    batch_size = 1_000_000
    batches = n_simulations // batch_size
    
    for b in range(batches):
        pbs = np.random.randint(1, 27, size=batch_size)
        rand_arrays = np.random.rand(batch_size, 69)
        wbs = np.argsort(rand_arrays, axis=1)[:, :5] + 1
        
        pb_matches = (pbs == target_pb)
        is_in_target = np.isin(wbs, list(target_white))
        wb_matches = np.sum(is_in_target, axis=1)
        
        jackpots += np.sum((wb_matches == 5) & pb_matches)
        match_5 += np.sum((wb_matches == 5) & ~pb_matches)
        match_4_pb += np.sum((wb_matches == 4) & pb_matches)
        match_4 += np.sum((wb_matches == 4) & ~pb_matches)
        match_3_pb += np.sum((wb_matches == 3) & pb_matches)
        match_3 += np.sum((wb_matches == 3) & ~pb_matches)
        match_2_pb += np.sum((wb_matches == 2) & pb_matches)
        match_1_pb += np.sum((wb_matches == 1) & pb_matches)
        match_0_pb += np.sum((wb_matches == 0) & pb_matches)
        
        print(f"Processed {(b+1) * batch_size:,} draws...")

    total_time = time.time() - start_time
    print(f"\nSimulation finished in {total_time:.2f} seconds.")
    print("\n=== MONTE CARLO EMPIRICAL RESULTS ===")
    
    results = {
        "Jackpot (5 + PB)": (jackpots, 292201338),
        "Match 5": (match_5, 11688053),
        "Match 4 + PB": (match_4_pb, 913129),
        "Match 4": (match_4, 36525),
        "Match 3 + PB": (match_3_pb, 14494),
        "Match 3": (match_3, 579),
        "Match 2 + PB": (match_2_pb, 701),
        "Match 1 + PB": (match_1_pb, 91),
        "Match 0 + PB": (match_0_pb, 38)
    }
    
    print(f"{'Prize Tier':<18} | {'Simulated Hits':<15} | {'Empirical Odds':<20} | {'Theoretical Odds':<20}")
    print("-" * 80)
    for tier, (hits, theory_odds) in results.items():
        empirical = f"1 in {(n_simulations / hits):,.1f}" if hits > 0 else "No Hits"
        print(f"{tier:<18} | {hits:<15} | {empirical:<20} | 1 in {theory_odds:,}")

if __name__ == "__main__":
    run_monte_carlo(n_simulations=10_000_000)
