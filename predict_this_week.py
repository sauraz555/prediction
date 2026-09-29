import pandas as pd
import numpy as np
import random

class MarkovChainModel:
    """Markov Chain that tracks transition probabilities between numbers appearing in consecutive draws."""
    def __init__(self, max_white_ball=35):
        self.max_wb = max_white_ball
        self.transitions = np.zeros((self.max_wb + 1, self.max_wb + 1))
        
    def fit(self, historical_draws):
        for i in range(len(historical_draws) - 1):
            current_draw = historical_draws[i]
            next_draw = historical_draws[i+1]
            for c_ball in current_draw:
                for n_ball in next_draw:
                    self.transitions[c_ball][n_ball] += 1
                    
        row_sums = self.transitions.sum(axis=1)
        row_sums[row_sums == 0] = 1 
        self.transitions = self.transitions / row_sums[:, np.newaxis]
        
    def predict(self, last_draw, max_pb=20, n_draws=1):
        predictions = []
        for _ in range(n_draws):
            pred_draw = []
            while len(pred_draw) < 7:
                base = random.choice(last_draw)
                probs = self.transitions[base]
                if probs.sum() == 0:
                    probs = np.ones(self.max_wb + 1) / (self.max_wb + 1)
                    probs[0] = 0
                
                next_ball = np.random.choice(range(self.max_wb + 1), p=probs)
                if next_ball != 0 and next_ball not in pred_draw and next_ball <= 35:
                    pred_draw.append(int(next_ball))
                    
            powerball = int(np.random.randint(1, max_pb + 1))
            predictions.append((sorted(pred_draw), powerball))
        return predictions

if __name__ == "__main__":
    print("Loading 15 years of Australian Powerball data...")
    df = pd.read_csv('powerball_15_years_results.csv')
    df = df.iloc[::-1].reset_index(drop=True)
    
    historical_wbs = []
    historical_pbs = []
    
    for _, row in df.iterrows():
        try:
            wbs = [int(x.strip()) for x in row['Winning Numbers'].split(',')]
            pb = int(row['Powerball'])
            historical_wbs.append(wbs)
            historical_pbs.append(pb)
        except Exception as e:
            continue
            
    print(f"Successfully parsed {len(historical_wbs)} historical draws.")
    
    latest_draw_wbs = historical_wbs[-1]
    latest_draw_pb = historical_pbs[-1]
    
    print(f"\nMost Recent Draw:")
    print(f"White Balls: {latest_draw_wbs}")
    print(f"Powerball: {latest_draw_pb}")
    
    print("\nTraining Markov Chain Model on 15 years of data...")
    markov = MarkovChainModel(max_white_ball=50)
    markov.fit(historical_wbs)
    
    print("Predicting the numbers for THIS WEEK based on the Markov Chain transitions...")
    prediction = markov.predict(latest_draw_wbs, max_pb=20, n_draws=1)[0]
    
    print("\n=======================================================")
    print(f"PREDICTED NUMBERS FOR THIS WEEK: {prediction[0]}")
    print(f"PREDICTED POWERBALL: {prediction[1]}")
    print("=======================================================")
    print("\n*DISCLAIMER: As mathematically proven in our project, these numbers")
    print("have the exact same 1 in 134,490,400 probability of hitting the jackpot")
    print("as any other random combination (e.g. 1, 2, 3, 4, 5, 6, 7 + PB 8).*")
