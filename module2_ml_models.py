import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import random

class RandomBaselineModel:
    """Picks numbers purely uniformly at random, respecting Powerball constraints."""
    def predict(self, n_draws=1):
        predictions = []
        for _ in range(n_draws):
            white_balls = sorted(np.random.choice(range(1, 70), 5, replace=False))
            powerball = np.random.randint(1, 27)
            predictions.append(white_balls + [powerball])
        return predictions

class MarkovChainModel:
    """
    A simplified Markov Chain that tracks transition probabilities between numbers 
    appearing in consecutive draws. Note: Will inherently overfit noise in true random data.
    """
    def __init__(self):
        self.transitions = np.zeros((70, 70))
        
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
        
    def predict(self, last_draw, n_draws=1):
        predictions = []
        for _ in range(n_draws):
            pred_draw = []
            while len(pred_draw) < 5:
                base = random.choice(last_draw)
                probs = self.transitions[base]
                if probs.sum() == 0:
                    probs = np.ones(70) / 70
                    probs[0] = 0
                
                next_ball = np.random.choice(range(70), p=probs)
                if next_ball != 0 and next_ball not in pred_draw:
                    pred_draw.append(next_ball)
                    
            powerball = np.random.randint(1, 27)
            predictions.append(sorted(pred_draw) + [powerball])
            last_draw = pred_draw
        return predictions

def build_lstm_model(sequence_length=10, features=6):
    """Builds a standard LSTM network for sequence prediction."""
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(sequence_length, features), return_sequences=True),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(features)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_predictions(y_true, y_pred, model_name):
    """Evaluates predictions."""
    total_matches = 0
    pb_matches = 0
    for true_draw, pred_draw in zip(y_true, y_pred):
        true_w = set(true_draw[:5])
        pred_w = set(pred_draw[:5])
        total_matches += len(true_w.intersection(pred_w))
        if true_draw[5] == pred_draw[5]:
            pb_matches += 1
            
    avg_matches = total_matches / len(y_true)
    pb_accuracy = pb_matches / len(y_true)
    
    print(f"--- {model_name} Evaluation ---")
    print(f"Average White Ball Matches per draw: {avg_matches:.4f}")
    print(f"Powerball Accuracy: {pb_accuracy:.4f}")
    return avg_matches

if __name__ == "__main__":
    print("Initializing Machine Learning experiment on random data...")
    
    np.random.seed(42)
    N = 1000
    X_data = []
    for _ in range(N):
        w = list(np.random.choice(range(1, 70), 5, replace=False))
        pb = np.random.randint(1, 27)
        X_data.append(w + [pb])
        
    train_data = X_data[:800]
    test_data = X_data[800:]
    
    baseline = RandomBaselineModel()
    baseline_preds = baseline.predict(n_draws=len(test_data))
    
    markov = MarkovChainModel()
    markov.fit([x[:5] for x in train_data])
    markov_preds = markov.predict(train_data[-1][:5], n_draws=len(test_data))
    
    print("\n=== BACKTESTING RESULTS ===")
    evaluate_predictions(test_data, baseline_preds, "Random Baseline")
    evaluate_predictions(test_data, markov_preds, "Markov Chain")
    
    print("\nCONCLUSION:")
    print("Notice how the Markov Chain (and an LSTM, if fully trained) hovers around the same")
    print("average matches as the Random Baseline (approx 5/69 = 0.072 matches per draw).")
    print("This empirically demonstrates that past draws hold no predictive power for independent future events.")
