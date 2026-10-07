# Powerball Data Science and Mathematics Project

This project provides a comprehensive data analysis, simulation, and machine learning pipeline for historical Powerball lottery data. It rigorously explores the statistical distributions, expected value, and limits of predictive modeling when applied to uniform random independent variables.

## Project Structure

1. **`module1_eda.py`**: Data ingestion and Exploratory Data Analysis (EDA). Computes frequency distributions, odd/even splits, high/low splits, and identifies hot/cold numbers.
2. **`module2_ml_models.py`**: Machine Learning models (Markov Chain and LSTM architecture) compared against a Random Baseline.
3. **`module3_monte_carlo.py`**: Simulates 10 million Powerball draws to empirically verify theoretical prize odds.
4. **`module4_expected_value.py`**: Calculates the mathematical Expected Value (EV) of a ticket, taking into account lump-sum options and the probability of splitting the jackpot (using the Poisson distribution).

## Top Predictions Pipeline (`top_predictions.py`) — recommended

Australian Powerball format: 7 from 35 + Powerball 1 from 20 (draw #1144 onwards).

No model can make a combination more likely to be drawn; every ticket is 1 in 134,490,400.
This pipeline optimises the things that *can* be optimised:

| Step | What it does | Why it helps |
|---|---|---|
| Bias test | Chi-square on per-ball counts, Monte Carlo p-value, James-Stein shrinkage toward uniform | Uses a ball bias only if the data shows one; otherwise the bias weight is ~0 |
| Anti-popularity | Penalises birthdays (≤31, ≤12), sequences, progressions, last draw's numbers, "hot" numbers, multiples of 7 | Fewer people share your jackpot or division if you win |
| Exact scoring | Enumerates and scores all 6,724,520 main combinations | No sampling noise |
| Coverage | Greedy ticket set with `--max-overlap` and balanced number usage, plus all 20 Powerballs spread across tickets | Better chance that at least one ticket wins some prize |
| Walk-forward backtest | Re-fits on past draws only and compares hits with the exact odds (z-scores) | Shows whether the model beats chance (it should not) |

### Running on DGX Spark

```bash
pip install -r requirements.txt
python extract_powerball.py                 # refresh results first
python top_predictions.py --tickets 20      # top 20 tickets -> top_predictions_draw_<N>.csv
python top_predictions.py --tickets 50 --max-overlap 2 --mc-reps 100000
python top_predictions.py --tickets 20 --backtest 300 --workers 20   # uses all cores
```

The first run builds `combos_7of35.npy` (~47 MB) and reuses it after that. Memory use is roughly 1–2 GB per backtest worker.

The old `predict_this_week.py` trained on a CSV that mixed the 5/45, 6/40 and 7/35 formats. It now filters to the 7/35 format.

## Environment Setup

To run these scripts, you will need Python 3.8+ and several data science libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Sourcing Historical Data
The `module1_eda.py` script looks for `us_powerball_historical.csv`. 
You can download the official historical data for US Powerball from Data.gov or state lottery websites (e.g., Texas Lottery or NY Lottery provide comprehensive CSV exports). Ensure the columns are formatted correctly (Date, Winning Numbers, Powerball). If no CSV is found, the script will automatically generate an unbiased dummy dataset for demonstration.

---

## Mathematical Summary: The Failure of Machine Learning on Lottery Data

In **Module 2**, we train Machine Learning models (a Markov Chain and build an LSTM Neural Network) to predict the next draw based on historical sequences.

**Result**: The ML models will completely fail to outperform the Random Baseline model (which simply picks numbers uniformly at random).

### Why does ML fail here?

Machine Learning algorithms are exceptional at finding hidden patterns, correlations, and non-linear relationships in data. However, for a physically drawn lottery like Powerball, the underlying data generation process is driven by independent, identically distributed (i.i.d.) random variables. 

1. **Independence**: Let $X_t$ be the draw at time $t$. By physical design (tumbling balls in a machine), $P(X_t | X_{t-1}, X_{t-2}, \dots) = P(X_t)$. The past history provides exactly zero mathematical information about the future.
2. **Uniformity**: Every ball has an equal probability of being selected.
3. **The Gambler's Fallacy**: Believing that a "cold" number is "due" to hit, or a "hot" number will keep hitting, is a cognitive bias known as the Gambler's Fallacy. 

When you feed i.i.d. random noise into a complex model like an LSTM or a Markov Chain, the model attempts to minimize loss by mapping noise to noise. It will either overfit to historical variance (finding "patterns" that are purely coincidental) or, with heavy regularization, it will mathematically converge back to predicting the uniform distribution—exactly matching the performance of a naive random guess. 

Therefore, any statistically significant deviation from random baseline performance in a lottery prediction model indicates either a flaw in the lottery's physical randomness (which is highly audited) or a bug in the model's evaluation logic (such as data leakage).
