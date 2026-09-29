import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(filepath='powerball_data.csv'):
    """
    Loads Powerball historical data.
    Assumes a CSV with columns: Date, Draw, Winning Numbers, Powerball
    For US Powerball (1-69 White, 1-26 Red).
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"File {filepath} not found. Returning a dummy dataset for demonstration.")
        return get_dummy_data()
        
    return df

def get_dummy_data(n_rows=500):
    """Generates dummy US Powerball data if no CSV is provided."""
    np.random.seed(42)
    dates = pd.date_range(start='2015-10-07', periods=n_rows, freq='W-WED')
    white_balls = [list(np.random.choice(range(1, 70), 5, replace=False)) for _ in range(n_rows)]
    powerballs = np.random.randint(1, 27, n_rows)
    
    df = pd.DataFrame({
        'Date': dates,
        'W1': [w[0] for w in white_balls],
        'W2': [w[1] for w in white_balls],
        'W3': [w[2] for w in white_balls],
        'W4': [w[3] for w in white_balls],
        'W5': [w[4] for w in white_balls],
        'Powerball': powerballs
    })
    return df

def perform_eda(df):
    """Performs Exploratory Data Analysis on the dataset."""
    print("--- Exploratory Data Analysis ---")
    
    white_balls = pd.concat([df['W1'], df['W2'], df['W3'], df['W4'], df['W5']])
    wb_counts = white_balls.value_counts().sort_index()
    pb_counts = df['Powerball'].value_counts().sort_index()
    
    print("\nTop 5 Hot White Balls:")
    print(white_balls.value_counts().head())
    print("\nTop 5 Cold White Balls:")
    print(white_balls.value_counts().tail())
    
    odd_even_split = white_balls.apply(lambda x: 'Odd' if x % 2 != 0 else 'Even').value_counts(normalize=True)
    print(f"\nWhite Balls Odd/Even Split:\n{odd_even_split}")
    
    high_low_split = white_balls.apply(lambda x: 'Low' if x <= 34 else 'High').value_counts(normalize=True)
    print(f"\nWhite Balls High/Low Split:\n{high_low_split}")
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    sns.barplot(x=wb_counts.index, y=wb_counts.values, color='skyblue')
    plt.title('Frequency Distribution of White Balls (1-69)')
    plt.xlabel('Ball Number')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 1, 2)
    sns.barplot(x=pb_counts.index, y=pb_counts.values, color='salmon')
    plt.title('Frequency Distribution of Powerball (1-26)')
    plt.xlabel('Powerball Number')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('frequency_distributions.png')
    print("\nSaved frequency distributions plot to 'frequency_distributions.png'")

if __name__ == "__main__":
    df = load_and_preprocess_data('us_powerball_historical.csv')
    perform_eda(df)
