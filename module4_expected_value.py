import numpy as np

def calculate_ev(advertised_jackpot, cash_multiplier=0.48, tickets_sold=50_000_000, ticket_price=2.0):
    """
    Calculates the Expected Value (EV) of a Powerball ticket.
    """
    
    prizes = {
        "Match 5": (1_000_000, 1 / 11_688_053),
        "Match 4 + PB": (50_000, 1 / 913_129),
        "Match 4": (100, 1 / 36_525),
        "Match 3 + PB": (100, 1 / 14_494),
        "Match 3": (7, 1 / 579),
        "Match 2 + PB": (7, 1 / 701),
        "Match 1 + PB": (4, 1 / 91),
        "Match 0 + PB": (4, 1 / 38)
    }
    
    jackpot_odds = 1 / 292_201_338
    
    non_jackpot_ev = sum([prize * prob for prize, prob in prizes.values()])
    cash_jackpot = advertised_jackpot * cash_multiplier
    lambda_param = tickets_sold * jackpot_odds
    
    if lambda_param > 0:
        expected_jackpot_share = cash_jackpot * (1 - np.exp(-lambda_param)) / lambda_param
    else:
        expected_jackpot_share = cash_jackpot
        
    jackpot_ev = expected_jackpot_share * jackpot_odds
    total_ev = non_jackpot_ev + jackpot_ev
    net_ev = total_ev - ticket_price
    
    return net_ev, total_ev, non_jackpot_ev, expected_jackpot_share

def find_breakeven_jackpot():
    """
    Finds the advertised jackpot size where the EV of a $2 ticket becomes strictly positive.
    """
    print("--- Searching for Break-Even Jackpot ---")
    jackpots = np.linspace(100_000_000, 2_000_000_000, 20)
    
    found_positive = False
    for j in jackpots:
        sim_sales = 15_000_000 + max(0, (j - 100_000_000) / 10)
        net_ev, total_ev, _, expected_share = calculate_ev(j, tickets_sold=sim_sales)
        print(f"Jackpot: ${j:,.0f} | Est. Sales: {sim_sales:,.0f} | EV: ${total_ev:.4f} | Net Profit/Loss: ${net_ev:.4f}")
        
        if net_ev > 0 and not found_positive:
            print(f"*** POSITIVE EV REACHED at approx ${j:,.0f} Jackpot! ***\n")
            found_positive = True

if __name__ == "__main__":
    print("=== Powerball Expected Value Analysis ===")
    current_jackpot = 500_000_000
    est_sales = 40_000_000
    net_ev, total_ev, non_jp_ev, split_share = calculate_ev(current_jackpot, tickets_sold=est_sales)
    
    print(f"Advertised Jackpot: ${current_jackpot:,.0f}")
    print(f"Estimated Sales: {est_sales:,}")
    print(f"Non-Jackpot EV component: ${non_jp_ev:.4f}")
    print(f"Expected Share if won (after splits & cash option): ${split_share:,.0f}")
    print(f"Total Ticket EV: ${total_ev:.4f}")
    print(f"Net Expected Value (Cost $2): ${net_ev:.4f}\n")
    
    find_breakeven_jackpot()
