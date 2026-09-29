import requests
from bs4 import BeautifulSoup
import csv

def fetch_powerball_results(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    table = soup.find('table', class_='table powerball mobFormat mobResult')
    if not table:
        print("Could not find the results table.")
        return results
        
    tbody = table.find('tbody')
    rows = tbody.find_all('tr')
    
    for row in rows:
        tds = row.find_all('td')
        if len(tds) < 3:
            continue
            
        date_info = list(tds[0].stripped_strings)
        draw_no = date_info[0].replace('Draw', '').strip() if len(date_info) > 0 else ''
        draw_date = date_info[1].strip() if len(date_info) > 1 else ''
        
        balls_ul = tds[1].find('ul', class_='balls')
        if not balls_ul:
            continue
            
        balls = balls_ul.find_all('li')
        regular_numbers = []
        powerball = None
        
        for ball in balls:
            if 'powerball' in ball.get('class', []):
                powerball = ball.text.strip()
            else:
                regular_numbers.append(ball.text.strip())
                
        total_winners = tds[2].text.strip().replace(',', '')
        results.append({
            'Draw Number': draw_no,
            'Date': draw_date,
            'Winning Numbers': ', '.join(regular_numbers),
            'Powerball': powerball,
            'Total Winners': total_winners
        })
        
    return results

if __name__ == "__main__":
    years = range(2012, 2027)  # 2012 to 2026 (last 15 years)
    all_results = []
    
    for year in sorted(years, reverse=True):
        url = f"https://australia.national-lottery.com/powerball/results-archive-{year}"
        print(f"Fetching results from {url}...")
        year_results = fetch_powerball_results(url)
        if year_results:
            all_results.extend(year_results)
            print(f"Extracted {len(year_results)} draws for {year}.")
        else:
            print(f"No results found for {year}.")
    
    if all_results:
        csv_filename = "powerball_15_years_results.csv"
        print(f"\nSuccessfully extracted {len(all_results)} total draws. Saving to {csv_filename}...")
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['Draw Number', 'Date', 'Winning Numbers', 'Powerball', 'Total Winners'])
            writer.writeheader()
            writer.writerows(all_results)
            
        print("Done!")
    else:
        print("No results found.")
