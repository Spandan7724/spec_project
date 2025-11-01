import itertools
import yfinance as yf
import pandas as pd
from tqdm import tqdm

# List of top-10 most traded currency ISO codes (for example)
currencies = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY", "HKD", "NZD"]

def generate_fx_tickers(currencies):
    tickers = []
    for base, quote in itertools.permutations(currencies, 2):
        tickers.append(f"{base}{quote}=X")
    return tickers

def test_tickers(tickers, start="2020-01-01", end=None, interval="1d"):
    results = []
    for ticker in tqdm(tickers):
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            success = not df.empty
            last_date = df.index.max() if success else None
            results.append({'ticker': ticker, 'success': success, 'last_date': last_date})
        except Exception as e:
            results.append({'ticker': ticker, 'success': False, 'error': str(e)})
    return pd.DataFrame(results)

# Generate all pairs
tickers = generate_fx_tickers(currencies)
print(f"Total tickers to test: {len(tickers)}")

# Test
df_results = test_tickers(tickers)
print(df_results.head(20))
# # Optionally save:
# df_results.to_csv("fx_ticker_test_results.csv", index=False)
