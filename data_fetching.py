import os
import pandas as pd
import yfinance as yf

def fetch_macro_data(start_date: str, end_date: str, save_path: str = 'data/raw_macro_data.csv') -> pd.DataFrame:
    """
    Fetches daily adjusted close prices for macroeconomic proxies.
    Aligns dates and handles missing values by forward-filling.
    """
    tickers = ["SPY", "^VIX", "IEF", "HYG"]
    print(f"Fetching data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data.ffill(inplace=True)  # Foward-fill missing values
    data.dropna(inplace=True)  # Drop any remaining rows with NaN values
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path)
    print(f"Data Sucessfully saved to {save_path}")

    return data

if __name__ == "__main__":
    start_date = "2010-01-01"
    end_date = "2026-01-01"
    df = fetch_macro_data(start_date, end_date)

    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print(f"\nFinal dataset shape: {df.shape} (rows, columns)")
