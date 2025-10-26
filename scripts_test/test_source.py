# test_source_fixed.py
import os
import datetime
import traceback
import requests
from dotenv import load_dotenv
import yfinance as yf

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------
load_dotenv()
ERH_API_KEY = os.getenv("EXCHANGE_RATE_HOST_API_KEY")
if not ERH_API_KEY:
    raise RuntimeError("Missing EXCHANGE_RATE_HOST_API_KEY in .env")

def iso_utc_from_unix(ts):
    if ts is None:
        return None
    return datetime.datetime.fromtimestamp(int(ts), datetime.UTC).replace(microsecond=0).isoformat()

# -------------------------------------------------------------------
# Providers
# -------------------------------------------------------------------
def fetch_exchangerate_host(symbols: list[str]):
    """
    Fetch live rates from ExchangeRate.host.
    Free plan: source/base currency is fixed to USD.
    Docs: https://exchangerate.host/documentation
    """
    url = "https://api.exchangerate.host/live"
    params = {
        "access_key": ERH_API_KEY,
        "currencies": ",".join(symbols)
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    if not j.get("success", False):
        raise RuntimeError(f"ExchangeRate.host error: {j}")

    return {
        "provider": "exchangerate_host",
        "base": j.get("source"),   # default USD for free plan
        "timestamp": j.get("timestamp"),
        "quotes": j.get("quotes", {})
    }

def fetch_yfinance_snapshot(base: str, quote: str):
    ticker = f"{base}{quote}=X"
    tk = yf.Ticker(ticker)

    last_price = None
    unix_time = None
    bid = None
    ask = None

    try:
        fi = tk.fast_info
        last_price = getattr(fi, "last_price", None)
        unix_time = getattr(fi, "last_price_time", None)
        bid = getattr(fi, "bid", None)
        ask = getattr(fi, "ask", None)
    except Exception:
        pass

    try:
        info = tk.info
        if last_price is None:
            last_price = info.get("regularMarketPrice")
        if unix_time is None:
            unix_time = info.get("regularMarketTime")
        if bid is None:
            bid = info.get("bid")
        if ask is None:
            ask = info.get("ask")
    except Exception:
        pass

    return {
        "provider": "yfinance",
        "symbol": ticker,
        "regularMarketPrice": last_price,
        "bid": bid,
        "ask": ask,
        "regularMarketTime": unix_time
    }

# -------------------------------------------------------------------
# Normalization
# -------------------------------------------------------------------
def normalized_erh(quote: str, erh_obj: dict):
    pair = f"{erh_obj['base']}/{quote}"
    mid = erh_obj["quotes"].get(erh_obj["base"] + quote)
    ts_iso = iso_utc_from_unix(erh_obj["timestamp"])
    return {
        "currency_pair": pair,
        "rate_timestamp": ts_iso,
        "mid_rate": mid,
        "provider": "exchangerate_host",
        "provider_raw": erh_obj
    }

def normalized_yf(base: str, quote: str, yf_obj: dict):
    bid = yf_obj.get("bid")
    ask = yf_obj.get("ask")
    price = yf_obj.get("regularMarketPrice")
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
        spread = ask - bid
    else:
        mid = price
        spread = None
    ts_iso = iso_utc_from_unix(yf_obj.get("regularMarketTime")) if yf_obj.get("regularMarketTime") else None
    return {
        "currency_pair": f"{base}/{quote}",
        "rate_timestamp": ts_iso,
        "mid_rate": mid,
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "provider": "yfinance",
        "provider_raw": yf_obj
    }

# -------------------------------------------------------------------
# Demo / Verify
# -------------------------------------------------------------------
def test_providers():
    base = "USD"   # ExchangeRate.host free tier always USD base
    quote = "EUR"

    print(f"=== ExchangeRate.host -> {base}/{quote} ===")
    try:
        erh = fetch_exchangerate_host([quote])
        snap_erh = normalized_erh(quote, erh)
        print(snap_erh)
    except Exception as e:
        print("ExchangeRate.host failed:", e)
        traceback.print_exc()

    print(f"\n=== yfinance -> {base}/{quote} ===")
    try:
        yfres = fetch_yfinance_snapshot(base, quote)
        snap_yf = normalized_yf(base, quote, yfres)
        print(snap_yf)
    except Exception as e:
        print("yfinance failed:", e)
        traceback.print_exc()

if __name__ == "__main__":
    test_providers()
