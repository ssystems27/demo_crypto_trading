import time
import pandas as pd
import numpy as np
import requests
from binance.client import Client
from dataclasses import dataclass


@dataclass
class Config:
    api_key: str
    api_secret: str
    symbol: str
    timeframe: str  # e.g., '5m'
    vwap_period: int
    buy_threshold: float
    sell_threshold: float
    fee_rate: float  # 0.1% = 0.001
    trade_allocation: float  # fraction of current USDC balance
    initial_balance: float  # starting paper USDC
    telegram_token: str
    telegram_chat_id: str
    poll_interval: int  # seconds to wait between polls (e.g. 300 for 5min)


# =============================================================================
# 1) CONFIGURE YOUR SETTINGS
# =============================================================================
CONFIG = Config(
    api_key="Key",
    api_secret="secret",
    symbol="IOUSDC",  # Example: IO paired with USDC
    timeframe=Client.KLINE_INTERVAL_5MINUTE,
    vwap_period=48,
    buy_threshold=-1.1,
    sell_threshold=0.7,
    fee_rate=0.001,
    trade_allocation=0.4,
    initial_balance=10000.0,
    telegram_token="token",
    telegram_chat_id="id",
    poll_interval=300  # 5 minutes
)

# =============================================================================
# 2) SET UP BINANCE CLIENT & TELEGRAM
# =============================================================================

client = Client(CONFIG.api_key, CONFIG.api_secret)


def send_telegram_message(token: str, chat_id: str, msg_text: str):
    """ Sends a Telegram message. """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": msg_text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, data=payload)
        if resp.status_code != 200:
            print(f"[Telegram Warning] {resp.text}")
    except Exception as e:
        print(f"[Telegram Error] {e}")


# =============================================================================
# 3) DATA FETCHING (REAL BINANCE)
# =============================================================================

def fetch_historical_klines(symbol: str, interval: str, limit=200):
    """
    Fetch the latest 'limit' candles from Binance for the given symbol & interval.
    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    # Up to 1000 klines allowed. We'll default to 200 for rolling calculations
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

    df = pd.DataFrame(klines, columns=[
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "qav", "num_trades", "taker_base_vol",
        "taker_quote_vol", "ignore"
    ])
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


# =============================================================================
# 4) INDICATOR: VWAP Z-SCORE
# =============================================================================

def compute_vwap_zscore(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """ Appends rolling VWAP, rolling stdev, then zscore_{period}. """
    df = df.copy()  # so we don't mutate the original outside
    df["cum_vol_price"] = df["Close"] * df["Volume"]
    df["cum_vol"] = df["Volume"]

    # Rolling sums
    df["mean"] = (
            df["cum_vol_price"].rolling(window=period).sum() /
            df["cum_vol"].rolling(window=period).sum()
    )

    df["price_diff_squared"] = (df["Close"] - df["mean"]) ** 2
    df["vwapsd"] = np.sqrt(df["price_diff_squared"].rolling(window=period).mean())

    zscore_col = f"zscore_{period}"
    df[zscore_col] = (df["Close"] - df["mean"]) / df["vwapsd"]
    return df


# =============================================================================
# 5) LIVE PAPER TRADING LOOP (REAL DATA, NO ACTUAL ORDERS)
# =============================================================================

def live_paper_trading_loop(config: Config):
    """
    Continuously poll Binance for the latest data,
    compute signals, place "paper trades" in memory,
    log/Telegram results, and do NOT place real orders on exchange.
    """
    print("[INFO] Starting real-time Paper Trading simulation.")

    # Paper trading state
    balance_usdc = config.initial_balance
    position_io = 0.0
    position_cost = 0.0

    # Keep a local DataFrame to store candles
    df_candles = pd.DataFrame()

    # We'll run an infinite loop (until you stop the script)
    while True:
        try:
            # 1) Fetch ~4x the period so rolling sums can be calculated
            #    e.g., if period=48, fetch 200 candles
            new_data = fetch_historical_klines(config.symbol, config.timeframe, limit=4 * config.vwap_period)

            # 2) Replace df_candles with the new_data (or you can do merges if you prefer)
            df_candles = new_data.copy()
            df_candles = compute_vwap_zscore(df_candles, config.vwap_period)

            zscore_col = f"zscore_{config.vwap_period}"
            # Drop incomplete rolling data
            df_candles.dropna(subset=[zscore_col], inplace=True)
            if df_candles.empty or len(df_candles) < 2:
                print("[INFO] Not enough data for a valid Z-Score. Sleeping.")
                time.sleep(config.poll_interval)
                continue

            # 3) Identify the last two rows for threshold cross detection
            last_row = df_candles.iloc[-1]
            prev_row = df_candles.iloc[-2]

            current_zscore = last_row[zscore_col]
            prev_zscore = prev_row[zscore_col]
            current_price = last_row["Close"]
            current_time = last_row["Date"]

            # Debug: Print info
            print(f"[DEBUG] Time={current_time}, Close={current_price:.4f}, Zscore={current_zscore:.2f}")

            # 4) SELL Logic
            if position_io > 0.0:
                if (prev_zscore < config.sell_threshold) and (current_zscore >= config.sell_threshold):
                    # "paper sell"
                    raw_proceeds = position_io * current_price
                    sell_fee = raw_proceeds * config.fee_rate
                    net_proceeds = raw_proceeds - sell_fee
                    profit = net_proceeds - position_cost
                    balance_usdc += net_proceeds

                    msg_text = (
                        f"*PAPER SELL*\n"
                        f"Time: {current_time}\n"
                        f"Symbol: {config.symbol}\n"
                        f"Sell Price: `{current_price:.4f}`\n"
                        f"Amount: `{position_io:.4f}` IO\n"
                        f"Gross Proceeds: `{raw_proceeds:.2f}` USDC\n"
                        f"Fee: `{sell_fee:.2f}` USDC\n"
                        f"Profit: `{profit:.2f}` USDC\n"
                        f"Balance Now: `{balance_usdc:.2f}` USDC"
                    )
                    print(msg_text)
                    send_telegram_message(config.telegram_token, config.telegram_chat_id, msg_text)

                    # reset position
                    position_io = 0.0
                    position_cost = 0.0

            else:
                # 5) BUY Logic
                if (prev_zscore > config.buy_threshold) and (current_zscore <= config.buy_threshold):
                    allocation = config.trade_allocation * balance_usdc
                    buy_fee = allocation * config.fee_rate
                    net_allocation = allocation - buy_fee

                    amount_io = net_allocation / current_price
                    position_io = amount_io
                    position_cost = allocation
                    balance_usdc -= allocation

                    msg_text = (
                        f"*PAPER BUY*\n"
                        f"Time: {current_time}\n"
                        f"Symbol: {config.symbol}\n"
                        f"Buy Price: `{current_price:.4f}`\n"
                        f"Amount: `{amount_io:.4f}` IO\n"
                        f"USDC Spent: `{allocation:.2f}`\n"
                        f"Fee: `{buy_fee:.2f}` USDC\n"
                        f"Balance Now: `{balance_usdc:.2f}` USDC"
                    )
                    print(msg_text)
                    send_telegram_message(config.telegram_token, config.telegram_chat_id, msg_text)

        except Exception as e:
            err_msg = f"[ERROR] {e}"
            print(err_msg)
            send_telegram_message(config.telegram_token, config.telegram_chat_id, err_msg)

        # 6) Sleep until next candle
        time.sleep(config.poll_interval)


# =============================================================================
# 6) MAIN
# =============================================================================
if __name__ == "__main__":
    live_paper_trading_loop(CONFIG)
