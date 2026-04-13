"""Fetch OHLCV candles from an exchange and save as Parquet.

Usage:
    # Full fetch
    python scripts/fetch_ohlcv.py \
        --exchange binance \
        --symbols BTC/USDT ETH/USDT \
        --start 2022-01-01 \
        --end 2025-12-31 \
        --timeframe 4h \
        --output data/raw

    # Incremental update (--start omitted → resumes from last timestamp in parquet)
    python scripts/fetch_ohlcv.py \
        --exchange binance \
        --symbols BTC/USDT ETH/USDT \
        --end 2026-04-12 \
        --timeframe 15m
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from msb_repr.data.ingestion import fetch_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OHLCV from exchange")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (omit to resume from last timestamp)")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--timeframe", default="4h")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    until = datetime.fromisoformat(args.end)

    for symbol in args.symbols:
        if args.start is not None:
            since = datetime.fromisoformat(args.start)
        else:
            symbol_safe = symbol.replace("/", "_")
            out_path = args.output / args.exchange / f"{symbol_safe}_{args.timeframe}.parquet"
            if not out_path.exists():
                raise FileNotFoundError(
                    f"No existing parquet for {symbol} at {out_path}. "
                    "Provide --start to do a full fetch."
                )
            last_ts = pl.read_parquet(out_path).select("timestamp").max().item()
            since = last_ts.replace(tzinfo=timezone.utc).replace(tzinfo=None)
            print(f"[{symbol}] Resuming from {since.date()}")

        fetch_ohlcv(
            exchange_id=args.exchange,
            symbol=symbol,
            timeframe=args.timeframe,
            since=since,
            until=until,
            output_dir=args.output / args.exchange,
        )


if __name__ == "__main__":
    main()
