"""OHLCV ingestion from cryptocurrency exchanges via ccxt."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from loguru import logger


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    since: datetime,
    until: datetime,
    output_dir: Path,
    rate_limit_ms: int = 200,
) -> Path:
    """Fetch OHLCV candles for a single symbol and save as Parquet.

    Args:
        exchange_id: ccxt exchange id, e.g. "binance".
        symbol: Market symbol, e.g. "BTC/USDT".
        timeframe: ccxt timeframe string, e.g. "1h".
        since: Start datetime (UTC).
        until: End datetime (UTC).
        output_dir: Directory to write <symbol_safe>_<timeframe>.parquet.
        rate_limit_ms: Sleep between pagination requests.

    Returns:
        Path to the written Parquet file.
    """
    import ccxt  # noqa: PLC0415

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    since_ms = int(since.replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ms = int(until.replace(tzinfo=timezone.utc).timestamp() * 1000)

    all_candles: list[list] = []
    cursor = since_ms

    logger.info(f"Fetching {symbol} {timeframe} from {since.date()} to {until.date()} via {exchange_id}")

    while cursor < until_ms:
        candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=1000)
        if not candles:
            break
        all_candles.extend(candles)
        cursor = candles[-1][0] + 1
        time.sleep(rate_limit_ms / 1000)

    df = pl.DataFrame(
        all_candles,
        schema=["timestamp_ms", "open", "high", "low", "close", "volume"],
        orient="row",
    ).with_columns(
        pl.from_epoch("timestamp_ms", time_unit="ms").dt.cast_time_unit("us").alias("timestamp"),
    ).filter(
        pl.col("timestamp_ms") < until_ms
    ).drop("timestamp_ms")

    symbol_safe = symbol.replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol_safe}_{timeframe}.parquet"

    # Merge with existing data if present — dedup by timestamp, keep all history
    if out_path.exists():
        existing = pl.read_parquet(out_path)
        merged = pl.concat([existing, df]).unique("timestamp").sort("timestamp")
        new_rows = len(merged) - len(existing)
        logger.info(f"Merged with existing {len(existing)} rows → {len(merged)} total (+{new_rows} new)")
        df = merged

    df.write_parquet(out_path)
    logger.info(f"Wrote {len(df)} rows → {out_path}")
    return out_path
