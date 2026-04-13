# Feature Reference

16 channels fed to the short autoencoder (`FEATURE_COLUMNS`), 7 channels fed to the long trend autoencoder (`TREND_FEATURE_COLUMNS`).
All features are z-score normalised at training time (see `Normalizer`).

## Price Dynamics

| Channel | Formula | Notes |
|---|---|---|
| `log_return` | log(close / prev_close) | Close-to-close log return; primary price signal |
| `open_close_return` | log(close / open) | Intrabar return; distinguishes gap opens from intraday moves |

## Volatility / Range

| Channel | Formula | Notes |
|---|---|---|
| `hl_range` | (high − low) / close | Per-bar normalised range; instantaneous volatility |
| `atr_norm` | ATR(14) / close | 14-bar smoothed true range normalised by close; session-consistent volatility |

ATR true range = max(high−low, \|high−prev_close\|, \|low−prev_close\|). First bar uses high−low as fallback (no prev_close).

## VWAP & Mean-Reversion

| Channel | Formula | Notes |
|---|---|---|
| `vwap_dev` | close / rolling_vwap(bars_per_day) − 1 | Deviation from 1-day rolling VWAP |
| `bb_position` | (close − lower_band) / (upper_band − lower_band) | Position within 20-bar Bollinger bands (2σ); 0 = lower, 1 = upper |
| `price_position` | (close − rolling_min(20)) / (rolling_max(20) − rolling_min(20)) | Position in 20-bar price range; complements bb_position with raw highs/lows |

## Volume

| Channel | Formula | Notes |
|---|---|---|
| `volume_zscore` | (volume − rolling_mean(7d)) / rolling_std(7d) | Volume anomaly relative to 7-day history |
| `volume_ratio` | volume / rolling_max(volume, 20) | Volume as fraction of recent peak; highlights spikes differently than z-score |

Both volume windows scale with `bars_per_day` for `volume_zscore`; `volume_ratio` uses a fixed 20-bar window.

---

## Trend AE Feature Columns (7 channels — `TREND_FEATURE_COLUMNS`)

Used with 288-bar (72h) long windows. Deliberately excludes `rsi` and `atr_norm` (already in short AE) — uses slower variants instead to avoid redundancy with `z_short`.

| Channel | Formula | Notes |
|---|---|---|
| `close_norm` | log(close / close[-bars_per_day*3]) | 3-day log return; encodes medium-term trend direction |
| `ema_slope` | log(EMA(bars_per_day)).diff() | 1-day EMA log-return; smoothed trend momentum |
| `rsi_42` | RSI(42) | Slow RSI (~10.5h at 15m); fewer oscillations over 288-bar window than RSI(14) |
| `atr_norm_28` | ATR(28) / close | Slower volatility regime indicator vs ATR(14) in short AE |
| `bb_position_50` | (close − lower_50) / (upper_50 − lower_50) | Position within 50-bar Bollinger bands; structural price level over 72h |
| `macd_hist` | MACD(12,26) − signal(9) | Trend acceleration / momentum divergence |

## Momentum

| Channel | Formula | Notes |
|---|---|---|
| `rsi` | RSI(14) | 14-bar RSI; overbought/oversold signal |
| `macd` | EMA(12) − EMA(26) | MACD line |
| `macd_signal` | EMA(macd, 9) | MACD signal line; crossovers indicate momentum shifts |

## Time Encoding (Cyclical)

Encodes the position of each bar within the day and week using sin/cos pairs so that temporal proximity is preserved (e.g. Sunday and Monday are close, not far apart).

| Channel | Formula | Notes |
|---|---|---|
| `hour_sin` | sin(2π × hour / 24) | UTC hour of bar open |
| `hour_cos` | cos(2π × hour / 24) | UTC hour of bar open |
| `dow_sin` | sin(2π × (weekday−1) / 7) | ISO weekday, Mon=1..Sun=7, shifted to 0-indexed |
| `dow_cos` | cos(2π × (weekday−1) / 7) | ISO weekday |

For 4h candles the UTC hours are always 0, 4, 8, 12, 16, 20 — capturing London open (08:00), NY afternoon (16:00), and thin overnight sessions. Day-of-week captures Friday expiry dynamics and weekend illiquidity.

## Scaling with timeframe

`bars_per_day` controls the rolling windows for `vwap_dev` and `volume_zscore` so they represent the same calendar duration across timeframes (4h, 1h, 1d). All other windows are fixed in bars.
