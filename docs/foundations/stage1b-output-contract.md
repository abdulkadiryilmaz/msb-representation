# Stage 1B Output Contract

**Status**: Draft v1
**Scope**: Stage 1B signal contract for downstream Stage 2 actionability work
**Primary selected baseline**: `stage1b_h8_predictor_v1a_z_fused`

## Purpose

Stage 1B answers a forward-structure question:

> Is a close-confirmed structural break likely within the near horizon, and in which direction?

Stage 1B does not produce a TradePlan. It does not decide entry, stop, target, leverage, sizing, or expected PnL.

The role of this contract is to define what Stage 1B can hand to Stage 2.

## Selected Horizon

Current selected primary horizon:

```text
H8 = 8 bars = approximately 2 hours on 15m candles
```

Rationale:

- H4 is highly readable but too close to immediate trigger / proximity behavior.
- H16 is useful but more break-heavy.
- H8 is the cleanest current compromise between near-term actionability and forward structure.

## Core Output

The minimal Stage 1B output is:

```text
Stage1BSignal:
  horizon: H8
  side: long | short | none
  p_break: float
  p_direction: float
  break_confidence: float
  direction_confidence: float
```

Definitions:

| Field | Meaning |
|---|---|
| `horizon` | Forecast horizon. Current primary value: `H8`. |
| `side` | Direction implied by the predictor if `p_break` crosses threshold. |
| `p_break` | Probability-like score for break occurrence within H8. |
| `p_direction` | Probability-like score for predicted direction given break. |
| `break_confidence` | `p_break` if break is predicted, otherwise `1 - p_break`. |
| `direction_confidence` | Confidence of the selected bullish/bearish direction head. |

Default decision threshold:

```text
p_break >= 0.50 => break candidate
```

Threshold sweeps did not produce robust test improvement over `0.50`.

## Context Required By Stage 2

Stage 2 cannot use the signal alone. It also needs price/structure context:

```text
Stage1BContext:
  current_price
  bull_level
  bear_level
  effective_break_pct
  distance_to_bull_level
  distance_to_bear_level
  predicted_side_distance_to_level
  time_horizon_bars
  symbol
  timestamp
```

Reason:

```text
break + direction != entry
```

Example:

```text
current_price = 10.00
predicted side = long
bull_level = 11.00
```

This does not mean "buy at 10.00". It means a bullish structural break may become actionable around the relevant break level or a later retest/confirmation condition.

## Confidence Policy

H8 error audit showed break confidence is meaningful:

| Break confidence | Test accuracy |
|---|---:|
| 0.50-0.60 | 0.4893 |
| 0.60-0.70 | 0.5714 |
| 0.70-0.80 | 0.7347 |
| 0.80-0.90 | 0.8437 |
| 0.90-1.00 | 0.9532 |

Initial Stage 2 work should treat low-confidence and high-confidence signals differently.

Suggested initial buckets:

```text
low_confidence:    break_confidence < 0.70
medium_confidence: 0.70 <= break_confidence < 0.90
high_confidence:   break_confidence >= 0.90
```

High-confidence signals are the best candidates for the first Stage 2 actionability experiments.

## What Stage 1B Does Not Decide

Stage 1B does not decide:

- market entry now
- limit/stop entry level
- retest requirement
- stop-loss level
- target level
- expected risk/reward
- position size
- no-trade vs actionable trade

These belong to Stage 2 and the final TradePlan contract.

## Stage 2 Bridge Question

Given:

```text
Stage1BSignal + Stage1BContext
```

Stage 2 should answer:

```text
no_trade
wait_for_break
wait_for_retest
actionable_break_candidate
```

Only after this actionability layer is validated should the system move toward:

```text
entry
stop
target
expected_rr
confidence
```

## Current Selected Baseline

```text
stage1b_h8_predictor_v1a_z_fused
```

Key test metrics:

| Metric | Value |
|---|---:|
| reconstructed 3-class macro F1 | 0.7307 |
| break occurrence macro F1 | 0.7497 |
| direction-given-break macro F1 | 0.8955 |

Decision:

Use this as the primary Stage 1B baseline for the Stage 2 bridge.
