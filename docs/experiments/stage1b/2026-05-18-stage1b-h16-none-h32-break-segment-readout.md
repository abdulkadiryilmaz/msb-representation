# Stage 1B H16=None / H32=Break Segment Readout

**Date**: 2026-05-18
**Status**: completed
**Source checkpoint**: `stage1b_h16_h32_multihorizon_v1b_z_fused/best.pt`
**Analysis artifact**: `data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused/segment_h16_none_h32_break.json`

## Goal

V1b sonrası şu segmenti ayrıca incelemek:

```text
H16 = none
H32 = bullish / bearish
```

Bu segment, hemen kırılmayan ama daha geniş intraday horizon içinde yönlü break'e dönüşen örnekleri temsil eder.

Araştırma sorusu:

> Model, yakın horizon'da no-break kalırken daha uzak horizon'da yönlü yapıya olgunlaşan örnekleri ayırt edebiliyor mu?

## Segment Definition

True segment:

```text
true h16_future_break_direction = none
true h32_future_break_direction in {bullish, bearish}
```

Predicted segment:

```text
p_break_16 < 0.50
p_break_32 >= 0.50
```

## Results

| Split | Samples | True segment | True rate | Pred segment | Pred rate | Precision | Recall | Segment F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 8,830 | 1,117 | 0.1265 | 2,096 | 0.2374 | 0.2223 | 0.4172 | 0.2901 |
| test | 9,671 | 1,317 | 0.1362 | 2,558 | 0.2645 | 0.2451 | 0.4761 | 0.3236 |

Inside true segment:

| Split | H16 no-break recall | H32 break recall | H32 direction accuracy | H32 direction macro F1 |
|---|---:|---:|---:|---:|
| val | 0.5148 | 0.9024 | 0.5452 | 0.5452 |
| test | 0.5680 | 0.9081 | 0.5429 | 0.5416 |

Mean break probabilities inside true segment:

| Split | mean `p_break_16` | mean `p_break_32` |
|---|---:|---:|
| val | 0.5340 | 0.6964 |
| test | 0.5103 | 0.6807 |

H32 direction confusion inside true test segment:

Labels: `bullish / bearish`

```text
bullish -> [323, 265]
bearish -> [337, 392]
```

Direction class counts inside true test segment:

```text
true: bullish=588, bearish=729
pred: bullish=660, bearish=657
```

Majority baseline inside true test segment:

| Metric | Value |
|---|---:|
| majority accuracy | 0.5535 |
| majority macro F1 | 0.3563 |

The model's direction macro F1 is above majority macro F1, but accuracy is slightly below majority accuracy because the model predicts both directions more evenly.

## Symbol Check

Test split:

| Symbol | Samples | True segment | Pred segment | Segment recall |
|---|---:|---:|---:|---:|
| BTC_USDT_15m | 2,439 | 342 | 699 | 0.4942 |
| ETH_USDT_15m | 2,355 | 314 | 662 | 0.4936 |
| SOL_USDT_15m | 2,439 | 316 | 591 | 0.4462 |
| XRP_USDT_15m | 2,438 | 345 | 606 | 0.4696 |

No single symbol dominates the segment behavior. The weakness is structural/model-level, not obviously a one-symbol artifact.

## Interpretation

This segment exists and is not rare: roughly 13.6% of test samples are `H16=none, H32=break`.

However, v1b does not yet model it cleanly:

- H32 break recall inside the true segment is high (`0.9081`).
- H16 no-break recall inside the true segment is only moderate (`0.5680`).
- Predicted delayed segment precision is low (`0.2451`).
- H32 direction in this segment is weak (`0.5416` macro F1).

This means H32 currently acts more like a broad "break likely sometime within 32 bars" signal than a clean early directional maturation signal.

## Decision

Do not treat `H16=none, H32=bullish/bearish` as TradePlan-ready early setup signal yet.

Use it as a diagnostic segment for the next Stage 1B iteration.

Next options:

1. Add a direct delayed-setup auxiliary target:

```text
delayed_break_16_32 =
  h16 == none and h32 in {bullish, bearish}
```

2. Evaluate whether H32 needs a different break threshold or calibration from H16.
3. Inspect examples from high-confidence predicted delayed segments to see whether labels or horizon definitions are too permissive.

## Artifacts

```text
scripts/analyze_stage1b_segments.py
data/stage1b/binance/15m/checkpoints/stage1b_h16_h32_multihorizon_v1b_z_fused/segment_h16_none_h32_break.json
```
