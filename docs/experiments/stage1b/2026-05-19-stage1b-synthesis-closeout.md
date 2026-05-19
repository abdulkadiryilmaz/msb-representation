# Stage 1B Synthesis / Close-Out

**Date**: 2026-05-19
**Status**: completed
**Selected Stage 1A source**: `ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`
**Selected Stage 1B baseline**: `stage1b_h8_predictor_v1a_z_fused`
**Output contract**: `docs/foundations/stage1b-output-contract.md`

## Purpose

Bu belge Stage 1B çalışmalarını kapatır ve Stage 2 actionability / TradePlan bridge çalışmasına hangi çıktının devredileceğini tanımlar.

Stage 1B'nin program içindeki rolü:

```text
Stage 1A -> current structure representation
Stage 1B -> near-term forward structure forecast
Stage 2  -> actionability / quality / no_trade vs candidate
```

Stage 1B TradePlan üretmez. Entry, stop, target, risk/reward veya position sizing kararları Stage 2 ve sonrasına aittir.

## Main Question

Stage 1B şu soruyu test etti:

> Stage 1A temsilinden yakın gelecekteki structural break yönü ve oluşma ihtimali okunabilir mi?

Kısa cevap:

```text
Evet, özellikle H8 horizon için okunabilir.
```

## Work Completed

### 1. Forward Label Contract

Label ailesi:

```text
future_break_direction_H
```

Label değerleri:

```text
none / bullish / bearish / ambiguous / insufficient_future
```

İlk label seti H16/H32/H48 idi. Sonra H16'nın break-heavy olduğu görülünce H4/H8/H16 kısa horizon seti üretildi.

Label audit sonucu:

- horizon consistency temiz
- short break -> long none yok
- direction flip yok
- horizon arttıkça break rate doğal olarak artıyor

### 2. Transfer Probes

Selected Stage 1A latent'inden forward label probe sonuçları:

| Target | Linear macro F1 | MLP macro F1 |
|---|---:|---:|
| H4 | 0.7494 | 0.7752 |
| H8 | 0.7233 | 0.7284 |
| H16 | 0.6768 | 0.6712 |
| H32 | 0.6183 | 0.6020 |
| H48 | 0.5807 | 0.5328 |

Sonuç:

Stage 1A temsilinde near-term forward structure sinyali açık biçimde var. Horizon uzadıkça sinyal zayıflıyor.

### 3. Predictor Experiments

| Predictor | Status | Key result |
|---|---|---|
| H16 v1a | accepted baseline | test reconstructed macro F1 `0.6731` |
| H16 + H32 v1b | accepted diagnostic | H16 test reconstructed macro F1 `0.6765`; H32 weak as early setup |
| H8 v1a | selected baseline | test reconstructed macro F1 `0.7307` |
| H8 + H16 v1b | rejected | H8 dropped to `0.7270`; equal-weight H16 auxiliary did not help |

Selected predictor:

```text
stage1b_h8_predictor_v1a_z_fused
```

Test metrics:

| Metric | Value |
|---|---:|
| reconstructed 3-class macro F1 | 0.7307 |
| break occurrence macro F1 | 0.7497 |
| direction-given-break macro F1 | 0.8955 |

## Why H8 Was Selected

H4 has the highest probe score, but it is very close to immediate trigger / current proximity behavior.

H16 is useful, but it is already break-heavy:

```text
H16 test break rate = 0.6871
```

H8 is the best current compromise:

```text
H8 test break rate = 0.5526
H8 predictor test reconstructed macro F1 = 0.7307
```

Interpretation:

- H8 is near enough to be relevant for actionable break candidates.
- H8 is not as ultra-short as H4.
- H8 is less break-heavy and cleaner than H16.

## Error Audit Findings

H8 predictor v1a test error profile:

| Segment | Support |
|---|---:|
| correct | 7,104 |
| false positive break | 908 |
| false negative break | 1,514 |
| wrong direction | 169 |

Main bottleneck:

```text
break / no-break and timing
```

Not main bottleneck:

```text
direction-given-break
```

Important findings:

- false negatives tend to be later H8 breaks
- wrong direction errors are small
- some false positives become H16 breaks, so they may represent early pressure rather than pure noise

## Confidence Finding

Break confidence is meaningful:

| Break confidence | Test accuracy |
|---|---:|
| 0.50-0.60 | 0.4893 |
| 0.60-0.70 | 0.5714 |
| 0.70-0.80 | 0.7347 |
| 0.80-0.90 | 0.8437 |
| 0.90-1.00 | 0.9532 |

This is the strongest Stage 1B bridge finding:

> Stage 1B confidence can be used by Stage 2 as an actionability / filtering feature.

## Proximity Diagnostic Findings

H8 high-confidence predictions are strongly related to structural level proximity.

Median nearest distance:

| Segment | Median nearest distance |
|---|---:|
| all | 0.0018 |
| correct | 0.0010 |
| high confidence | -0.0116 |
| false positive break | 0.0027 |
| false negative break | 0.0026 |

Negative distance means price is already beyond the relevant break level.

Interpretation:

- high-confidence predictions are not abstract-only signals
- they are often at or beyond structural break levels
- false positives are usually close to the predicted break level but do not complete H8 confirmation
- false negatives are often close but break later inside H8

This supports passing price-level context to Stage 2.

## Stage 1B Output

The selected Stage 1B output is defined in:

```text
docs/foundations/stage1b-output-contract.md
```

Minimal signal:

```text
Stage1BSignal:
  horizon: H8
  side: long | short | none
  p_break
  p_direction
  break_confidence
  direction_confidence
```

Required context:

```text
Stage1BContext:
  current_price
  bull_level
  bear_level
  effective_break_pct
  distance_to_bull_level
  distance_to_bear_level
  predicted_side_distance_to_level
  symbol
  timestamp
```

Stage 1B should not be consumed without the price/level context.

## What Stage 1B Does Not Solve

Stage 1B does not decide:

- whether to enter now
- entry type
- exact entry level
- stop-loss
- target
- expected risk/reward
- position sizing
- trade quality
- no_trade vs actionable

These remain Stage 2 responsibilities.

## Decision

Stage 1B is complete enough to move to Stage 2 design.

Selected baseline:

```text
stage1b_h8_predictor_v1a_z_fused
```

Rejected for primary use:

```text
stage1b_h8_h16_multihorizon_v1b_z_fused
```

Diagnostic-only:

```text
H16 + H32 delayed setup analysis
H8 + H16 equal-weight multi-horizon
H4 as possible future trigger auxiliary
```

## Stage 2 Handoff

Stage 2 should start with the following question:

> Given an H8 Stage 1B signal and structural price context, is this an actionable trade candidate, a wait condition, or no trade?

Initial Stage 2 actionability classes:

```text
no_trade
wait_for_break
wait_for_retest
actionable_break_candidate
```

Recommended initial Stage 2 scope:

1. Use only H8 Stage 1B signals.
2. Include `break_confidence` bucket.
3. Include distance-to-level features.
4. Treat high-confidence candidates separately from low-confidence candidates.
5. Do not jump directly to PnL optimization before actionability labels are defined.

## Primary Artifacts

Foundation:

```text
docs/foundations/stage1b-forward-labels.md
docs/foundations/stage1b-output-contract.md
```

Selected checkpoint:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/best.pt
```

Key readouts:

```text
docs/experiments/stage1b/2026-05-18-stage1b-short-horizon-label-probe-v1-readout.md
docs/experiments/stage1b/2026-05-18-stage1b-h8-predictor-v1a-readout.md
docs/experiments/stage1b/2026-05-18-stage1b-h8-predictor-error-audit-v1-readout.md
docs/experiments/stage1b/2026-05-19-stage1b-h8-visual-proximity-diagnostic-v1-readout.md
```

Supporting scripts:

```text
scripts/generate_stage1b_forward_labels.py
scripts/probe_stage1b_forward_labels.py
scripts/train_stage1b_predictor.py
scripts/evaluate_stage1b_thresholds.py
scripts/audit_stage1b_predictor_errors.py
scripts/audit_stage1b_proximity.py
```
