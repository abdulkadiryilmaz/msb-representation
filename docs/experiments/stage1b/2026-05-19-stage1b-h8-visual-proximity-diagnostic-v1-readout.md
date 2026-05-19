# Stage 1B H8 Visual / Proximity Diagnostic v1 Readout

**Date**: 2026-05-19
**Status**: completed
**Source checkpoint**: `stage1b_h8_predictor_v1a_z_fused/best.pt`
**Split**: test
**Related contract**: `docs/foundations/stage1b-output-contract.md`

## Goal

Stage 1B H8 predictor çıktısının fiyat seviyeleri açısından ne ifade ettiğini incelemek.

Soru:

> H8 predictor gerçekten break seviyesine yakın / actionable adayları mı işaretliyor, yoksa fiyat bağlamından kopuk istatistiksel bir sinyal mi üretiyor?

Bu diagnostic görsel plot üretmez; visual/manual audit öncesi proximity tablosu üretir.

## Method

H8 predictor test tahminleri label artifact'ındaki structure seviyeleriyle birleştirildi:

```text
current_close
bull_level
bear_level
effective_break_pct
distance_to_bull_level
distance_to_bear_level
predicted_side_distance_to_level
true_side_distance_to_level
```

Distance convention:

```text
bull distance = (bull_level - current_close) / current_close
bear distance = (current_close - bear_level) / current_close
```

Negative distance means price is already beyond the relevant break level.

## Key Results

### Segment Distance Summary

Median values:

| Segment | Support | nearest distance pct | predicted-side distance pct | true-side distance pct |
|---|---:|---:|---:|---:|
| all | 9,695 | 0.0018 | -0.0039 | -0.0029 |
| correct | 7,104 | 0.0010 | -0.0064 | -0.0064 |
| false positive break | 908 | 0.0027 | 0.0029 | 0.0000 |
| false negative break | 1,514 | 0.0026 | 0.0000 | 0.0029 |
| wrong direction | 169 | 0.0016 | 0.0048 | 0.0077 |
| high confidence | 1,859 | -0.0116 | -0.0126 | -0.0130 |
| high confidence wrong | 87 | 0.0022 | 0.0003 | 0.0124 |

Interpretation:

- Correct break predictions are often already beyond the predicted break level.
- False positives are usually close to the predicted break level but not confirmed inside H8.
- False negatives are also close, but the true-side distance is still positive at anchor and breaks later.
- High-confidence predictions are strongly proximity-driven: median nearest distance is negative.

### Normalized Distance

Distance divided by `effective_break_pct`:

| Segment | nearest distance / break pct median | predicted-side distance / break pct median | true-side distance / break pct median |
|---|---:|---:|---:|
| correct | 0.4568 | -2.6106 | -2.6106 |
| false positive break | 1.1139 | 1.1583 | 0.0000 |
| false negative break | 1.1328 | 0.0000 | 1.2667 |
| high confidence | -4.8345 | -5.1317 | -5.2904 |
| high confidence wrong | 0.8802 | 0.1620 | 4.4093 |

High-confidence correct examples are well past the break threshold. High-confidence wrong examples are much closer/borderline and are mostly false positives.

## Practical Interpretation

The H8 predictor is not only using abstract latent structure; its output is strongly tied to level proximity.

This is good for TradePlan bridging:

```text
high p_break_8 + high confidence
```

often means price is already close to or beyond a structural break level.

But it also clarifies the limitation:

```text
Stage 1B signal != entry decision
```

For false positives, the model may be detecting pressure near the break level, but H8 confirmation does not complete. This can become:

```text
wait_for_break
```

rather than immediate trade.

For false negatives, the true break often occurs later in H8. This suggests:

```text
time_to_break / urgency
```

would be useful if we want to distinguish immediate trigger from later maturation.

## Stage 2 Implication

Stage 2 should receive both:

```text
Stage1BSignal:
  p_break_8
  direction
  break_confidence
  direction_confidence
```

and:

```text
Stage1BContext:
  current_price
  break_level
  distance_to_level
  effective_break_pct
```

Without price-level context, Stage 2 cannot decide:

- enter now
- wait for break
- wait for retest
- no trade

## Decision

Stage 1B H8 predictor is suitable as a Stage 2 input signal, but only with structural price context.

Keep:

```text
stage1b_h8_predictor_v1a_z_fused
```

as the Stage 1B baseline.

Before Stage 2 training, define the Stage 2 actionability label / rule scaffold around:

```text
no_trade
wait_for_break
wait_for_retest
actionable_break_candidate
```

## Artifacts

```text
scripts/audit_stage1b_proximity.py
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/proximity_audit_test/summary.json
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/proximity_audit_test/predictions_with_proximity.csv
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/proximity_audit_test/high_confidence_wrong.csv
```
