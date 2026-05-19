# Stage 1B H8 Predictor Error Audit v1 Readout

**Date**: 2026-05-18
**Status**: completed
**Source checkpoint**: `stage1b_h8_predictor_v1a_z_fused/best.pt`
**Split**: test
**Artifact**: `data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/error_audit_test/summary.json`

## Goal

H8 predictor v1a'nın hata yapısını anlamak:

```text
true none, predicted break
true break, predicted none
wrong direction
high-confidence correct / wrong
H8 none, H16 break delayed segment
```

Bu çalışma Stage 2'ye geçmeden önce Stage 1B çıktısının hangi durumlarda güvenilir olduğunu anlamak için yapıldı.

## Overall

H8 predictor v1a test split:

| Segment | Support |
|---|---:|
| all | 9,695 |
| correct | 7,104 |
| false positive break | 908 |
| false negative break | 1,514 |
| wrong direction | 169 |

Error profile:

- Ana hata `break -> none` false negative.
- Wrong direction az.
- False positive break örneklerinin bir kısmı H16'da gerçekten break'e dönüşüyor.

## Segment Profiles

### False Positive Break

Definition:

```text
true H8 = none
pred H8 = bullish/bearish
```

| Metric | Value |
|---|---:|
| support | 908 |
| mean p_break | 0.6750 |
| median p_break | 0.6552 |
| mean direction confidence | 0.9115 |

Predicted direction counts:

```text
bullish=458
bearish=450
```

H16 label distribution:

```text
H16 none=557
H16 bullish=162
H16 bearish=188
H16 insufficient_future=1
```

Interpretation:

False positives are not all pure noise. `350 / 908 = 38.5%` become bullish/bearish by H16. The model may be detecting some early pressure, but H8 label marks them as no-break.

### False Negative Break

Definition:

```text
true H8 = bullish/bearish
pred H8 = none
```

| Metric | Value |
|---|---:|
| support | 1,514 |
| mean p_break | 0.3206 |
| median p_break | 0.3250 |
| mean direction confidence | 0.7655 |

True direction counts:

```text
bullish=737
bearish=777
```

Time-to-break:

| min | q25 | median | q75 | max |
|---:|---:|---:|---:|---:|
| 2 | 2 | 4 | 6 | 8 |

Interpretation:

False negatives are later H8 breaks on average. Correct predictions have median time-to-break `2`, while false negatives have median `4`. The model is better at near/immediate breaks and weaker when break occurs later in the H8 horizon.

### Wrong Direction

Definition:

```text
true break
pred break
pred direction != true direction
```

| Metric | Value |
|---|---:|
| support | 169 |
| mean p_break | 0.6740 |
| median p_break | 0.6465 |
| mean direction confidence | 0.8226 |

Direction errors are small relative to break/no-break errors. This confirms the previous readout: direction-given-break is not the main bottleneck.

## H8 None / H16 Break Segment

Definition:

```text
true H8 = none
true H16 = bullish/bearish
```

| Metric | Value |
|---|---:|
| support | 1,307 |
| mean p_break_8 | 0.3837 |
| median p_break_8 | 0.3247 |

Prediction distribution:

```text
pred none=957
pred bullish=188
pred bearish=162
```

H16 direction counts:

```text
H16 bullish=587
H16 bearish=720
```

Interpretation:

The H8 predictor mostly does not fire on delayed H16 breaks. This is good if H8 is meant to be a near-term trigger. It also means delayed setup detection should not be expected from H8 alone.

## Confidence Buckets

Break confidence is defined as:

```text
if pred break: p_break
if pred none : 1 - p_break
```

| Break confidence | Support | Accuracy | Main errors |
|---|---:|---:|---|
| 0.50-0.60 | 1,588 | 0.4893 | mixed FP/FN |
| 0.60-0.70 | 1,729 | 0.5714 | mixed FP/FN |
| 0.70-0.80 | 2,254 | 0.7347 | mostly FN |
| 0.80-0.90 | 2,265 | 0.8437 | mostly FN |
| 0.90-1.00 | 1,859 | 0.9532 | few errors |

This is the strongest practical finding:

> H8 predictor confidence is meaningful. High-confidence predictions are much more reliable.

## Interpretation

H8 predictor v1a is not only producing class labels; its confidence carries useful quality information.

Main conclusions:

- `p_break_8` is a usable near-term structure signal.
- Direction-given-break is strong; direction error is not the primary bottleneck.
- Many false negatives are later H8 breaks, suggesting `time_to_break` or urgency could be useful.
- Some false positives become H16 breaks, suggesting they may represent early pressure rather than pure noise.
- Confidence buckets are likely useful for Stage 2 quality filtering.

## Decision

Keep H8 predictor v1a as the primary Stage 1B baseline.

Do not add H16 equal-weight auxiliary as default.

Next Stage 1B/Stage 2 bridge candidates:

1. Add `time_to_break` bucket or urgency target:

```text
early_break: time_to_break <= 2
late_break:  time_to_break > 2
```

2. Use `p_break_8` confidence as an input to Stage 2 quality evaluation.
3. Run visual/manual audit on high-confidence false positives and false negatives before changing labels.

## Artifacts

```text
scripts/audit_stage1b_predictor_errors.py
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/error_audit_test/summary.json
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/error_audit_test/all_predictions.csv
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/error_audit_test/top_false_positive_break.csv
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/error_audit_test/top_false_negative_break.csv
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/error_audit_test/top_wrong_direction.csv
```
