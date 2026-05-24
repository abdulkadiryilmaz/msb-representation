# Stage 2 Actionability Threshold Sweep v1 Readout

**Date**: 2026-05-22
**Status**: completed
**Source readout**: `2026-05-22-stage2-actionability-label-v1-readout.md`
**Split**: test

## Amaç

Stage 2 actionability label v1 kurallarının threshold hassasiyetini ölçmek.

Sweep edilen alanlar:

```text
min_confidence: 0.60, 0.70, 0.80, 0.90
near_threshold: 0.25%, 0.50%, 1.00%
watch_threshold: 1.00%, 1.50%, 2.00%
min_mfe_r: 1.0R, 1.5R, 2.0R
```

Toplam kombinasyon: `108`

## Command

```bash
PYTHONPATH=src python scripts/sweep_stage2_actionability_thresholds.py \
  --prediction-csv data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/proximity_audit_test/predictions_with_proximity.csv \
  --raw-data-root data/raw/binance \
  --output-dir data/stage2/binance/15m/stage2_actionability_v1_test/threshold_sweep_v1
```

## Default-like Baseline

Default v1:

```text
min_confidence = 0.70
near_threshold = 0.50%
watch_threshold = 1.50%
min_mfe_r = 1.0R
```

Result:

| Metric | Value |
|---|---:|
| `no_trade` | 7140 |
| `wait_for_break` | 155 |
| `wait_for_retest` | 948 |
| `actionable_break_candidate` | 1452 |
| actionable rate | 0.1498 |
| Stage 1B false positives total | 908 |
| false positives to `no_trade` | 866 |
| false positives to `actionable` | 2 |
| actionable correct rate | 0.9979 |

Default v1 is usable, but leaves two Stage 1B false-positive break predictions as actionable.

## Stronger Candidate

Best practical candidate from the sweep:

```text
min_confidence = 0.80
near_threshold = 0.50% or 1.00%
watch_threshold = 1.00% to 2.00%
min_mfe_r = 1.0R
```

Representative result:

```text
min_confidence = 0.80
near_threshold = 0.50%
watch_threshold = 1.50%
min_mfe_r = 1.0R
```

| Metric | Value |
|---|---:|
| `no_trade` | 7593 |
| `wait_for_break` | 71 |
| `wait_for_retest` | 753 |
| `actionable_break_candidate` | 1278 |
| actionable rate | 0.1318 |
| Stage 1B false positives total | 908 |
| false positives to `no_trade` | 891 |
| false positives to `actionable` | 0 |
| actionable correct rate | 1.0000 |

This candidate removes all Stage 1B false-positive actionable examples while preserving a meaningful actionable subset.

## High-confidence Behavior

For `min_confidence=0.80`, the high-confidence Stage 1B segment remains stable:

| Label | Support |
|---|---:|
| `actionable_break_candidate` | ~1013 |
| `wait_for_retest` | 419 |
| `no_trade` | 407-408 |

This suggests the high-confidence segment is not overly sensitive to the distance thresholds.

## Effect of `min_mfe_r`

Raising `min_mfe_r` from `1.0R` to `1.5R` sharply reduces actionable support but also removes false-positive actionable examples.

Example at `min_confidence=0.70`:

| min_mfe_r | actionable | false-positive actionable | actionable correct rate |
|---:|---:|---:|---:|
| 1.0R | 1452 | 2 | 0.9979 |
| 1.5R | 909 | 0 | 1.0000 |

Interpretation:

- `1.0R` is broader and useful for first predictor training.
- `1.5R` is cleaner but may be too selective for early Stage 2 modeling.

## Effect of Distance Thresholds

Distance thresholds have smaller impact than confidence and MFE thresholds.

Across reasonable values:

- actionable support changes moderately
- false-positive actionable count is mostly controlled by `min_confidence` and `min_mfe_r`
- `watch_threshold` mainly affects `wait_for_break`

## Decision

Use the following candidate for the next Stage 2 label generation unless visual sanity checks contradict it:

```text
min_confidence = 0.80
near_threshold = 0.005
watch_threshold = 0.015
min_mfe_r = 1.0
```

Reason:

- keeps actionable support above `1200`
- filters Stage 1B false-positive actionable examples to zero
- keeps high-confidence actionable/retest structure stable
- avoids making the label too sparse before predictor training

## Next Step

Before training a Stage 2 predictor:

1. regenerate labels with the selected thresholds
2. visually inspect actionable and wait-for-retest samples
3. document whether the path semantics match the label names
4. then train a baseline Stage 2 classifier

Selected-threshold labels were generated at:

```text
data/stage2/binance/15m/stage2_actionability_v1_selected_test/stage2_actionability_labels.csv
```

Visual inspection tool:

```bash
streamlit run scripts/inspect_stage2_actionability.py -- \
  --labels data/stage2/binance/15m/stage2_actionability_v1_selected_test/stage2_actionability_labels.csv
```

Inspection focus:

- `actionable_break_candidate`: trigger, target, invalidation ordering should visually make sense
- `wait_for_retest`: break should exist, but immediate chase should look less attractive than retest/hold
- `wait_for_break`: price should be near/watchable and not yet clearly actionable at anchor
- `no_trade`: especially high-confidence no-trade cases should have a clear blocking reason

## Verification

```text
ruff check scripts/generate_stage2_actionability_labels.py scripts/audit_stage2_actionability_labels.py scripts/sweep_stage2_actionability_thresholds.py
python -m py_compile scripts/generate_stage2_actionability_labels.py scripts/audit_stage2_actionability_labels.py scripts/sweep_stage2_actionability_thresholds.py
```

Both passed.
