# Stage 2 Actionability Label v1 Readout

**Date**: 2026-05-22
**Status**: completed
**Foundation**: `docs/foundations/stage2-actionability-labels.md`
**Stage 1B source**: `stage1b_h8_predictor_v1a_z_fused`
**Split**: test

## Amaç

Bu çalışma Stage 2 için ilk rule-based actionability label setini üretir ve audit eder.

Ana soru:

> Stage 1B H8 predictor çıktısı, fiyat yolu ve seviye bağlamıyla `no_trade / wait / actionable` ayrımına dönüştürülebiliyor mu?

Bu çalışma PnL optimizasyonu değildir. Label, path-quality / actionability proxy olarak okunmalıdır.

## Inputs

Stage 1B proximity audit çıktısı:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/proximity_audit_test/predictions_with_proximity.csv
```

Raw OHLCV:

```text
data/raw/binance/*.parquet
```

Output:

```text
data/stage2/binance/15m/stage2_actionability_v1_test/
```

## Commands

```bash
PYTHONPATH=src python scripts/generate_stage2_actionability_labels.py \
  --prediction-csv data/stage1b/binance/15m/checkpoints/stage1b_h8_predictor_v1a_z_fused/proximity_audit_test/predictions_with_proximity.csv \
  --raw-data-root data/raw/binance \
  --output-dir data/stage2/binance/15m/stage2_actionability_v1_test
```

```bash
PYTHONPATH=src python scripts/audit_stage2_actionability_labels.py \
  --label-csv data/stage2/binance/15m/stage2_actionability_v1_test/stage2_actionability_labels.csv \
  --output-dir data/stage2/binance/15m/stage2_actionability_v1_test/audit
```

## Label Distribution

Total support: `9695`

| Label | Support |
|---|---:|
| `no_trade` | 7140 |
| `wait_for_break` | 155 |
| `wait_for_retest` | 948 |
| `actionable_break_candidate` | 1452 |

This distribution does not collapse to a single class. The first v1 rules produce a meaningful actionable subset.

## Reason Codes

| Reason | Support |
|---|---:|
| `stage1b_no_break` | 4940 |
| `target_before_invalidation` | 1452 |
| `low_confidence` | 1411 |
| `retest_after_break` | 948 |
| `fallback_no_actionability` | 370 |
| `no_break_in_observation_horizon` | 348 |
| `break_expected_within_observation` | 155 |
| `missing_trigger_level` | 28 |
| `too_far_from_trigger` | 26 |
| `invalidated_before_break` | 11 |
| `insufficient_future` | 6 |

The largest `no_trade` group is Stage 1B `pred_label=none`, as expected. The second largest explicit filter is low confidence.

## Distance Buckets

| Distance bucket | Support |
|---|---:|
| `not_applicable` | 4974 |
| `at_or_beyond_trigger` | 3387 |
| `near_trigger` | 884 |
| `watchable` | 383 |
| `far` | 67 |

`not_applicable` corresponds mostly to Stage 1B no-break predictions.

## Confidence Behavior

High-confidence Stage 1B segment (`break_confidence >= 0.90`):

| Label | Support |
|---|---:|
| `actionable_break_candidate` | 1013 |
| `wait_for_retest` | 419 |
| `no_trade` | 407 |
| `wait_for_break` | 20 |

This is a good sign: high-confidence Stage 1B signals are not all actionable, but they concentrate strongly in actionable or retest-wait classes.

## Stage 1B False Positive Filtering

Stage 1B false-positive break segment label counts:

| Label | Support |
|---|---:|
| `no_trade` | 866 |
| `wait_for_break` | 36 |
| `wait_for_retest` | 4 |
| `actionable_break_candidate` | 2 |

This is the strongest positive result of the label v1 audit.

Interpretation:

> The Stage 2 rule scaffold filters most Stage 1B false-positive break predictions into `no_trade`.

This directly supports the Stage 2 role: not every Stage 1B break candidate should become a TradePlan candidate.

## Actionable Segment Quality

`actionable_break_candidate` support: `1452`

Stage 1B error type counts inside actionable:

| Stage 1B error type | Support |
|---|---:|
| `correct` | 1449 |
| `false_positive_break` | 2 |
| `wrong_direction` | 1 |

MFE/MAE profile:

| Metric | q25 | median | q75 |
|---|---:|---:|---:|
| `mfe_r` | 1.3005 | 1.7277 | 2.4984 |
| `mae_r` | 0.0000 | 0.0000 | 0.0000 |

The actionable subset has strong favorable excursion by construction, and most examples do not move adversely below the trigger before reaching the v1 target proxy.

## Wait Classes

`wait_for_retest`:

| Metric | q25 | median | q75 |
|---|---:|---:|---:|
| `mfe_r` | 0.3832 | 0.5571 | 0.7367 |
| `mae_r` | 0.0326 | 0.1790 | 0.4349 |

`wait_for_break`:

| Metric | q25 | median | q75 |
|---|---:|---:|---:|
| `mfe_r` | 0.2689 | 0.3816 | 0.6414 |
| `mae_r` | 0.1810 | 0.2935 | 0.5233 |

The wait classes are materially weaker than actionable candidates on `mfe_r`, which is expected.

## Interpretation

Label v1 is a usable first scaffold.

Positive signs:

- label distribution does not collapse
- high-confidence Stage 1B segment concentrates in actionable/retest classes
- Stage 1B false positives are mostly filtered to `no_trade`
- actionable candidates are almost entirely Stage 1B-correct examples
- actionable MFE profile is clearly stronger than wait/no-trade classes

Risks / caveats:

- actionable label is partly constructed from future path, so it is not a predictor result yet
- invalidation currently uses short-window extrema with opposite-level fallback
- thresholds are hand-picked and need sweep
- `wait_for_break` support is small (`155`)
- `wait_for_retest` semantics need visual inspection before model training

## Decision

Proceed to threshold and path sanity audit before training a Stage 2 predictor.

Recommended next steps:

1. inspect actionable and retest samples visually
2. sweep `min_confidence`, `near_threshold`, `watch_threshold`, and `min_mfe_r`
3. compare label distribution and false-positive filtering across sweeps
4. only then train a baseline Stage 2 classifier

## Verification

```text
ruff check scripts/generate_stage2_actionability_labels.py scripts/audit_stage2_actionability_labels.py
python -m py_compile scripts/generate_stage2_actionability_labels.py scripts/audit_stage2_actionability_labels.py
```

Both passed.
