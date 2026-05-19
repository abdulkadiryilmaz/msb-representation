# Stage 1B H8 Predictor v1a Plan

**Date**: 2026-05-18
**Status**: completed
**Baseline context**: `2026-05-18-stage1b-short-horizon-label-probe-v1-readout.md`
**Stage 1A source**: `ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`

## Goal

Train the first H8-focused Stage 1B predictor.

V1a question:

> Can the selected Stage 1A representation support a practical two-head predictor for the next 8 bars?

H8 corresponds to roughly 2 hours on 15m candles.

## Motivation

Short-horizon probe v1 showed:

| Target | Test break rate | Linear macro F1 | MLP macro F1 |
|---|---:|---:|---:|
| H4 | 0.4464 | 0.7494 | 0.7752 |
| H8 | 0.5526 | 0.7233 | 0.7284 |
| H16 | 0.6871 | 0.6768 | 0.6712 |

H8 is selected as the next primary target because it is less break-heavy than H16 and less ultra-short than H4.

## Scope

This is still Stage 1B. It does not produce entry, stop, target, sizing, leverage, or TradePlan quality.

Outputs:

```text
p_break_8
p_direction_8_given_break
```

## Objective

```text
loss =
  CE(break_occurs_8)
  + CE(direction_8 | break_8)
```

Direction loss is masked for `none` samples.

## Input

```text
z_fused
```

## Dataset

Forward labels:

```text
stage1b_forward_labels_train_h4_h8_h16.parquet
stage1b_forward_labels_val_h4_h8_h16.parquet
stage1b_forward_labels_test_h4_h8_h16.parquet
```

Target:

```text
h8_future_break_direction
```

Dropped labels:

```text
ambiguous
insufficient_future
```

## Evaluation

Primary metric:

```text
reconstructed 3-class macro F1
```

Secondary metrics:

- break occurrence macro F1
- direction-given-break macro F1
- threshold sweep on `p_break_8`
- confusion matrices

Reference to beat:

```text
H8 z_fused linear probe macro F1 = 0.7233
H8 z_fused MLP probe macro F1    = 0.7284
```

## Decision Rule

Accept H8 predictor v1a if:

- reconstructed macro F1 is near the H8 probe range
- break occurrence is materially cleaner than H16 predictor
- direction head does not collapse to one side

If accepted, next step is H8 + H16 multi-horizon.
