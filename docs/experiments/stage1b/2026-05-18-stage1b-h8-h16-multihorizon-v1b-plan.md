# Stage 1B H8 + H16 Multi-Horizon Predictor v1b Plan

**Date**: 2026-05-18
**Status**: completed
**Baseline**: `stage1b_h8_predictor_v1a_z_fused`
**Stage 1A source**: `ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`

## Goal

Test whether H16 follow-through supervision improves or stabilizes the H8 primary predictor.

V1b question:

> Can the model learn H8 near-term break prediction and H16 wider follow-through together without degrading H8?

## Scope

This remains Stage 1B. It does not generate TradePlan, entry, stop, target, sizing, or quality.

Outputs:

```text
H8:
  p_break_8
  p_direction_8_given_break

H16:
  p_break_16
  p_direction_16_given_break
```

## Objective

```text
loss =
  CE(break_occurs_8)
  + CE(direction_8 | break_8)
  + CE(break_occurs_16)
  + CE(direction_16 | break_16)
```

Initial weights are equal.

## Input

```text
z_fused
```

## Evaluation

Primary guardrail:

```text
H8 reconstructed 3-class macro F1 must not degrade materially vs H8 v1a.
```

Reference:

```text
H8 v1a test reconstructed macro F1 = 0.7307
H8 v1a test break macro F1 = 0.7497
H8 v1a test direction macro F1 = 0.8955
```

Secondary metrics:

- H16 reconstructed macro F1
- H8 break occurrence macro F1
- H16 break occurrence macro F1
- H8 direction-given-break macro F1
- H16 direction-given-break macro F1

Key segment:

```text
H8 = none
H16 = bullish / bearish
```

This segment represents structures that do not break within 2 hours but may break within 4 hours.

## Decision Rule

Accept v1b if:

- H8 reconstructed macro F1 stays close to or above v1a
- H16 produces useful follow-through signal
- H8 direction head does not degrade

Reject or revise if:

- H16 supervision degrades H8 materially
- model becomes break-heavy again
- H16 does not add useful secondary signal
