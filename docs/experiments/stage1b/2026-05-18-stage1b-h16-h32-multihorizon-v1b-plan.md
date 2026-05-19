# Stage 1B H16 + H32 Multi-Horizon Predictor v1b Plan

**Date**: 2026-05-18
**Status**: completed
**Baseline**: `stage1b_h16_predictor_v1a_z_fused`
**Label contract**: `docs/foundations/stage1b-forward-labels.md`

## Goal

Test whether adding H32 supervision improves Stage 1B forecasting without degrading the H16 baseline.

V1b question:

> Can the predictor learn near break behavior at H16 and delayed / wider intraday structure bias at H32 together?

## Scope

V1b remains Stage 1B. It does not predict entry, stop, target, PnL, or tradability.

Outputs:

```text
H16:
  p_break_16
  p_direction_16_given_break

H32:
  p_break_32
  p_direction_32_given_break
```

## Objective

```text
loss =
  CE(break_occurs_16)
  + CE(direction_16 | break_16)
  + CE(break_occurs_32)
  + CE(direction_32 | break_32)
```

Initial equal weights:

```text
h16_break_weight = 1.0
h16_direction_weight = 1.0
h32_break_weight = 1.0
h32_direction_weight = 1.0
```

Direction losses are masked for `none` samples.

## Input

Initial input remains:

```text
z_fused
```

This keeps the comparison with v1a clean.

## Evaluation

Primary guardrail:

```text
H16 reconstructed 3-class macro F1 must not degrade materially vs v1a.
```

Reference:

```text
v1a H16 test reconstructed macro F1 = 0.6731
```

Secondary metrics:

- H32 reconstructed macro F1
- H16 break occurrence macro F1
- H32 break occurrence macro F1
- H16 direction-given-break macro F1
- H32 direction-given-break macro F1

Key segment:

```text
H16 = none
H32 = bullish/bearish
```

This segment represents structures that do not break immediately but may mature within the wider intraday horizon.

## Decision Rule

Accept v1b if:

- H16 reconstructed macro F1 stays close to v1a
- H32 produces useful forecast signal
- H16 none / H32 directional segment becomes inspectable via outputs

Reject or revise if:

- H32 supervision degrades H16 materially
- both horizons collapse toward break-heavy predictions
- direction heads collapse to one side
