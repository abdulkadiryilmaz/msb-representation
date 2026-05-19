# Stage 1B Short-Horizon Label + Probe v1 Readout

**Date**: 2026-05-18
**Status**: completed
**Scope**: H4 / H8 / H16 forward labels and z_fused probes
**Stage 1A source**: `ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`

## Goal

H16'nın bile break-heavy görünmesi sonrası daha kısa horizon'ları test etmek:

```text
H4  = 1 saat
H8  = 2 saat
H16 = 4 saat
```

Ana soru:

> H4 / H8 horizonları, H16'ya göre daha dengeli ve daha okunabilir forward-structure target'ları üretiyor mu?

## Label Artifacts

```text
stage1b_forward_labels_train_h4_h8_h16.parquet
stage1b_forward_labels_val_h4_h8_h16.parquet
stage1b_forward_labels_test_h4_h8_h16.parquet
stage1b_forward_label_audit_h4_h8_h16.json
```

## Label Distribution

Test split:

| Horizon | none | bullish | bearish | ambiguous | insufficient_future | break rate |
|---|---:|---:|---:|---:|---:|---:|
| H4 | 5,368 | 2,052 | 2,279 | 1 | 4 | 0.4464 |
| H8 | 4,334 | 2,534 | 2,827 | 1 | 8 | 0.5526 |
| H16 | 3,025 | 3,115 | 3,547 | 1 | 16 | 0.6871 |

Compared with H16, shorter horizons are much less break-heavy:

- H4 is none-majority but still has substantial directional break support.
- H8 is close to balanced between no-break and break.
- H16 is already break-heavy.

## Consistency

Suspicious transitions:

| Transition | Split | short break -> long none | direction flip |
|---|---|---:|---:|
| H4 -> H8 | train | 0 | 0 |
| H4 -> H8 | val | 0 | 0 |
| H4 -> H8 | test | 0 | 0 |
| H8 -> H16 | train | 0 | 0 |
| H8 -> H16 | val | 0 | 0 |
| H8 -> H16 | test | 0 | 0 |

The short-horizon label set is internally consistent.

## Delayed Segments

Test split:

| Segment | Support | Rate | Bullish | Bearish |
|---|---:|---:|---:|---:|
| H4 none / H8 break | 1,032 | 0.1063 | 484 | 548 |
| H8 none / H16 break | 1,307 | 0.1347 | 587 | 720 |

Both delayed segments have enough support for targeted analysis or auxiliary labels.

## Probe Results

Input:

```text
z_fused
```

Training:

```text
train split -> probe fit
test split  -> evaluation
drop: ambiguous, insufficient_future
```

| Target | Majority macro F1 | Linear macro F1 | MLP macro F1 |
|---|---:|---:|---:|
| H4 | 0.2375 | 0.7494 | 0.7752 |
| H8 | 0.2060 | 0.7233 | 0.7284 |
| H16 | 0.1622 | 0.6768 | 0.6712 |

Shorter horizons are substantially more readable from the selected Stage 1A representation.

## Confusion Matrices

H4 linear probe:

Labels: `none / bullish / bearish`

```text
none    -> [3721,  791,  856]
bullish -> [ 300, 1698,   54]
bearish -> [ 344,   48, 1887]
```

H8 linear probe:

```text
none    -> [2915,  691,  728]
bullish -> [ 503, 1907,  124]
bearish -> [ 538,  124, 2165]
```

H16 linear probe:

```text
none    -> [1975,  530,  520]
bullish -> [ 722, 2143,  250]
bearish -> [ 819,  292, 2436]
```

## Interpretation

H4/H8 labels look more suitable than H16 as near-term Stage 1B targets.

Key points:

- H4 has the highest probe F1, but it may be closer to immediate level proximity / near-touch behavior.
- H8 is likely the best next predictor target: more balanced than H16, less ultra-short than H4, and strongly readable.
- H16 remains useful as a wider follow-through / delayed horizon, but should not be the only primary target.

This changes the Stage 1B direction:

```text
old primary: H16
new candidate primary: H8
candidate multi-horizon: H8 + H16
diagnostic near-term: H4
```

## Decision

Proceed with H8 predictor v1.

Do not train another H16/H32 predictor before testing H8.

Recommended next run:

```text
Stage 1B H8 predictor v1a
input: z_fused
target: h8_future_break_direction
objective: break occurrence + direction-given-break
selection: val reconstructed 3-class macro F1
```

After H8 predictor:

1. Compare H8 predictor to H8 probe.
2. Run threshold sweep.
3. If H8 is stable, test H8 + H16 multi-horizon.

## Artifacts

```text
scripts/generate_stage1b_forward_labels.py
scripts/audit_stage1b_forward_labels.py
scripts/probe_stage1b_forward_labels.py

data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_train_h4_h8_h16.parquet
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_val_h4_h8_h16.parquet
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_test_h4_h8_h16.parquet
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_label_audit_h4_h8_h16.json

data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/stage1b_h4_direction_3class_test_z_fused/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/stage1b_h8_direction_3class_test_z_fused/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/stage1b_h16_direction_3class_test_z_fused_h4set/
```
