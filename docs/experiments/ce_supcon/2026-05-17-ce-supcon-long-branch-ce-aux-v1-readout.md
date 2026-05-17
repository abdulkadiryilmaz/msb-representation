# CE + SupCon Long Branch CE Aux v1 Readout

**Date**: 2026-05-17  
**Status**: Readout  
**Runs**: `seed_41`, `seed_42`  
**Checkpoint dir**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50`

## Setup

Objective:

```text
fused CE
+ 0.05 * SupCon(z_long_proj)
+ 0.10 * CE(long_aux_head(z_long), label)
```

No CE warmup or SupCon frontload was used.

```text
epochs = 50
patience = 100
epoch_checkpoints = 5, 10, 20, 30, 40, 50
seeds = 41, 42
```

## Training Dynamics

| epoch | train_ce | train_supcon | train_long_aux | val_ce | val_supcon | val_long_aux | val_macro_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4424 | 3.1216 | 0.5582 | 0.3632 | 3.0307 | 0.4164 | 0.8463 |
| 5 | 0.3312 | 2.9726 | 0.3681 | 0.2929 | 2.9580 | 0.3309 | 0.8764 |
| 10 | 0.2908 | 2.9393 | 0.3360 | 0.2786 | 2.9382 | 0.3164 | 0.8834 |
| 20 | 0.2496 | 2.9029 | 0.3070 | 0.2830 | 2.9297 | 0.3086 | 0.8849 |
| 30 | 0.2240 | 2.8809 | 0.2904 | 0.3023 | 2.9230 | 0.3106 | 0.8844 |
| 40 | 0.2075 | 2.8609 | 0.2760 | 0.3059 | 2.9272 | 0.3085 | 0.8874 |
| 44 | 0.2002 | 2.8553 | 0.2717 | 0.2929 | 2.9101 | 0.2948 | 0.8927 |
| 50 | 0.1933 | 2.8472 | 0.2667 | 0.3113 | 2.9200 | 0.3064 | 0.8832 |

Training confirms the intended mechanism:

- SupCon loss starts decreasing immediately; there is no long plateau.
- `long_aux_loss` also decreases steadily.
- Val macro F1 remains in the same broad range as prior variants, so geometry remains the decision axis.

## Val Geometry Sweep

| epoch | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.7921 | 0.3093 | 0.8065 | 0.3914 | 0.8043 | 0.2737 | 0.8286 | 0.3050 | 0.6297 |
| 10 | 0.7673 | 0.2925 | 0.8168 | 0.3002 | 0.8158 | 0.2680 | 0.8418 | 0.3056 | 0.8177 |
| 20 | 0.7615 | 0.2900 | 0.8227 | 0.2853 | 0.8214 | 0.2677 | 0.8429 | 0.2991 | 1.0069 |
| 30 | 0.7538 | 0.3090 | 0.8257 | 0.3019 | 0.8238 | 0.2729 | 0.8466 | 0.3161 | 1.1493 |
| 40 | 0.7251 | 0.3025 | 0.8262 | 0.2974 | 0.8246 | 0.2631 | 0.8440 | 0.3104 | 1.2964 |
| 50 | 0.7308 | 0.3405 | 0.8301 | 0.2920 | 0.8285 | 0.2749 | 0.8460 | 0.3371 | 1.3439 |

## Selected Checkpoint

Validation-first selection favors:

```text
epoch_020.pt
```

Reason:

- `z_long` is already stronger than dominant warmup seed 41.
- `z_fused` remains clean: label `0.8429`, symbol `0.2991`.
- effective long/short is close to balanced at `1.0069`.
- later epochs improve `z_long` slightly but start pushing effective long/short too high and degrade `z_short` / `z_fused` symbol behavior.

## Test Check

| checkpoint | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch_020.pt` | 0.7714 | 0.2848 | 0.8220 | 0.2745 | 0.8216 | 0.2586 | 0.8418 | 0.2913 | 1.0714 |
| `epoch_040.pt` | 0.7440 | 0.2975 | 0.8244 | 0.2789 | 0.8236 | 0.2628 | 0.8423 | 0.3011 | 1.3702 |

Test split supports the val-selected checkpoint:

- `epoch_020.pt` keeps `z_long` strong and `z_fused` cleaner.
- `epoch_040.pt` has slightly higher `z_long` label agreement, but branch balance is more long-heavy.

## Borderline Bucket Readout

This run started from the earlier `borderline_intact_break` problem, so `seed_41/epoch_020.pt` was also checked against the canonical `ce_supcon_long_v1` bucket diagnostics.

### Borderline Classifier / Neighbor Diagnosis

The headline result is positive:

| run | misclassified_rate | avg_nn_intact_rate | avg_nn_confirmed_rate | avg_nn_same_symbol_rate |
|---|---:|---:|---:|---:|
| `long_v1 / z_fused` | 0.2818 | 0.6847 | 0.3153 | 0.2814 |
| `long_v1 / z_long` | 0.2818 | 0.6602 | 0.3398 | 0.2900 |
| `branch_ce_aux_v1 e20 / z_fused` | 0.2134 | 0.6918 | 0.3082 | 0.2792 |
| `branch_ce_aux_v1 e20 / z_long` | 0.2134 | 0.6762 | 0.3238 | 0.2532 |

Interpretation:

- Borderline intact misclassification drops from `0.2818` to `0.2134`.
- `z_fused` neighbors become slightly more intact-family and slightly less confirmed-family.
- `z_long` same-symbol neighbor rate drops from `0.2900` to `0.2532`, so this improvement is not a symbol shortcut.

### Bucket Centroid Distances

Cosine distance is larger when bucket centroids are farther apart.

| embedding | pair | `long_v1` | `branch_ce_aux_v1 e20` | delta |
|---|---|---:|---:|---:|
| `z_long` | `borderline_down` vs `bearish_confirmed` | 0.2917 | 0.2974 | +0.0057 |
| `z_long` | `borderline_up` vs `bullish_confirmed` | 0.4199 | 0.3100 | -0.1099 |
| `z_long_proj` | `borderline_down` vs `bearish_confirmed` | 0.0632 | 0.0707 | +0.0075 |
| `z_long_proj` | `borderline_up` vs `bullish_confirmed` | 0.0623 | 0.0780 | +0.0157 |
| `z_fused` | `borderline_down` vs `bearish_confirmed` | 0.2234 | 0.2916 | +0.0682 |
| `z_fused` | `borderline_up` vs `bullish_confirmed` | 0.3193 | 0.2901 | -0.0291 |
| `z_long` | `clean_intact` vs `borderline_down` | 0.1476 | 0.1999 | +0.0523 |
| `z_long` | `clean_intact` vs `borderline_up` | 0.1216 | 0.1514 | +0.0298 |
| `z_fused` | `clean_intact` vs `borderline_down` | 0.2457 | 0.1854 | -0.0603 |
| `z_fused` | `clean_intact` vs `borderline_up` | 0.1400 | 0.1225 | -0.0176 |

This is a mixed but useful signal:

- `z_fused` improves the `borderline_down` vs `bearish_confirmed` separation materially.
- `z_long_proj` improves both borderline-vs-confirmed distances.
- `z_long` improves `borderline_down`, but `borderline_up` moves closer to `bullish_confirmed`.
- `z_fused` clean-intact vs borderline distances shrink, meaning the classifier improvement may partly come from keeping borderline intact closer to the broader intact family.

### Bucket-Level Interpretation

The bucket readout supports the auxiliary-head direction, but not as a complete fix.

Positive:

- The actual borderline misclassification rate drops substantially.
- The improvement is visible in `z_fused`, which is the classifier input.
- Same-symbol rates do not rise.

Caveat:

- Borderline centroid geometry is not uniformly better across directions.
- `borderline_up` remains structurally ambiguous and can still drift toward the confirmed bullish family.

Seed 42 should therefore be checked on the same bucket diagnostics, not only global NN geometry.

## Interpretation

This is a stronger result than pure schedule tuning for seed 41.

Compared with `SupCon-dominant warmup v1 seed_41 epoch_010`:

- `z_long` test label agreement improves from `0.8102` to `0.8220`
- `z_long` test symbol agreement improves from `0.2826` to `0.2745`
- `z_fused` test label agreement improves from `0.8361` to `0.8418`
- `z_fused` test symbol agreement improves from `0.3012` to `0.2913`
- effective long/short rises from `0.5465` to `1.0714`

The auxiliary head appears to do what it was designed to do: make `z_long` actively useful without relying on a warmup schedule.

## Remaining Question

## Seed 42 Readout

Seed 42 used the same setup.

### Training Dynamics

| epoch | train_ce | train_supcon | train_long_aux | val_ce | val_supcon | val_long_aux | val_macro_f1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.4411 | 3.1370 | 0.5759 | 0.3679 | 3.0220 | 0.4001 | 0.8526 |
| 5 | 0.3357 | 2.9826 | 0.3766 | 0.3068 | 2.9842 | 0.3585 | 0.8743 |
| 10 | 0.2934 | 2.9491 | 0.3449 | 0.2907 | 2.9582 | 0.3349 | 0.8820 |
| 20 | 0.2519 | 2.9134 | 0.3152 | 0.2868 | 2.9450 | 0.3242 | 0.8869 |
| 30 | 0.2258 | 2.8898 | 0.2968 | 0.2884 | 2.9398 | 0.3164 | 0.8852 |
| 40 | 0.2087 | 2.8705 | 0.2846 | 0.3058 | 2.9292 | 0.3120 | 0.8825 |
| 50 | 0.1944 | 2.8560 | 0.2732 | 0.3112 | 2.9236 | 0.3081 | 0.8901 |

SupCon and long auxiliary losses decrease from the start, matching seed 41.

### Val Geometry Sweep

| epoch | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.7926 | 0.3255 | 0.7957 | 0.3379 | 0.7940 | 0.2650 | 0.8266 | 0.3111 | 0.6855 |
| 10 | 0.7639 | 0.3221 | 0.8058 | 0.3218 | 0.8013 | 0.2760 | 0.8361 | 0.3248 | 0.8545 |
| 20 | 0.7592 | 0.2948 | 0.8142 | 0.3024 | 0.8152 | 0.2786 | 0.8411 | 0.3101 | 1.0290 |
| 30 | 0.7276 | 0.2969 | 0.8176 | 0.2862 | 0.8193 | 0.2684 | 0.8476 | 0.3248 | 1.1640 |
| 40 | 0.7288 | 0.3021 | 0.8229 | 0.2820 | 0.8187 | 0.2661 | 0.8413 | 0.3034 | 1.3048 |
| 50 | 0.7169 | 0.3371 | 0.8270 | 0.2923 | 0.8270 | 0.2661 | 0.8474 | 0.3296 | 1.4036 |

Validation-first selection again favors:

```text
epoch_020.pt
```

Rationale:

- `epoch_030` has better `z_fused` label agreement but worse `z_fused` symbol agreement.
- `epoch_040` improves `z_long`, but effective long/short rises to `1.3048`.
- `epoch_020` is the most balanced checkpoint.

### Test Check

| checkpoint | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `epoch_020.pt` | 0.7714 | 0.2759 | 0.8125 | 0.2754 | 0.8114 | 0.2596 | 0.8416 | 0.2842 | 1.0729 |
| `epoch_030.pt` | 0.7364 | 0.2922 | 0.8193 | 0.2710 | 0.8205 | 0.2575 | 0.8419 | 0.3095 | 1.2086 |
| `epoch_040.pt` | 0.7315 | 0.2913 | 0.8205 | 0.2706 | 0.8176 | 0.2583 | 0.8392 | 0.2888 | 1.3648 |

Seed 42 is mixed-positive:

- `epoch_020.pt` is balanced and clean.
- It does not clearly beat dominant warmup seed 42 on global `z_long` / `z_fused` metrics.
- It does maintain the intended branch activation without becoming symbol-heavy.

### Bucket Check

| run | misclassified_rate | avg_nn_intact_rate | avg_nn_confirmed_rate | avg_nn_same_symbol_rate |
|---|---:|---:|---:|---:|
| `long_v1 / z_fused` | 0.2818 | 0.6847 | 0.3153 | 0.2814 |
| `branch_ce_aux_v1 seed41 e20 / z_fused` | 0.2134 | 0.6918 | 0.3082 | 0.2792 |
| `branch_ce_aux_v1 seed42 e20 / z_fused` | 0.2668 | 0.6824 | 0.3176 | 0.2848 |
| `branch_ce_aux_v1 seed42 e20 / z_long` | 0.2668 | 0.6700 | 0.3300 | 0.2606 |

Seed 42 bucket result is weaker than seed 41 but still improves over canonical `long_v1` on borderline misclassification.

Bucket centroid distances:

| embedding | pair | `long_v1` | `branch seed41 e20` | `branch seed42 e20` |
|---|---|---:|---:|---:|
| `z_fused` | `borderline_down` vs `bearish_confirmed` | 0.2234 | 0.2916 | 0.2650 |
| `z_fused` | `borderline_up` vs `bullish_confirmed` | 0.3193 | 0.2901 | 0.2570 |
| `z_long_proj` | `borderline_down` vs `bearish_confirmed` | 0.0632 | 0.0707 | 0.0823 |
| `z_long_proj` | `borderline_up` vs `bullish_confirmed` | 0.0623 | 0.0780 | 0.0755 |
| `z_long` | `borderline_down` vs `bearish_confirmed` | 0.2917 | 0.2974 | 0.3308 |
| `z_long` | `borderline_up` vs `bullish_confirmed` | 0.4199 | 0.3100 | 0.2648 |

This confirms the same pattern:

- `z_long_proj` borderline-vs-confirmed separation improves for both directions.
- `z_fused` improves `borderline_down` but worsens `borderline_up`.
- `z_long` also worsens `borderline_up`.

## Final Interpretation

`long branch CE aux v1` is a real improvement over schedule-only tuning for seed 41, but seed 42 is more conservative:

- Seed 41 clearly beats dominant warmup.
- Seed 42 is roughly comparable to dominant warmup globally and slightly improves borderline misclassification over canonical `long_v1`.
- Both seeds show immediate SupCon activation and active `z_long` usage.
- The remaining weakness is directional asymmetry in borderline buckets, especially `borderline_up`.

Decision:

```text
Select seed_41/epoch_020.pt as the Stage 1A downstream-transfer checkpoint.
Treat seed 42 bucket asymmetry as a model-improvement item for the next Stage 1A iteration.
```

Selected checkpoint:

```text
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt
```

Reason:

- It is the strongest global `z_long` improvement observed so far.
- It improves `z_fused` and borderline misclassification over canonical `long_v1`.
- It keeps symbol shortcut suppressed.
- It is sufficient to test whether the Stage 1A representation transfers to downstream tasks, which is now more important than another local Stage 1A refinement loop.
