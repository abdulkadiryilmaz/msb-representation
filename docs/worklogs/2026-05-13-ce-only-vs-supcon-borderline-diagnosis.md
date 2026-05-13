# Worklog: 2026-05-13 — CE-only vs SupCon Borderline Diagnosis

**Status**: Complete  
**Scope**: Stage 1A `z_fused` borderline intact diagnosis

---

## Question

Previous fusion/head diagnosis showed that misclassified borderline intact examples are already close to same-direction confirmed break families in `z_fused`.

Open question:

```text
Is this confirmed-family drift already present in CE-only,
or did branch-aware SupCon amplify it?
```

This matters because the next experiment should target the actual source:

- if CE-only already has the same issue, it may be inherent to the label / geometry setup
- if SupCon amplifies it, the next objective should preserve pressure direction without pulling borderline pressure into confirmed break clusters

---

## Artifacts

CE-only diagnosis:

- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/fusion_head_diagnosis/z_fused_test_borderline/`

SupCon long v1 diagnosis:

- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/fusion_head_diagnosis/z_fused_test_borderline/`

Script:

- `scripts/diagnose_stage1a_fusion_head.py`
- `scripts/probe_stage1a_structural_state.py`

---

## Overall Comparison

Subset:

```text
confirmed_state = intact
break_maturity = borderline
```

| Metric | CE-only | SupCon long v1 |
|---|---:|---:|
| count | `731` | `731` |
| misclassified | `185` | `206` |
| misclassified rate | `0.2531` | `0.2818` |
| avg confidence | `0.7861` | `0.8033` |
| avg prob margin | `0.5794` | `0.6117` |
| avg NN intact rate | `0.6958` | `0.6847` |
| avg NN confirmed rate | `0.3042` | `0.3153` |
| avg NN same-symbol rate | `0.6003` | `0.2814` |

Readout:

- SupCon long v1 slightly increases borderline misclassification rate
- average confirmed-neighbor rate rises slightly
- same-symbol neighbor rate drops dramatically

So SupCon long v1 improves cross-symbol structure alignment, but may slightly worsen confirmed-state boundary behavior for borderline intact examples.

---

## Correct vs Misclassified

### CE-only

Correct:

| Metric | Value |
|---|---:|
| count | `546` |
| confidence | `0.8148` |
| prob margin | `0.6375` |
| `p_intact` | `0.8148` |
| NN intact rate | `0.8249` |
| NN confirmed rate | `0.1751` |
| NN same-direction rate | `0.4529` |
| NN same-symbol rate | `0.5978` |

Misclassified:

| Metric | Value |
|---|---:|
| count | `185` |
| confidence | `0.7014` |
| prob margin | `0.4082` |
| `p_intact` | `0.2932` |
| NN intact rate | `0.3146` |
| NN confirmed rate | `0.6854` |
| NN same-direction rate | `0.7859` |
| NN same-symbol rate | `0.6076` |

### SupCon long v1

Correct:

| Metric | Value |
|---|---:|
| count | `525` |
| confidence | `0.8329` |
| prob margin | `0.6718` |
| `p_intact` | `0.8329` |
| NN intact rate | `0.8339` |
| NN confirmed rate | `0.1661` |
| NN same-direction rate | `0.4577` |
| NN same-symbol rate | `0.2815` |

Misclassified:

| Metric | Value |
|---|---:|
| count | `206` |
| confidence | `0.7280` |
| prob margin | `0.4586` |
| `p_intact` | `0.2695` |
| NN intact rate | `0.3044` |
| NN confirmed rate | `0.6956` |
| NN same-direction rate | `0.8238` |
| NN same-symbol rate | `0.2811` |

The bimodal pattern exists in both runs:

```text
correct borderline -> mostly intact-family neighbors
misclassified borderline -> mostly same-direction confirmed-family neighbors
```

SupCon does not create the pattern from nothing. It appears already present in CE-only.

However, SupCon strengthens cross-symbol alignment and slightly increases the confirmed-family pull for the misclassified subset.

---

## Transition Analysis

The borderline population is identical across both runs: `731` examples.

Correctness transition:

| CE-only | SupCon long v1 | Count |
|---|---|---:|
| correct | correct | `478` |
| wrong | wrong | `138` |
| wrong | correct | `47` |
| correct | wrong | `68` |

The most useful comparison is the changed groups:

| Group | Count | CE-only NN confirmed | SupCon NN confirmed | Delta |
|---|---:|---:|---:|---:|
| became wrong | `68` | `0.3412` | `0.5824` | `+0.2412` |
| became correct | `47` | `0.5532` | `0.3638` | `-0.1894` |
| wrong both | `138` | `0.7304` | `0.7514` | `+0.0210` |
| correct both | `478` | `0.1515` | `0.1467` | `-0.0048` |

Interpretation:

- examples that became wrong under SupCon moved substantially toward confirmed-family neighbors
- examples that became correct moved away from confirmed-family neighbors
- stable wrong examples were already confirmed-family-heavy in CE-only
- stable correct examples remain intact-family-heavy in both

This is a strong signal that the geometry shift is meaningful, not only classifier noise.

---

## Neighbor Family Readout

Misclassified CE-only top families:

- `bearish / confirmed / down / holding`: `46`
- `bearish / confirmed / down / reverted`: `46`
- `bullish / confirmed / up / holding`: `38`
- `bullish / confirmed / up / reverted`: `38`

Misclassified SupCon long v1 top families:

- `bearish / confirmed / down / holding`: `65`
- `bullish / confirmed / up / holding`: `43`
- `bearish / confirmed / down / reverted`: `36`
- `bullish / confirmed / up / reverted`: `35`

Both runs put misclassified borderline examples near confirmed directional families.

Difference:

- CE-only does this with high same-symbol rate
- SupCon does this with much lower same-symbol rate

So SupCon appears to replace symbol-local clustering with cross-symbol structural clustering.

This is mostly good, but for the borderline subset it can over-align near-confirmation pressure with confirmed breaks.

---

## Diagnosis

The current evidence supports a nuanced conclusion:

1. The borderline/confirmed ambiguity is inherent to the current Stage 1A setup.
   - CE-only already misclassifies `185 / 731`
   - CE-only misclassified examples already have high NN confirmed rate: `0.6854`

2. SupCon long v1 improves cross-symbol alignment.
   - same-symbol rate drops from `0.6003` to `0.2814`
   - this is aligned with the representation-learning goal

3. SupCon long v1 slightly worsens borderline confirmed-state boundary.
   - misclassified count rises from `185` to `206`
   - examples that became wrong moved from `0.3412` to `0.5824` NN confirmed rate

Therefore:

```text
SupCon is not the root cause,
but it can amplify confirmed-family pull for borderline pressure examples.
```

---

## Pre-change Frozen Structural Probe

Before changing the model, we measured whether current frozen latents already expose the maturity distinction:

```text
borderline_intact
vs
confirmed_break
```

This is intentionally not a new training experiment. It is a diagnostic probe over existing latent exports.

Artifacts:

- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/structural_probe/z_fused_maturity_binary/`
- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/structural_probe/z_long_maturity_binary/`
- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/structural_probe/z_fused_maturity_binary/`
- `data/stage1a/binance/15m/checkpoints/ce_only/analysis/structural_probe/z_long_maturity_binary/`

Target distribution:

| Split | borderline intact | confirmed break |
|---|---:|---:|
| val/train | `618` | `4680` |
| test/eval | `731` | `4897` |

Majority baseline on test:

| Metric | Value |
|---|---:|
| accuracy | `0.8701` |
| macro F1 | `0.4653` |

Accuracy is not useful by itself here because the target is highly imbalanced. Macro F1 is the main readout.

### Probe Results

| Checkpoint | Embedding | Linear macro F1 | MLP macro F1 |
|---|---|---:|---:|
| CE-only | `z_long` | `0.4379` | `0.4653` |
| CE-only | `z_fused` | `0.7317` | `0.7434` |
| SupCon long v1 | `z_long` | `0.6995` | `0.7134` |
| SupCon long v1 | `z_fused` | `0.7158` | `0.7529` |

Readout:

- CE-only `z_long` does not expose the maturity distinction well.
- SupCon long v1 makes `z_long` much more maturity-readable.
- `z_fused` already carries maturity information in both runs.
- MLP only modestly improves over linear for `z_fused`, so the signal is not purely nonlinear/hidden.

### Confusion Matrix Readout

SupCon long v1 `z_fused`, linear probe:

| True \\ Pred | borderline intact | confirmed break |
|---|---:|---:|
| borderline intact | `626` | `105` |
| confirmed break | `931` | `3966` |

SupCon long v1 `z_fused`, MLP probe:

| True \\ Pred | borderline intact | confirmed break |
|---|---:|---:|
| borderline intact | `388` | `343` |
| confirmed break | `252` | `4645` |

Interpretation:

- A balanced linear probe can recover most borderline intact examples, but pays with many confirmed-break false positives.
- MLP becomes more conservative: it protects confirmed-break precision but misses more borderline intact examples.
- This means the maturity signal exists, but the current geometry does not provide a clean, high-margin boundary.

### Directional Breakdown

SupCon long v1 `z_fused`, linear probe:

| Direction / Target | Pred borderline | Pred confirmed |
|---|---:|---:|
| down borderline intact | `282` | `51` |
| down confirmed break | `504` | `2054` |
| up borderline intact | `263` | `40` |
| up confirmed break | `427` | `1912` |

SupCon long v1 `z_fused`, MLP probe:

| Direction / Target | Pred borderline | Pred confirmed |
|---|---:|---:|
| down borderline intact | `139` | `194` |
| down confirmed break | `108` | `2450` |
| up borderline intact | `187` | `116` |
| up confirmed break | `144` | `2195` |

This is the key detail for the next objective:

```text
same-direction maturity is partially readable,
but not cleanly separated with enough margin.
```

So the next model change should not only add a head-side readout. It should improve representation geometry so that:

```text
down-pressure borderline intact
is close to down-structure semantics,
but not collapsed into confirmed bearish break
```

and symmetrically for the up direction.

---

## Experiment Implication

The next experiment should not abandon branch-aware SupCon.

It solved a real problem:

```text
z_long / z_fused symbol shortcut reduction
```

But the next objective should distinguish:

```text
same-direction pressure
vs
same-direction confirmed break
```

This points toward a factor-aware design:

- keep `confirmed_state` CE
- preserve pressure direction as an explicit auxiliary/readout axis
- add a mechanism that prevents borderline intact pressure from being treated as equivalent to confirmed break positives

Candidate directions:

1. Factor-aware auxiliary objective
   - `confirmed_state`: intact / bullish / bearish
   - `pressure_state`: neutral / up / down / non-intact
   - optional `break_maturity`: clean / wick / borderline / confirmed / reverted

2. Maturity-aware contrastive rule
   - same direction is not enough for positive pairing
   - `borderline intact` and `confirmed break` should be near but not collapsed

3. Borderline-aware CE weighting
   - smaller change
   - may reduce early confirmed predictions
   - risk: may suppress useful pressure geometry if used alone

---

## Decision

The next experiment should be planned around:

```text
pressure-aware but maturity-sensitive representation learning
```

Success criteria should explicitly include:

- preserve low symbol agreement from SupCon long v1
- preserve pressure readability
- reduce borderline intact → same-direction confirmed misclassification
- reduce confirmed-family NN rate for misclassified borderline subset
- avoid collapsing all borderline pressure into clean intact
