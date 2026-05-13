# Worklog: 2026-05-13 — Factor-aware SupCon Design

**Status**: Implemented / smoke-tested  
**Scope**: Stage 1A objective design, next experiment after `ce_supcon_long_v1`  
**Proposed checkpoint**: `ce_supcon_long_factor_v1`

---

## Motivation

Current best candidate:

```text
ce_supcon_long_v1
```

Objective:

```text
CE(confirmed_state) + 0.05 * SupCon(z_long_proj, confirmed_state)
```

This run did several things right:

- improved `z_long` semantic structure
- reduced symbol shortcut compared with CE-only
- preserved readable directional pressure in latent probes
- made `z_long_proj` a useful branch-specific contrastive target

But the later borderline diagnosis exposed a specific failure mode:

```text
borderline intact pressure examples can be pulled into same-direction confirmed break families.
```

This is not a pure classifier-head problem. Misclassified borderline intact examples are already embedded near confirmed break neighbors in `z_fused`.

Therefore the next experiment should not remove SupCon. It should make SupCon more factor-aware.

---

## Diagnosis Being Addressed

The current SupCon positive rule uses the coarse `confirmed_state` label.

That means all `intact` examples are treated as one positive family:

```text
clean intact
wick-sweep intact
borderline up-pressure intact
borderline down-pressure intact
mixed pressure intact
```

This is too coarse for the problem now visible in the latent geometry.

At the same time, making direction alone the positive rule would also be wrong:

```text
intact + down_pressure + borderline
bearish + confirmed + down
```

These two are semantically related, but they should not collapse into the same state cluster.

Desired relation:

```text
near enough to preserve pressure semantics,
separate enough to preserve confirmed_state boundary.
```

---

## Design Goal

The new experiment should learn a representation that is:

```text
pressure-aware but maturity-sensitive
```

In concrete terms:

- preserve the useful directionality learned by `ce_supcon_long_v1`
- keep symbol agreement low
- prevent borderline intact pressure from becoming equivalent to confirmed break
- avoid collapsing all borderline pressure into clean intact

---

## Proposed Objective

Base objective remains:

```text
CE(confirmed_state)
```

Keep branch-aware SupCon target:

```text
z_long_proj
```

Change the contrastive positive rule from coarse label equality to factor-aware equality:

```text
SupCon(z_long_proj, factor_target)
```

Add conservative auxiliary supervision:

```text
lambda_pressure * CE(pressure_state)
lambda_maturity * CE(maturity_target)
```

First proposed loss:

```text
loss =
  CE(confirmed_state)
  + 0.05 * SupCon(z_long_proj, factor_target)
  + 0.01 * CE(pressure_state)
  + 0.01 * CE(maturity_target)
```

The auxiliary weights are intentionally small because `pressure_labels_v1` is validated as diagnostic/auxiliary, not primary ground truth.

---

## Factor Target for SupCon

The important design decision is the `factor_target`.

It should avoid treating all `intact` examples as equivalent, while not fragmenting the batch so much that no positive pairs remain.

Recommended first version:

```text
factor_target =
  clean_intact
  wick_sweep_up
  wick_sweep_down
  borderline_up
  borderline_down
  borderline_mixed
  bullish_confirmed
  bearish_confirmed
  bullish_weak_or_choppy
  bearish_weak_or_choppy
```

Observed distribution on existing `ce_supcon_long_v1` exports:

| Factor target | Val count | Test count |
|---|---:|---:|
| `clean_intact` | `2468` | `2855` |
| `bearish_confirmed` | `2386` | `2558` |
| `bullish_confirmed` | `2294` | `2339` |
| `wick_sweep_down` | `543` | `567` |
| `wick_sweep_up` | `447` | `557` |
| `borderline_down` | `299` | `333` |
| `borderline_up` | `252` | `303` |
| `borderline_mixed` | `143` | `192` |

Readout:

- the factor classes are imbalanced but not empty
- `borderline_mixed` is the thinnest class
- batch fragmentation is a real risk at `batch_size=32`
- the first implementation should record effective SupCon anchor counts / positive-pair counts per epoch

If positive-pair coverage is too sparse, the v1 fallback should be:

```text
increase batch size if feasible,
or merge borderline_mixed into a broader borderline family,
or use a factor-balanced sampler in v2.
```

Mapping:

| confirmed_state | pressure / maturity | factor target |
|---|---|---|
| intact | clean + neutral | `clean_intact` |
| intact | wick_sweep + up | `wick_sweep_up` |
| intact | wick_sweep + down | `wick_sweep_down` |
| intact | borderline + up | `borderline_up` |
| intact | borderline + down | `borderline_down` |
| intact | borderline + mixed | `borderline_mixed` |
| bullish | confirmed / strong_confirmed | `bullish_confirmed` |
| bearish | confirmed / strong_confirmed | `bearish_confirmed` |
| bullish | weak_or_choppy_confirmed | `bullish_weak_or_choppy` |
| bearish | weak_or_choppy_confirmed | `bearish_weak_or_choppy` |

High-vol intact variants should initially be folded into the nearest intact family:

- high-vol neutral -> `clean_intact`
- high-vol wick up/down -> `wick_sweep_up/down`
- high-vol ambiguous -> `borderline_mixed` or excluded from factor SupCon if needed

Rationale:

- same-direction borderline intact examples get positives with each other
- confirmed bullish/bearish examples get positives with confirmed peers
- borderline down is no longer a positive pair with bearish confirmed
- clean intact is no longer a positive pair with borderline pressure

This directly addresses the observed collapse.

---

## Maturity Target Definition

`maturity_target` is separate from `factor_target`.

It should be coarser, because the auxiliary head is only a low-weight readability regularizer.

Recommended first version:

```text
maturity_target =
  clean
  wick_sweep
  borderline
  confirmed
```

Mapping:

| Structural fields | maturity target |
|---|---|
| intact + `clean` | `clean` |
| intact + `high_vol_intact` + neutral | `clean` |
| intact + `wick_sweep` | `wick_sweep` |
| intact + `high_vol_intact_wick_sweep` | `wick_sweep` |
| intact + `borderline` | `borderline` |
| bullish/bearish + `confirmed` | `confirmed` |
| bullish/bearish + `strong_confirmed` | `confirmed` |
| bullish/bearish + `weak_or_choppy_confirmed` | `confirmed` for v1 |

Observed distribution:

| Maturity target | Val count | Test count |
|---|---:|---:|
| `confirmed` | `4680` | `4897` |
| `clean` | `2475` | `2867` |
| `wick_sweep` | `1059` | `1209` |
| `borderline` | `618` | `731` |

This means `maturity_target` is less fragmented than `factor_target`, which is appropriate for a low-weight auxiliary head.

Do not use `weak_or_choppy_confirmed` as a separate maturity target in v1 unless its support is explicitly measured and large enough.

---

## Relation to Auxiliary Heads

Auxiliary heads are not the main fix by themselves.

They serve two purposes:

1. keep `pressure_state` readable
2. keep `break_maturity` readable

But the main geometry intervention is the factor-aware SupCon positive mask.

This distinction matters:

```text
auxiliary heads measure/encourage factors,
factor-aware SupCon changes which samples are pulled together.
```

Do not use `holding_status` as a training target in v1.

Reason:

- manual review found `holding/reverted` semantically useful but threshold-sensitive
- it is better kept as an analysis/evaluation field for the first factor-aware run

---

## Implementation Surface

Expected code changes:

1. Dataset / batch metadata
   - expose factor labels derived from existing domain summary fields
   - keep the existing `labels` tensor as confirmed_state target

2. Model
   - keep existing encoders, fusion, classifier, `z_proj`, `z_long_proj`
   - add optional auxiliary heads on `z_fused`:
     - `pressure_logits`
     - `maturity_logits`

3. Trainer
   - compute existing CE on `outputs["logits"]`
   - compute factor-aware SupCon on `outputs["z_long_proj"]`
   - compute low-weight auxiliary CE losses when enabled
   - log effective SupCon coverage:
     - anchors with at least one positive
     - average positive count per anchor
     - optional per-factor positive count summary
   - record separate history fields:
     - `train_pressure_loss`
     - `train_maturity_loss`
     - `val_pressure_loss`
     - `val_maturity_loss`

4. SupCon loss
   - existing `SupConLoss` can be reused if `factor_target` is passed as labels
   - no need for a new weighted contrastive loss in v1
   - do not add weak-positive weighting yet; keep the first intervention simple and diagnosable

5. CLI
   - add a new positive mode / factor mode flag
   - add pressure/maturity auxiliary weight flags
   - checkpoint should record all factor-label config values

---

## Why Not Weighted Weak Positives Yet?

A more nuanced design could treat:

```text
borderline_down <-> bearish_confirmed
```

as weak-positive rather than neutral.

That may eventually be right, but it is harder to debug.

For the first experiment, use a simpler rule:

```text
same factor target = positive
different factor target = not positive
```

This lets the readout answer a clean question:

```text
Does preventing direct positive pairing between borderline intact and confirmed break reduce the collapse?
```

If this over-separates pressure from confirmed direction, a later v2 can introduce weak-positive or hierarchical SupCon.

---

## Success Criteria

The new run is successful only if it beats or matches `ce_supcon_long_v1` on the actual diagnosis axes.

Required:

- val macro F1 should not materially degrade
- `z_long` and `z_fused` symbol agreement should remain low
- `z_long` same-direction neighbor behavior should remain close to `ce_supcon_long_v1`
- pressure probe should remain above CE-only and close to `ce_supcon_long_v1`
- maturity binary probe should not degrade
- borderline intact misclassification rate should fall
- misclassified borderline intact NN confirmed rate should fall

Guardrails:

- do not accept a run that improves borderline accuracy by erasing pressure readability
- do not accept a run that restores symbol shortcut
- do not accept a run that fragments `z_long` so much that label agreement collapses

---

## Expected Readout Artefacts

Minimum post-run artefacts:

- val/test latent exports
- latent analysis for `z_short`, `z_long`, `z_fused`
- pressure probe
- maturity binary probe
- fusion/head borderline diagnosis
- manual NN review for borderline subset if metrics improve or regress ambiguously

Compare against:

- `ce_only`
- `ce_supcon_long_v1`

Primary comparison should focus on:

```text
z_long pressure preservation
z_fused confirmed-state boundary
borderline intact confirmed-family drift
```

---

## Decision

Proceed with:

```text
factor-aware SupCon on z_long_proj
+ low-weight pressure/maturity auxiliary heads
```

First implementation should stay conservative:

- no weighted weak positives yet
- no `holding_status` target yet
- no architecture change beyond auxiliary heads
- keep `confirmed_state` CE as primary objective

---

## Implementation Notes

Implemented files:

- `src/msb_repr/stage1a/analysis.py`
  - added structural training target builder:
    - `factor_targets`
    - `pressure_targets`
    - `maturity_targets`
  - added target name maps for factor and maturity labels
- `src/msb_repr/stage1a/dataset.py`
  - extended `Stage1AMeta` with optional factor / pressure / maturity targets
- `src/msb_repr/stage1a/model.py`
  - added optional `pressure_classifier`
  - added optional `maturity_classifier`
- `src/msb_repr/stage1a/trainer.py`
  - added `supcon_positive_mode="factor"`
  - added pressure/maturity auxiliary losses
  - added SupCon coverage logging:
    - anchor rate
    - average positive count per anchor
- `scripts/train_stage1a.py`
  - added CLI flags:
    - `--supcon-positive-mode factor`
    - `--pressure-loss-weight`
    - `--maturity-loss-weight`
  - default factor-aware checkpoint name:
    - `ce_supcon_long_factor_v1`

Implemented v1 target counts:

- factor classes: `8`
- pressure classes: `5`
  - `neutral`
  - `up_pressure`
  - `down_pressure`
  - `mixed_pressure`
  - `non_intact`
- maturity classes: `4`
  - `clean`
  - `wick_sweep`
  - `borderline`
  - `confirmed`

Smoke command:

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 1 \
  --batch-size 256 \
  --patience 1 \
  --checkpoint-dir /private/tmp/msb_factor_smoke_mps \
  --use-supcon \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode factor \
  --supcon-weight 0.05 \
  --pressure-loss-weight 0.01 \
  --maturity-loss-weight 0.01
```

Smoke result:

| Metric | Value |
|---|---:|
| device | `mps` |
| train loss | `0.73017` |
| val loss | `0.65360` |
| val macro F1 | `0.8511` |
| train SupCon anchor rate | `0.9999` |
| val SupCon anchor rate | `0.9999` |
| train avg positive count | `56.2115` |
| val avg positive count | `58.9069` |

Readout:

- factor-aware training path runs end-to-end
- auxiliary heads run end-to-end
- SupCon coverage logging works
- with `batch_size=256`, factor positive coverage is healthy in smoke mode

Note:

- full experiment should still monitor positive coverage under the actual batch size
- if using `batch_size=32`, batch fragmentation risk remains and should be read from history before interpreting metrics
