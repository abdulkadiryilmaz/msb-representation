# Worklog: 2026-05-13 — CE + SupCon Long Factor V1 Readout

**Status**: Rejected as candidate  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_factor_v1`  
**Baseline**: `ce_supcon_long_v1`

---

## Goal

This run tested:

```text
factor-aware SupCon on z_long_proj
+ low-weight pressure/maturity auxiliary heads
```

The intended fix was specific:

```text
reduce borderline intact -> same-direction confirmed break collapse
without losing pressure readability or low symbol shortcut.
```

Design reference:

- `docs/worklogs/2026-05-13-factor-aware-supcon-design.md`

Plan:

- `docs/experiments/ce_supcon/2026-05-13-ce-supcon-long-factor-v1-plan.md`

---

## Setup

Training command:

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 256 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_factor_v1 \
  --use-supcon \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode factor \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --pressure-loss-weight 0.01 \
  --maturity-loss-weight 0.01
```

Notes:

- device: `mps`
- batch size was increased from the historical default to reduce factor-target fragmentation
- `confirmed_state` CE remained the primary objective

---

## Training Summary

| Metric | `ce_supcon_long_v1` | `long_factor_v1` |
|---|---:|---:|
| best val macro F1 | `0.8900` | `0.8874` |
| best F1 epoch | `16` | `18` |
| last epoch | `29` | `24` |
| min val loss | `0.4533` | `0.5298` |
| min val loss epoch | `19` | `14` |

SupCon coverage:

| Metric | Value |
|---|---:|
| final train anchor rate | `0.9998` |
| final val anchor rate | `0.9999` |
| final train avg positives / anchor | `56.2208` |
| final val avg positives / anchor | `58.9069` |

Readout:

- factor-target fragmentation was not a runtime coverage problem at `batch_size=256`
- classifier F1 stayed usable but below `ce_supcon_long_v1`
- val loss is materially higher than baseline

---

## Latent Geometry

### Val

| Embedding | Metric | `ce_supcon_long_v1` | `long_factor_v1` |
|---|---|---:|---:|
| `z_long` | NN label agreement | `0.8096` | `0.8004` |
| `z_long` | NN symbol agreement | `0.3171` | `0.7466` |
| `z_fused` | NN label agreement | `0.8424` | `0.8469` |
| `z_fused` | NN symbol agreement | `0.2986` | `0.4987` |
| `z_long_proj` | NN label agreement | `0.8079` | `0.7945` |
| `z_long_proj` | NN symbol agreement | `0.2661` | `0.3690` |

### Test

| Embedding | Metric | `ce_supcon_long_v1` | `long_factor_v1` |
|---|---|---:|---:|
| `z_long` | NN label agreement | `0.8080` | `0.7994` |
| `z_long` | NN symbol agreement | `0.2987` | `0.7250` |
| `z_fused` | NN label agreement | `0.8425` | `0.8446` |
| `z_fused` | NN symbol agreement | `0.2784` | `0.5026` |
| `z_long_proj` | NN label agreement | `0.8067` | `0.7945` |
| `z_long_proj` | NN symbol agreement | `0.2621` | `0.3803` |

Readout:

- `z_fused` label agreement slightly improves
- `z_long` label agreement slightly degrades
- symbol shortcut returns strongly, especially in `z_long`
- this violates a primary guardrail

The key failure is:

```text
factor-aware SupCon improved some semantic probes,
but the learned z_long / z_fused geometry became much more symbol-bound.
```

---

## Pressure Probe

Test split linear macro F1:

| Embedding | `ce_only` | `ce_supcon_long_v1` | `long_factor_v1` |
|---|---:|---:|---:|
| `z_short` | `0.5227` | `0.5257` | `0.5497` |
| `z_long` | `0.2067` | `0.4842` | `0.5547` |
| `z_fused` | `0.5254` | `0.5407` | `0.5782` |

Readout:

- pressure readability improved across all three views
- the largest gain is in `z_long`
- this confirms that pressure auxiliary / factor targets injected the intended signal

But this is not enough for acceptance because symbol agreement regressed heavily.

---

## Maturity Probe

Target:

```text
borderline_intact vs confirmed_break
```

| Embedding | Metric | `ce_supcon_long_v1` | `long_factor_v1` |
|---|---|---:|---:|
| `z_long` | linear macro F1 | `0.6995` | `0.6681` |
| `z_long` | MLP macro F1 | `0.7134` | `0.4651` |
| `z_fused` | linear macro F1 | `0.7158` | `0.7341` |
| `z_fused` | MLP macro F1 | `0.7529` | `0.7695` |

Linear confusion matrix for `z_fused`:

| True \\ Pred | borderline intact | confirmed break |
|---|---:|---:|
| borderline intact | `621` | `110` |
| confirmed break | `823` | `4074` |

Readout:

- `z_fused` maturity readability improves
- `z_long` maturity readability degrades
- the intervention helped final fused readout more than branch geometry

This is not the desired tradeoff because the main SupCon target is `z_long_proj` and the branch geometry became symbol-heavy.

---

## Borderline Diagnosis

Subset:

```text
confirmed_state = intact
break_maturity = borderline
```

| Metric | `ce_supcon_long_v1` | `long_factor_v1` |
|---|---:|---:|
| count | `731` | `731` |
| misclassified count | `206` | `199` |
| misclassified rate | `0.2818` | `0.2722` |
| avg confidence | `0.8033` | `0.7882` |
| avg NN intact rate | `0.6847` | `0.6818` |
| avg NN confirmed rate | `0.3153` | `0.3182` |
| avg NN same-symbol rate | `0.2814` | `0.4970` |

Readout:

- borderline misclassification improves only slightly
- NN confirmed rate does not improve
- same-symbol neighbor rate rises sharply
- the targeted collapse was not meaningfully fixed

This is the decisive failure:

```text
The run reduced a few early confirmed predictions,
but did not reduce confirmed-family drift in the neighborhood geometry.
```

---

## Decision

Reject `ce_supcon_long_factor_v1` as a candidate.

Reason:

- symbol shortcut guardrail fails badly
- `z_long` symbol agreement rises from `0.2987` to `0.7250` on test
- `z_fused` symbol agreement rises from `0.2784` to `0.5026`
- borderline NN confirmed rate does not decrease

What worked:

- pressure probe improved substantially
- `z_fused` maturity probe improved
- factor target coverage was healthy at `batch_size=256`

What failed:

- factor-aware labels alone did not preserve cross-symbol structural geometry
- the larger batch / factor objective combination likely allowed symbol-specific geometry to re-emerge
- the exact borderline collapse mechanism was not solved

---

## Next Direction

Do not continue this exact objective as the next best candidate.

Useful next questions:

1. Is the regression caused by the factor target itself, or by the batch-size change?
2. Would factor-aware SupCon without auxiliary heads keep pressure gains but reduce symbol drift?
3. Would keeping coarse label SupCon and adding only low-weight auxiliary heads preserve `ce_supcon_long_v1` geometry?
4. Does factor-aware SupCon need explicit cross-symbol positive constraints?

Recommended next controlled ablation:

```text
ce_supcon_long_aux_v1:
  CE + SupCon(z_long_proj, confirmed_state)
  + low-weight pressure/maturity auxiliary heads
```

This keeps the successful `ce_supcon_long_v1` SupCon pairing rule and isolates whether auxiliary heads alone can improve pressure/maturity behavior without reviving symbol shortcut.
