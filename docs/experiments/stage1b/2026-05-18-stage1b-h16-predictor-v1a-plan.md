# Stage 1B H16 Predictor v1a Plan

**Date**: 2026-05-18
**Status**: planned
**Stage 1A source checkpoint**: `ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`
**Label contract**: `docs/foundations/stage1b-forward-labels.md`
**Previous readout**: `2026-05-18-stage1b-forward-direction-probe-v1-readout.md`

## Goal

Stage 1B için ilk gerçek predictor baseline'ını kurmak.

Probe aşamasında Stage 1A latent'inin future break direction sinyali taşıdığı görüldü. Predictor v1a'nın amacı bu sinyali kalıcı bir Stage 1B forecast artifact'ına çevirmek, fakat Stage 2 / TradePlan alanına geçmemektir.

V1a sorusu:

> Önümüzdeki 16 bar içinde structural break olacak mı? Olacaksa yönü ne?

15m timeframe'de H16 yaklaşık 4 saattir.

## Why H16 First

Probe sonuçları:

| Target | Embedding | Linear macro F1 |
|---|---|---:|
| `h16_future_break_direction` 3-class | `z_fused` | 0.6768 |
| `h32_future_break_direction` 3-class | `z_fused` | 0.6183 |
| `h48_future_break_direction` 3-class | `z_fused` | 0.5807 |

H16 daha temiz ve daha güçlü ilk sinyal verdi. Bu nedenle ilk predictor baseline H16-only olacak.

H32 şimdilik dışarıda bırakılıyor. Gerekçe:

- H32 stratejik olarak değerli olabilir.
- Ancak H32 eklemek H16 sinyalini bozabilir.
- Önce H16-only baseline kurulmalı, ardından H16 + H32 multi-horizon ablation ile gerçek katkı ölçülmeli.

## Predictor Scope

V1a, probe'daki tek 3-class direction classifier'ı aynen kopyalamaz. Break occurrence ve direction ayrılır.

Output:

```text
p_break_16
p_bullish_given_break_16
p_bearish_given_break_16
```

Interpretation:

```text
p_break_16:
  H16 içinde structural break olma olasılığı

p_bullish_given_break_16 / p_bearish_given_break_16:
  break oluştuğu koşulda yön olasılığı
```

Bu output TradePlan değildir. Stage 2 daha sonra şu soruyu soracaktır:

> Bu predicted break trade edilmeye değer mi?

## Targets

Stage 1B forward label artifact'ından türetilir:

```text
h16_future_break_direction
```

Mapping:

```text
none    -> break_occurs_16 = 0
bullish -> break_occurs_16 = 1, direction_16 = bullish
bearish -> break_occurs_16 = 1, direction_16 = bearish
```

Excluded:

```text
ambiguous
insufficient_future
```

## Objective

Loss:

```text
loss =
  break_loss
  + direction_loss
```

Where:

```text
break_loss = CE(break_occurs_16)
direction_loss = CE(direction_16) only for bullish/bearish samples
```

Direction loss is masked for `none` samples.

Initial weights:

```text
break_loss_weight = 1.0
direction_loss_weight = 1.0
```

If class imbalance dominates, weights may be introduced, but v1a should start simple and document any change.

## Inputs

Initial input:

```text
z_fused
```

Rationale:

- strongest probe signal
- already combines short and long branch information
- simplest baseline

Planned ablations after baseline:

```text
z_short
z_long
z_fused + logits
z_fused + probs
z_short + z_long + logits
```

## Model

Initial predictor should be intentionally small:

```text
MLP(
  input_dim,
  hidden_dim=64 or 128,
  dropout=0.1,
  outputs:
    break_logits: 2
    direction_logits: 2
)
```

No recurrence, no temporal encoder, no raw OHLCV input in v1a. Stage 1B v1a tests the frozen Stage 1A representation, not a new price-window model.

## Evaluation

Primary split protocol:

- train on train split
- select model / epoch on val split
- final confirmation on test split

Metrics:

### Break occurrence

```text
accuracy
macro F1
precision / recall for break class
calibration / probability distribution if available
```

### Direction given break

Only bullish/bearish samples:

```text
macro F1
accuracy
confusion matrix
```

### Reconstructed 3-class forecast

Combine outputs:

```text
if p_break_16 < threshold:
    none
else:
    argmax(direction)
```

Evaluate against:

```text
none / bullish / bearish
```

This allows comparison with the probe readout.

Threshold policy:

- default threshold: `0.5`
- report validation-tuned threshold separately if used
- do not tune threshold on test split

## Baselines

Must compare against:

1. Majority baseline
2. Probe result:
   - `z_fused` H16 3-class linear macro F1: `0.6768`
3. Classifier-output baseline if rerun:
   - `logits`
   - `probs`

The predictor does not need to beat the linear probe immediately to be useful, because it exposes calibrated `p_break` and conditional direction. But if it underperforms badly, model capacity / objective / thresholding must be reviewed before proceeding to H16 + H32.

## Success Criteria

Minimum useful result:

- break occurrence macro F1 meaningfully above majority baseline
- direction-given-break does not collapse to one class
- reconstructed 3-class result reasonably close to H16 linear probe

Strong result:

- reconstructed 3-class H16 macro F1 close to or above `0.6768`
- `p_break_16` separates `none` from directional break samples cleanly on val/test
- direction head maintains balanced bullish/bearish recall

Failure modes:

- predictor predicts break for almost everything
- predictor predicts `none` too often and misses directional examples
- direction head collapses to bullish or bearish
- validation good, test poor
- MLP underperforms linear probe with no calibration benefit

## Planned Follow-Up

After v1a:

1. Document H16-only readout.
2. If v1a is stable, run Stage 1B multi-horizon v1b:

```text
H16 + H32
```

v1b will test whether H32 adds useful delayed-setup information without degrading H16 performance.

Key segment for v1b:

```text
H16 = none
H32 = bullish/bearish
```

This segment represents setups that do not break immediately but may mature within the wider intraday horizon.
