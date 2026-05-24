# Stage 1B H8 Fresh Break v2 Readout

**Tarih**: 2026-05-23
**Durum**: Completed
**Amaç**: Stage 1B v1 `future_break_direction` label'ındaki fresh break / already-broken karışımını ayırmak ve H8 fresh-break hedefinin öğrenilebilirliğini ölçmek.

---

## Background

Stage 2 görsel incelemesi ve ardından yapılan semantics audit, H8 directional label'ların büyük kısmının fresh break olmadığını gösterdi:

- H8 directional label içinde already-broken oranı: `%66.00`
- H8 predictor directional çıktıları içinde already-broken oranı: `%71.34`

Bu nedenle v1 predictor, "henüz trigger oluşacak" sinyali değil, çoğunlukla "break seviyesi zaten aşılmış / continuation state" sinyali olarak okunmalı.

---

## Label v2 Change

`scripts/generate_stage1b_forward_labels.py` v1 kolonlarını koruyarak şu v2 alanları üretecek şekilde güncellendi:

```text
current_close
h{H}_break_anchor_status
h{H}_fresh_break_direction
h{H}_break_semantic_label
```

Status tanımı:

```text
bullish fresh_candidate:
  current_close < bull_level

bullish already_broken:
  current_close >= bull_level

bearish fresh_candidate:
  current_close > bear_level

bearish already_broken:
  current_close <= bear_level
```

Üretilen artifact'lar:

```text
stage1b_forward_labels_train_h4_h8_h16_v2.parquet
stage1b_forward_labels_val_h4_h8_h16_v2.parquet
stage1b_forward_labels_test_h4_h8_h16_v2.parquet
```

---

## H8 Label Distribution

### Train

| Label | Count |
|---|---:|
| none | 106387 |
| bullish | 12441 |
| bearish | 12207 |
| ambiguous | 5 |

### Val

| Label | Count |
|---|---:|
| none | 7115 |
| bullish | 871 |
| bearish | 844 |
| ambiguous | 2 |

### Test

| Label | Count |
|---|---:|
| none | 7872 |
| bullish | 856 |
| bearish | 967 |
| ambiguous | 1 |
| insufficient_future | 8 |

Yorum: Fresh-break hedefi v1 H8 direction hedefinden çok daha dengesiz. Test split'te actionable directional support `1823 / 9695` civarına düşüyor.

---

## Probe Result

Target:

```text
h8_fresh_break_direction
```

Embedding:

```text
z_fused
```

Dropped:

```text
ambiguous
insufficient_future
```

| Split | Model | Macro F1 | Accuracy | Majority Macro F1 |
|---|---|---:|---:|---:|
| Val | linear | 0.3952 | 0.4664 | 0.2975 |
| Val | MLP | 0.2982 | 0.8058 | 0.2975 |
| Test | linear | 0.3738 | 0.4462 | 0.2987 |
| Test | MLP | 0.2987 | 0.8120 | 0.2987 |

Linear probe class-balanced olduğu için directional recall yakalıyor, fakat precision düşük. MLP çoğunluk sınıfına çöküyor.

---

## Predictor v2a — Unweighted

Checkpoint:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h8_fresh_break_v2_z_fused_unweighted
```

Sonuç:

| Metric | Test |
|---|---:|
| recon macro F1 | 0.2987 |
| break macro F1 | 0.4481 |
| direction macro F1 | 0.7074 |

Break confusion:

```text
none  -> [7872,    0]
break -> [1823,    0]
```

Yorum: Unweighted objective tamamen `none` sınıfına çöküyor. Bu hedefte class-balanced break loss gerekli.

---

## Predictor v2b — Balanced Break Loss

Code change:

```text
--balanced-break-loss
```

Checkpoint:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h8_fresh_break_v2_z_fused_balanced_break
```

Best val epoch:

```text
11
```

Default threshold `0.50` test:

| Metric | Test |
|---|---:|
| recon macro F1 | 0.4213 |
| break macro F1 | 0.5458 |
| direction macro F1 | 0.7234 |
| break pred rate | 0.4981 |

Break confusion:

```text
none  -> [4384, 3488]
break -> [ 482, 1341]
```

Yorum: Balanced loss collapse'ı kırıyor, fakat threshold `0.50` çok fazla false-positive fresh break üretiyor.

---

## Threshold Sweep

Val üzerinde seçilen threshold:

```text
0.57
```

Val:

| Metric | Value |
|---|---:|
| recon macro F1 | 0.4414 |
| break macro F1 | 0.5864 |
| direction macro F1 | 0.7167 |
| break pred rate | 0.3082 |

Test at selected threshold:

| Metric | Value |
|---|---:|
| recon macro F1 | 0.4472 |
| break macro F1 | 0.5933 |
| direction macro F1 | 0.7234 |
| break pred rate | 0.3007 |
| recon accuracy | 0.6715 |
| break accuracy | 0.6996 |

Test break confusion:

```text
none  -> [5870, 2002]
break -> [ 910,  913]
```

Test reconstructed confusion:

```text
none    -> [5870, 1129,  873]
bullish -> [ 424,  339,   93]
bearish -> [ 486,  180,  301]
```

---

## Interpretation

Fresh-break prediction is materially harder than v1 H8 forward direction prediction.

The v1 H8 predictor looked strong partly because the target included already-broken / continuation states. Once the target is restricted to fresh trigger formation, the signal weakens:

- break occurrence is highly imbalanced
- linear separability drops
- unweighted predictor collapses to `none`
- balanced predictor recovers recall but produces many false positives

This does not invalidate Stage 1A transfer. It narrows what Stage 1B can currently claim:

```text
Stage 1B v1:
  readable near-term structural state / continuation signal

Stage 1B v2 fresh:
  weak but nonzero fresh-trigger signal
```

---

## Decision

Stage 2 should not use v1 `p_break_8` as a direct fresh-entry trigger.

For TradePlan path, Stage 1B needs either:

1. a fresh-break predictor with stronger occurrence precision, or
2. a two-mode output:

```text
fresh_trigger_candidate
already_broken_continuation
no_directional_setup
```

The second option is likely more faithful to observed data, because already-broken / continuation cases are common and may still be tradable under retest/continuation rules, but they require a different Stage 2 entry model.

---

## Next

1. Add a Stage 1B v2 output contract separating fresh and continuation modes.
2. Run a proximity/error audit on the balanced fresh predictor at threshold `0.57`.
3. Decide whether Stage 2 v1 should:
   - consume only fresh predictor output, or
   - explicitly branch into fresh-break vs continuation actionability.
