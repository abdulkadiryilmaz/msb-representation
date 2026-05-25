# 2026-05-25 - Stage 1B H16 Outcome-Joint Predictor v1 Readout

## Amaç

H16 joint event-sequence v1 modeli `fresh_break` event'ini önceliklendiriyordu.

Bu, bazı trade-relevant reversal pencerelerinde yanlış semantik üretiyordu:

```text
index 452:
  event-joint target   = fresh_break_bullish
  forward path outcome = bullish_to_bearish_reversal
```

Bu deney, target'ı ilk event yerine post-break outcome'a taşır:

```text
h16_post_break_outcome + h16_dominant_forward_direction -> outcome_joint
```

Amaç, Stage 2'ye daha trade-context uyumlu H16 sinyali verilip verilemeyeceğini test etmektir.

## Target Mapping

Outcome-joint sınıfları aynı joint label space'i kullanır:

```text
no_event
fresh_break_bullish
fresh_break_bearish
continuation_bullish
continuation_bearish
reversal_bullish
reversal_bearish
```

Bu run'da outcome alanı nedeniyle aktif sınıflar şunlardır:

```text
no_event
continuation_bullish
continuation_bearish
reversal_bullish
reversal_bearish
```

`fresh_break_*` sınıfları test split'te support almaz; bu yüzden ortak 7 sınıflı raw macro F1, outcome task için biraz cezalandırıcı okunmalıdır.

## Setup

Input:

```text
z_fused
+ explicit proximity / level-interaction context
```

Target mode:

```text
target_mode = outcome
outcome_column = h16_post_break_outcome
dominant_direction_column = h16_dominant_forward_direction
```

Checkpoint:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_outcome_joint_v1_z_fused_proximity
```

## Command

```bash
PYTHONPATH=src python scripts/train_stage1b_joint_event_sequence.py \
  --train-latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_train_latents.npz \
  --train-label-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_train_h4_h8_h16_v2.parquet \
  --val-latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_val_latents.npz \
  --val-label-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_val_h4_h8_h16_v2.parquet \
  --test-latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_test_latents.npz \
  --test-label-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_test_h4_h8_h16_v2.parquet \
  --feature-keys z_fused \
  --context-feature-keys bull_distance_pct_scaled bear_distance_pct_scaled nearest_distance_pct_scaled nearest_side_signed bull_close_count_norm bear_close_count_norm bull_wick_count_norm bear_wick_count_norm bull_final_excess_scaled bear_final_excess_scaled current_label_intact current_label_bullish current_label_bearish \
  --target-mode outcome \
  --outcome-column h16_post_break_outcome \
  --dominant-direction-column h16_dominant_forward_direction \
  --event-type-column h16_event_type \
  --event-direction-column h16_event_direction \
  --balanced-loss \
  --epochs 50 \
  --patience 10 \
  --batch-size 256 \
  --checkpoint-dir data/stage1b/binance/15m/checkpoints/stage1b_h16_outcome_joint_v1_z_fused_proximity
```

## Metrics

Best validation checkpoint:

```text
epoch = 14
best_val_joint_macro_f1 = 0.2769
```

Test:

```text
joint_macro_f1             = 0.2840
derived event_type_f1      = 0.3375
derived event_direction_f1 = 0.6048
```

Joint per-class F1:

```text
no_event              0.6501
fresh_break_bullish   0.0000  # no support in outcome target
fresh_break_bearish   0.0000  # no support in outcome target
continuation_bullish  0.5761
continuation_bearish  0.5799
reversal_bullish      0.0742
reversal_bearish      0.1081
```

Test support:

```text
no_event              4611
fresh_break_bullish      0
fresh_break_bearish      0
continuation_bullish  2216
continuation_bearish  2641
reversal_bullish        96
reversal_bearish       123
```

## 449-454 Window Check

Prediction artifact:

```text
data/stage1b/binance/15m/checkpoints/stage1b_h16_outcome_joint_v1_z_fused_proximity/test_joint_event_predictions.csv
```

| index | true outcome joint | pred outcome joint | conf | comment |
|---:|---|---|---:|---|
| 449 | no_event | no_event | 0.392 | correct |
| 450 | no_event | no_event | 0.380 | correct |
| 451 | reversal_bearish | reversal_bearish | 0.357 | correct but low confidence |
| 452 | reversal_bearish | reversal_bearish | 0.437 | correct; fixes event-joint target mismatch |
| 453 | reversal_bearish | reversal_bearish | 0.509 | correct |
| 454 | continuation_bearish | reversal_bullish | 0.368 | wrong direction/type |

Bu segmentte outcome-joint target, 451-453 aralığını beklediğimiz gibi bearish reversal bağlamına taşıdı.

## Interpretation

Outcome-joint model genel metrikte zayıf:

- Reversal support çok düşük.
- Reversal F1 hâlâ düşük.
- `fresh_break_*` sınıfları outcome target'ta hiç support almadığı için ortak joint label space içinde sıfır F1 yazılıyor.

Fakat tartıştığımız trade-context probleminde daha doğru semantik veriyor:

```text
event-joint:
  452 true = fresh_break_bullish

outcome-joint:
  452 true = reversal_bearish
  452 pred = reversal_bearish
```

Bu, Stage 2'ye aktarılacak H16 bilgisinin "ilk kırılım yönü" yerine "post-break path outcome" olarak da temsil edilmesi gerektiğini gösteriyor.

## Decision

Outcome-joint v1, selected predictor değil; fakat contract açısından önemli bir tanı sonucu:

```text
Stage 1B output tek bir event sınıfına indirgenmemeli.
```

Daha doğru H16 contract iki ayrı sinyal taşımalı:

1. `first_event_signal`
   - horizon içindeki ilk structural event
   - timing / trigger bilgisi için yararlı

2. `outcome_signal`
   - post-break continuation / reversal yönü
   - Stage 2 trade bias ve target feasibility için daha yararlı

## Next Direction

Stage 1B H16 için tek model / tek target yerine çok-head veya çok-output tasarım denenmeli:

```text
shared trunk:
  z_fused + explicit context

heads:
  first_event_joint
  outcome_joint
  dominant_direction
  optional timing bucket
```

Bu tasarım, 452 gibi örneklerde hem `fresh_break_bullish` bilgisini hem de `reversal_bearish` outcome bilgisini aynı anda koruyabilir.

## Artifacts

```text
src/msb_repr/stage1b/dataset.py
scripts/train_stage1b_joint_event_sequence.py
data/stage1b/binance/15m/checkpoints/stage1b_h16_outcome_joint_v1_z_fused_proximity/
```

## Verification

```bash
python -m py_compile src/msb_repr/stage1b/dataset.py scripts/train_stage1b_joint_event_sequence.py
ruff check src/msb_repr/stage1b/dataset.py scripts/train_stage1b_joint_event_sequence.py
```
