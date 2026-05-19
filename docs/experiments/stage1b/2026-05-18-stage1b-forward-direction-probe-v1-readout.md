# Stage 1B Forward Direction Probe v1 Readout

**Date**: 2026-05-18
**Status**: initial downstream transfer probe completed
**Stage 1A source checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`
**Latent exports**:

- `analysis/selected_epoch_020_train_latents.npz`
- `analysis/selected_epoch_020_val_latents.npz`
- `analysis/selected_epoch_020_test_latents.npz`

## Goal

Stage 1A temsilinin Stage 1B'ye aktarılabilirliğini ilk kez forward yapı etiketiyle test etmek.

Soru:

> Stage 1A latent'i, anchor'dan sonraki yakın vadeli ilk yapısal kırılım yönü hakkında okunabilir sinyal taşıyor mu?

Bu çalışma TradePlan üretmez. Program hedefi açısından yalnızca TradePlan'ın yön / forward-structure bileşenine giden ilk transfer sanity check'tir.

## Label Definition

Analysis-only forward label artifact üretildi:

- `stage1b_forward_labels_train_h16_h32_h48.parquet`
- `stage1b_forward_labels_val_h16_h32_h48.parquet`
- `stage1b_forward_labels_test_h16_h32_h48.parquet`

Her anchor için:

1. Stage 1A short window içinden mevcut structure seviyeleri çıkarılır.
2. Stage 1A ile aynı pivot / effective break threshold mantığı kullanılır.
3. Anchor sonrasındaki future close serisinde `H = 16, 32, 48` bar içinde ilk close-confirmed break aranır.
4. Direction label atanır:

```text
none / bullish / bearish / ambiguous
```

`ambiguous` iki yönün aynı veya çok yakın confirmation penceresinde gelmesi için ayrıldı. Bu koşuda çok nadir olduğu için ana probe okumaları 3 sınıflı yapılmıştır:

```text
none / bullish / bearish
```

## Implementation

Yeni analysis-only scriptler:

- `scripts/generate_stage1b_forward_labels.py`
- `scripts/probe_stage1b_forward_labels.py`

Probe setup:

- train split ile probe eğitimi
- test split ile değerlendirme
- embeddings: `z_short`, `z_long`, `z_fused`
- models: balanced logistic regression, small sklearn MLP
- ana metrik: macro F1
- majority baseline raporlandı

## Label Distribution

Test split 3-class dağılımları:

| Horizon | bearish | bullish | none |
|---|---:|---:|---:|
| H16 | 3547 | 3115 | 3025 |
| H32 | 4276 | 3689 | 1706 |
| H48 | 4633 | 3940 | 1082 |

Horizon uzadıkça `none` azalıyor; daha çok örnek horizon içinde bir yöne break ediyor. Bu beklenen bir davranış, fakat daha uzun horizon'da forward label'ın piyasa rejimi gürültüsüne daha açık olabileceği not edilmeli.

## Main Results

3-class `z_fused` horizon comparison:

| Target | Majority macro F1 | Linear macro F1 | MLP macro F1 |
|---|---:|---:|---:|
| `h16_future_break_direction` | 0.1622 | 0.6768 | 0.6712 |
| `h32_future_break_direction` | 0.1841 | 0.6183 | 0.6020 |
| `h48_future_break_direction` | 0.1932 | 0.5807 | 0.5328 |

H32 embedding comparison:

| Embedding | Majority macro F1 | Linear macro F1 | MLP macro F1 |
|---|---:|---:|---:|
| `probs` | 0.1841 | 0.5759 | 0.4719 |
| `logits` | 0.1841 | 0.5943 | 0.4801 |
| `z_short` | 0.1841 | 0.6138 | 0.5921 |
| `z_long` | 0.1841 | 0.5922 | 0.5453 |
| `z_fused` | 0.1841 | 0.6183 | 0.6020 |

Horizon confusion matrices for 3-class `z_fused` linear probe:

H16 labels: `none / bullish / bearish`

```text
none    -> [1975,  530,  520]
bullish -> [ 722, 2143,  250]
bearish -> [ 819,  292, 2436]
```

H32 labels: `none / bullish / bearish`

```text
none    -> [1072,  349,  285]
bullish -> [ 883, 2377,  429]
bearish -> [1040,  489, 2747]
```

H48 labels: `none / bullish / bearish`

```text
none    -> [ 661,  262,  159]
bullish -> [ 932, 2488,  520]
bearish -> [1083,  625, 2925]
```

## Interpretation

Stage 1A temsilinde forward direction sinyali belirgin biçimde okunabiliyor.

Özellikle H16 sonucu güçlü:

- majority baseline çok düşük
- linear probe `0.6768` macro F1 veriyor
- MLP linear probe'u belirgin geçmiyor

Bu, sinyalin latent uzayda lineer okunabilir olduğunu düşündürür. MLP'nin düşük kalması şu aşamada olumsuz sinyal değildir; sklearn MLP probe'u class weighting kullanmıyor ve noisy forward label'da logistic regression daha düzenli genelleyebilir.

Horizon etkisi önemli:

- H16 en güçlü
- H32 hala anlamlı
- H48 zayıflıyor

Bu, seçili Stage 1A temsilinin yakın vadeli structure flow için daha uygun olduğunu gösteriyor olabilir. 15m timeframe'de H16 yaklaşık 4 saat, H32 yaklaşık 8 saat, H48 yaklaşık 12 saattir.

H32 embedding karşılaştırmasında `z_fused` en iyi, `z_short` çok yakın, `z_long` ise daha zayıf ama majority baseline'ın açık biçimde üstünde. Bu, forward direction için lokal yapı bilgisinin güçlü katkı verdiğini; long branch'in ise tek başına yeterli ama baskın olmayan bir sinyal taşıdığını gösterir.

`probs` ve `logits` baseline'ları da güçlüdür, fakat `z_short` / `z_fused` bunların üstüne çıkıyor. Bu, forward direction sinyalinin yalnızca Stage 1A sınıf olasılıklarından ibaret olmadığını; latent geometride ek bilgi bulunduğunu gösterir.

Confusion matrix tarafında ana hata deseni `none` ile directional break sınıflarının karışmasıdır. Direction geldiğinde bullish / bearish ayrımı tamamen collapse etmiyor; fakat horizon uzadıkça `none` sınıfı küçülüyor ve break-vs-no-break ayrımı daha zor okunuyor.

## Caveats

- Bu henüz predictor değil, probe seviyesinde transfer testi.
- Label tanımı ilk versiyondur; future break direction doğası gereği gürültülü olabilir.
- `ambiguous` sınıfı çok nadir olduğu için ana metrik dışında bırakıldı.
- Current label, entry / stop / target / quality üretmez.
- Bu sonuç Stage 2 tradability veya PnL edge kanıtı değildir.

## Decision

Stage 1A seçili checkpoint, Stage 1B yön tahmini için kullanılabilir downstream sinyal taşıyor.

Bir sonraki adım:

1. `future_break_direction_H` label v1'i dokümante etmek.
2. H16 / H32 odağında gerçek Stage 1B predictor tasarımına geçmeden önce:
   - label definition guardrail'lerini netleştirmek,
   - val split üzerinde predictor-selection protokolünü belirlemek,
   - basit Stage 1B predictor tasarımını planlamak.

## Artifacts

Forward labels:

```text
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_train_h16_h32_h48.parquet
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_val_h16_h32_h48.parquet
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_test_h16_h32_h48.parquet
```

Probe outputs:

```text
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/stage1b_h16_direction_3class_test_z_fused/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/stage1b_h32_direction_3class_test_z_short/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/stage1b_h32_direction_3class_test_z_long/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/stage1b_h32_direction_3class_test_z_fused/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/stage1b_h48_direction_3class_test_z_fused/
```
