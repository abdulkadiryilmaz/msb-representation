# Experiment Readout: 2026-05-13 — CE + SupCon Long Aux V1

**Status**: mixed / promising ablation  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_aux_v1`  
**Baseline to beat**: `ce_supcon_long_v1`  

---

## Amaç

`ce_supcon_long_factor_v1`, pressure probe'u iyileştirdi ama `z_long` / `z_fused` symbol shortcut guardrail'ini bozdu. Bu koşuda factor-aware SupCon pairing geri alındı; başarılı `ce_supcon_long_v1` pairing kuralı korundu ve sadece düşük ağırlıklı pressure/maturity auxiliary head eklendi.

Hedef:

- `ce_supcon_long_v1` düşük symbol agreement profilini korumak
- pressure / maturity okunabilirliğini artırmak
- borderline intact misclassification oranını düşürmek
- genel latent geometry'yi bozmamak

---

## Objective

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
+ 0.01 * CE(pressure_state)
+ 0.01 * CE(maturity_target)
```

Komut:

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 32 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_aux_v1 \
  --use-supcon \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode label \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --pressure-loss-weight 0.01 \
  --maturity-loss-weight 0.01
```

---

## Training Summary

| Metric | Value |
|---|---:|
| device | `mps` |
| completed epochs | 26 |
| best val macro F1 | 0.8870 |
| best epoch | 23 |
| min val loss | 0.4306 |
| min val loss epoch | 16 |
| final train SupCon anchor rate | 0.9999 |
| final train avg positive count | 11.5287 |
| final val SupCon anchor rate | 0.9989 |
| final val avg positive count | 12.3666 |

Val macro F1, `ce_supcon_long_v1` seviyesinin biraz altında kaldı: `0.8900 -> 0.8870`.

---

## Latent Geometry

Test split:

| Variant | Embedding | Label agree ↑ | Symbol agree ↓ |
|---|---|---:|---:|
| `ce_supcon_long_v1` | `z_long` | 0.8080 | 0.2987 |
| `ce_supcon_long_aux_v1` | `z_long` | 0.7977 | 0.3141 |
| `ce_supcon_long_factor_v1` | `z_long` | 0.7994 | 0.7250 |
| `ce_supcon_long_v1` | `z_fused` | 0.8425 | 0.2784 |
| `ce_supcon_long_aux_v1` | `z_fused` | 0.8398 | 0.2816 |
| `ce_supcon_long_factor_v1` | `z_fused` | 0.8446 | 0.5026 |
| `ce_supcon_long_v1` | `z_long_proj` | 0.8067 | 0.2621 |
| `ce_supcon_long_aux_v1` | `z_long_proj` | 0.7952 | 0.2634 |

Okuma:

- `long_factor_v1`teki symbol shortcut geri dönüşü burada yok.
- `z_fused` symbol agreement neredeyse `long_v1` seviyesinde kaldı.
- Ancak `z_long` label agreement `0.8080 -> 0.7977` düştü.
- `z_long_proj` label agreement da `0.8067 -> 0.7952` düştü.

Bu yüzden aux head'ler symbol guardrail'i korusa da ana `z_long` geometry üzerinde küçük ama karar açısından önemli bir maliyet yaratıyor.

---

## Pressure Probe

Test split, intact subset, linear macro F1:

| Variant | `z_short` | `z_long` | `z_fused` |
|---|---:|---:|---:|
| `ce_supcon_long_v1` | 0.5257 | 0.4842 | 0.5407 |
| `ce_supcon_long_aux_v1` | 0.5639 | 0.5135 | 0.5686 |
| `ce_supcon_long_factor_v1` | 0.5497 | 0.5547 | 0.5782 |

Pressure açısından aux v1 başarılı:

- `z_short`: `0.5257 -> 0.5639`
- `z_long`: `0.4842 -> 0.5135`
- `z_fused`: `0.5407 -> 0.5686`

Bu, low-weight pressure auxiliary head'in representation içinde okunabilir pressure sinyalini artırabildiğini gösteriyor.

---

## Maturity Probe

Test split, borderline intact vs confirmed break, macro F1:

| Variant | Embedding | Linear | MLP |
|---|---|---:|---:|
| `ce_supcon_long_v1` | `z_long` | 0.6995 | 0.7134 |
| `ce_supcon_long_aux_v1` | `z_long` | 0.6888 | 0.6586 |
| `ce_supcon_long_v1` | `z_fused` | 0.7158 | 0.7529 |
| `ce_supcon_long_aux_v1` | `z_fused` | 0.7286 | 0.7642 |

Okuma:

- `z_fused` maturity okunabilirliği hafif iyileşti.
- `z_long` maturity okunabilirliği geriledi.
- Auxiliary maturity sinyali daha çok fused head tarafında kullanışlı hale gelmiş görünüyor; `z_long` representation tarafında net kazanç üretmedi.

---

## Borderline Diagnosis

`z_fused`, test split, borderline intact:

| Variant | Misclassified | Rate | Avg NN confirmed | Avg NN same-symbol | Avg confidence |
|---|---:|---:|---:|---:|---:|
| `ce_supcon_long_v1` | 206 | 0.2818 | 0.3153 | 0.2814 | 0.8033 |
| `ce_supcon_long_aux_v1` | 176 | 0.2408 | 0.3109 | 0.2766 | 0.7946 |
| `ce_supcon_long_factor_v1` | 199 | 0.2722 | 0.3182 | 0.4970 | 0.7882 |

Bu koşunun en iyi sonucu burada:

- borderline misclassification `206 -> 176` düştü
- misclassification rate `0.2818 -> 0.2408`
- NN confirmed drift hafif azaldı
- same-symbol drift artmadı

Yani auxiliary heads, factor-aware SupCon kadar agresif symbol shortcut üretmeden borderline davranışını iyileştirdi.

---

## Karar

`ce_supcon_long_aux_v1` best current olarak alınmıyor.

Gerekçe:

- borderline davranışı ve pressure probe net iyileşti
- symbol shortcut guardrail korundu
- ancak `z_long` label agreement ve `z_long_proj` label agreement `ce_supcon_long_v1`e göre geriledi
- val macro F1 de küçük düşüş gösterdi

Bu koşu reddedilmiş bir yön değil; aksine doğru mekanizmayı işaret eden promising ablation. Ana sonuç şu:

> Coarse SupCon pairing korunmalı. Factor-aware SupCon pairing yerine düşük ağırlıklı auxiliary pressure/maturity sinyali daha kontrollü çalışıyor, fakat ağırlıklar `z_long` geometry'yi koruyacak şekilde yeniden ayarlanmalı.

---

## Next

Önerilen takip koşusu:

```text
ce_supcon_long_aux_v2
```

Adaylar:

1. `pressure_loss_weight=0.005`, `maturity_loss_weight=0.005`
2. pressure-only ablation: `pressure_loss_weight=0.01`, `maturity_loss_weight=0.0`

İlk tercih pressure-only ablation olmalı. Çünkü aux v1'de pressure probe ve borderline behavior iyileşirken maturity sinyali `z_long` tarafında net kazanç vermedi.
