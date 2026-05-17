# Experiment Readout: 2026-05-13 — CE + SupCon Long Short-Pressure Aux V1

**Status**: rejected  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_short_pressure_aux_v1`  
**Baseline to beat**: `ce_supcon_long_v1`

---

## Amaç

`aux_v2` pressure-only auxiliary ile pressure probe'u güçlendirdi, fakat pressure gradient `z_fused` üzerinden `z_long`a aktığı için `z_long` / `z_long_proj` symbol-heavy hale geldi. Bu koşuda pressure auxiliary head `z_fused` yerine `z_short` üstüne taşındı.

Objective:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
+ 0.01 * CE(pressure_state from z_short)
```

Maturity auxiliary kapalı.

---

## Training Summary

| Metric | Value |
|---|---:|
| device | `mps` |
| completed epochs | 29 |
| early stopping epoch | 29 |
| best val macro F1 | 0.8870 |
| best epoch | 19 |
| min val loss | 0.4501 |
| min val loss epoch | 19 |
| final train SupCon anchor rate | 1.0000 |
| final train avg positive count | 11.5267 |
| final val SupCon anchor rate | 0.9989 |
| final val avg positive count | 12.3666 |

Classifier F1 `aux_v1` seviyesinde kaldı ve `long_v1`in altına indi: `0.8900 -> 0.8870`.

---

## Latent Geometry

Test split:

| Variant | Embedding | Label agree ↑ | Symbol agree ↓ |
|---|---|---:|---:|
| `ce_supcon_long_v1` | `z_long` | 0.8080 | 0.2987 |
| `ce_supcon_long_aux_v2` | `z_long` | 0.7963 | 0.7807 |
| `ce_supcon_long_short_pressure_aux_v1` | `z_long` | 0.4742 | 0.9321 |
| `ce_supcon_long_v1` | `z_fused` | 0.8425 | 0.2784 |
| `ce_supcon_long_aux_v2` | `z_fused` | 0.8457 | 0.4094 |
| `ce_supcon_long_short_pressure_aux_v1` | `z_fused` | 0.8424 | 0.5701 |
| `ce_supcon_long_v1` | `z_long_proj` | 0.8067 | 0.2621 |
| `ce_supcon_long_aux_v2` | `z_long_proj` | 0.7884 | 0.6374 |
| `ce_supcon_long_short_pressure_aux_v1` | `z_long_proj` | 0.4461 | 0.9194 |

Bu koşu `z_long` geometry açısından açık başarısızlık:

- `z_long` label agreement `0.8080 -> 0.4742`
- `z_long` symbol agreement `0.2987 -> 0.9321`
- `z_long_proj` label agreement `0.8067 -> 0.4461`
- `z_long_proj` symbol agreement `0.2621 -> 0.9194`

Pressure gradient `z_long`a doğrudan gitmese bile koşu, branch-aware SupCon geometry'yi koruyamadı.

---

## Pressure Probe

Test split, intact subset, linear macro F1:

| Variant | `z_short` | `z_long` | `z_fused` |
|---|---:|---:|---:|
| `ce_supcon_long_v1` | 0.5257 | 0.4842 | 0.5407 |
| `ce_supcon_long_aux_v2` | 0.5700 | 0.5103 | 0.5801 |
| `ce_supcon_long_short_pressure_aux_v1` | 0.5774 | 0.2621 | 0.5702 |

Pressure objective `z_short` üzerinde beklenen işi yaptı:

- `z_short` pressure F1 `0.5257 -> 0.5774`
- `z_fused` pressure F1 de `0.5702` ile yüksek kaldı

Ancak `z_long` pressure probe çöktü. Bu, pressure'ın gerçekten short branch'e izole edildiğini gösteriyor; fakat aynı zamanda `z_long`un SupCon ile iyi structural geometry kuramadığını ortaya koyuyor.

---

## Maturity Probe

Test split, borderline intact vs confirmed break, macro F1:

| Variant | Embedding | Linear | MLP |
|---|---|---:|---:|
| `ce_supcon_long_v1` | `z_long` | 0.6995 | 0.7134 |
| `ce_supcon_long_short_pressure_aux_v1` | `z_long` | 0.4006 | 0.4652 |
| `ce_supcon_long_v1` | `z_fused` | 0.7158 | 0.7529 |
| `ce_supcon_long_short_pressure_aux_v1` | `z_fused` | 0.7222 | 0.7525 |

`z_fused` maturity korunuyor, fakat `z_long` maturity ciddi düşüyor. Bu da failure'ın özellikle long branch / long projection geometry tarafında olduğunu destekliyor.

---

## Borderline Diagnosis

`z_fused`, test split, borderline intact:

| Variant | Misclassified | Rate | Avg NN confirmed | Avg NN same-symbol | Avg confidence |
|---|---:|---:|---:|---:|---:|
| `ce_supcon_long_v1` | 206 | 0.2818 | 0.3153 | 0.2814 | 0.8033 |
| `ce_supcon_long_aux_v1` | 176 | 0.2408 | 0.3109 | 0.2766 | 0.7946 |
| `ce_supcon_long_aux_v2` | 188 | 0.2572 | 0.3138 | 0.4015 | 0.7958 |
| `ce_supcon_long_short_pressure_aux_v1` | 216 | 0.2955 | 0.3233 | 0.5617 | 0.8022 |

Borderline behavior da kötüleşti:

- misclassified `206 -> 216`
- same-symbol NN rate `0.2814 -> 0.5617`

---

## Karar

`ce_supcon_long_short_pressure_aux_v1` reddedildi.

Önemli çıkarım:

> Pressure gradient'i `z_short`a izole etmek tek başına `z_long` geometry'yi korumaya yetmedi. Hatta bu koşuda `z_long` / `z_long_proj` branch-aware SupCon kazanımını tamamen kaybetti.

Bu, önceki hipotezi daraltıyor. Sorun sadece pressure CE gradient'inin `z_long`a doğrudan akması değil. Muhtemel mekanizma:

- pressure-supervised `z_short` classifier CE için daha baskın hale geliyor
- fused CE, long branch'i semantik olarak daha az zorlayabiliyor
- `z_long` üzerinde SupCon tek başına yeterli corrective force üretmiyor
- sonuçta `z_long` yeniden symbol/time-regime shortcut'a düşüyor

---

## Next

`ce_supcon_long_v1` current best kalmalı.

Auxiliary supervision için sıradaki düz loss/head denemeleri düşük bilgi değerinde görünüyor. Daha anlamlı sonraki yön, branch rolünü açıkça koruyan bir yapı olmalı:

- `z_long` için CE/SupCon dışı ek regularization veya stronger branch objective
- pressure auxiliary'nin fused classifier üzerinde short branch'i baskınlaştırmasını engelleyen gating/adaptor
- ya da pressure/maturity sinyallerini eğitim objective yerine diagnostic/readout olarak tutmak

Kısa vadeli karar: yeni auxiliary loss denemelerine ara verip `ce_supcon_long_v1`in neden branch-aware SupCon geometry'yi koruduğunu daha yakından analiz etmek daha doğru.
