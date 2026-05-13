# Experiment Readout: 2026-05-13 — CE + SupCon Long Aux V2

**Status**: rejected  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_aux_v2`  
**Baseline to beat**: `ce_supcon_long_v1`  

---

## Amaç

`aux_v1` pressure ve borderline davranışını iyileştirdi, fakat maturity auxiliary head `z_long` geometry üzerinde maliyet yaratmış olabilir. Bu koşuda maturity auxiliary kapatıldı ve sadece pressure auxiliary bırakıldı.

Objective:

```text
CE(confirmed_state)
+ 0.05 * SupCon(z_long_proj, confirmed_state)
+ 0.01 * CE(pressure_state)
```

---

## Training Summary

| Metric | Value |
|---|---:|
| device | `mps` |
| completed epochs | 31 |
| early stopping epoch | 31 |
| best val macro F1 | 0.8898 |
| best epoch | 18 |
| min val loss | 0.4249 |
| min val loss epoch | 21 |
| final train SupCon anchor rate | 1.0000 |
| final train avg positive count | 11.5168 |
| final val SupCon anchor rate | 0.9989 |
| final val avg positive count | 12.3666 |

Classifier tarafında koşu güçlü görünüyor: `0.8898`, yani `ce_supcon_long_v1` seviyesine çok yakın.

---

## Latent Geometry

Test split:

| Variant | Embedding | Label agree ↑ | Symbol agree ↓ |
|---|---|---:|---:|
| `ce_supcon_long_v1` | `z_long` | 0.8080 | 0.2987 |
| `ce_supcon_long_aux_v1` | `z_long` | 0.7977 | 0.3141 |
| `ce_supcon_long_aux_v2` | `z_long` | 0.7963 | 0.7807 |
| `ce_supcon_long_v1` | `z_fused` | 0.8425 | 0.2784 |
| `ce_supcon_long_aux_v1` | `z_fused` | 0.8398 | 0.2816 |
| `ce_supcon_long_aux_v2` | `z_fused` | 0.8457 | 0.4094 |
| `ce_supcon_long_v1` | `z_long_proj` | 0.8067 | 0.2621 |
| `ce_supcon_long_aux_v2` | `z_long_proj` | 0.7884 | 0.6374 |

Bu koşunun ana problemi burada:

- `z_long` symbol agreement `0.2987 -> 0.7807`
- `z_long_proj` symbol agreement `0.2621 -> 0.6374`
- `z_fused` symbol agreement `0.2784 -> 0.4094`

Pressure-only auxiliary, classifier F1'i korusa da latent geometry'yi coin identity tarafına yeniden çekti.

---

## Pressure Probe

Test split, intact subset, linear macro F1:

| Variant | `z_short` | `z_long` | `z_fused` |
|---|---:|---:|---:|
| `ce_supcon_long_v1` | 0.5257 | 0.4842 | 0.5407 |
| `ce_supcon_long_aux_v1` | 0.5639 | 0.5135 | 0.5686 |
| `ce_supcon_long_aux_v2` | 0.5700 | 0.5103 | 0.5801 |

Pressure probe iyileşti. Özellikle `z_fused` `0.5801` ile bu zincirdeki en yüksek pressure okunabilirliğini verdi.

Ancak bu kazanım symbol shortcut pahasına geldiği için kabul edilebilir değil.

---

## Maturity Probe

Test split, borderline intact vs confirmed break, macro F1:

| Variant | Embedding | Linear | MLP |
|---|---|---:|---:|
| `ce_supcon_long_v1` | `z_long` | 0.6995 | 0.7134 |
| `ce_supcon_long_aux_v1` | `z_long` | 0.6888 | 0.6586 |
| `ce_supcon_long_aux_v2` | `z_long` | 0.6653 | 0.6390 |
| `ce_supcon_long_v1` | `z_fused` | 0.7158 | 0.7529 |
| `ce_supcon_long_aux_v1` | `z_fused` | 0.7286 | 0.7642 |
| `ce_supcon_long_aux_v2` | `z_fused` | 0.7195 | 0.7494 |

Maturity head kapatılınca `z_long` maturity okunabilirliği geri kazanılmadı; aksine daha da düştü.

---

## Borderline Diagnosis

`z_fused`, test split, borderline intact:

| Variant | Misclassified | Rate | Avg NN confirmed | Avg NN same-symbol | Avg confidence |
|---|---:|---:|---:|---:|---:|
| `ce_supcon_long_v1` | 206 | 0.2818 | 0.3153 | 0.2814 | 0.8033 |
| `ce_supcon_long_aux_v1` | 176 | 0.2408 | 0.3109 | 0.2766 | 0.7946 |
| `ce_supcon_long_aux_v2` | 188 | 0.2572 | 0.3138 | 0.4015 | 0.7958 |

Borderline misclassification `long_v1`e göre iyileşti, fakat `aux_v1`den kötüleşti. Daha önemlisi same-symbol NN rate `0.4015`e çıktı.

---

## Karar

`ce_supcon_long_aux_v2` reddedildi.

Gerekçe:

- pressure probe güçlü
- val macro F1 güçlü
- borderline misclassification `long_v1`e göre iyi
- fakat `z_long`, `z_long_proj` ve `z_fused` symbol shortcut guardrail'i bozuldu

Bu sonuç, `aux_v1`deki maturity auxiliary head'in tek başına problem olmadığını gösteriyor. Pressure-only auxiliary de representation'ı symbol-heavy hale getirebiliyor. Sorun auxiliary sinyalin varlığı değil, bu sinyalin `z_long` / projection geometry üzerinde nasıl taşındığı.

---

## Next

`ce_supcon_long_v1` current best kalmalı.

Auxiliary direction tamamen kapanmıyor, fakat sonraki deneme daha düşük ağırlıkla yapılmalı:

```text
pressure_loss_weight=0.005
maturity_loss_weight=0.005
```

Alternatif olarak pressure auxiliary yalnızca fused head'ten geriye daha sınırlı gradient taşıyacak şekilde ayrıştırılabilir; bu mimari değişiklik gerektirir ve ayrı worklog ister.
