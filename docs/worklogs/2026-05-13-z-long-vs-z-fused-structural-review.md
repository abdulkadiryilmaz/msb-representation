# Worklog: 2026-05-13 — `z_long` vs `z_fused` Structural Review

**Status**: Complete  
**Scope**: Stage 1A structural-state analysis, `ce_supcon_long_v1`

---

## Why This Review Exists

Önceki manuel NN okuması ağırlıklı olarak `z_long` üstünde yapıldı.

Bu faydalıydı, çünkü `ce_supcon_long_v1` koşusunda SupCon doğrudan `z_long_proj` / long branch hattını düzenliyordu. Ancak final classifier `z_long` üstünde değil, `z_fused` üstünde çalışıyor.

Bu yüzden yeni deney kararı için şu ayrım gerekli:

- sorun long branch'teki pressure/context temsilinde mi?
- fusion sonrası final representation'da mı?
- yoksa latent iyi olduğu halde classifier boundary mi problemli?

Bu worklog, aynı structural-state contract ile `z_long` ve `z_fused` çıktılarının yan yana okunmasını kaydeder.

---

## Artifacts

Structural analysis:

- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/test_latents_z_long_structural/`
- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/test_latents_z_fused_structural/`

Manual NN review:

- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/manual_nn_review/z_long_test_borderline_structural/`
- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/manual_nn_review/z_fused_test_borderline_structural/`

Filtre:

- split: `test`
- buckets:
  - `borderline_intact_break_up`
  - `borderline_intact_break_down`
  - `borderline_intact_break_mixed`
- top-k neighbors: `5`
- query count: `12`

---

## Overall NN Geometry

Test split overall:

| Embedding | mean NN label agreement | classifier accuracy |
|---|---:|---:|
| `z_long` | `0.8080` | `0.8851` |
| `z_fused` | `0.8425` | `0.8851` |

Classifier accuracy aynı, çünkü iki okuma aynı model tahminlerini kullanıyor. Değişen şey yalnızca NN geometry.

İlk sonuç:

- `z_fused`, overall label-neighbor agreement tarafında daha güçlü
- `z_long`, branch-aware SupCon etkisiyle güçlü bir long/context representation üretse de final fused uzay label komşuluğunu biraz daha iyi tutuyor

---

## Borderline Intact Error Surface

Structural full distribution'da borderline intact popülasyonu aynıdır; embedding seçimine bağlı değildir:

| Subset | Count | Misclassified | Rate |
|---|---:|---:|---:|
| borderline intact total | `731` | `206` | `0.2818` |
| down pressure | `367` | `116` | `0.3161` |
| up pressure | `358` | `88` | `0.2458` |
| neutral | `6` | `2` | `0.3333` |

Holding status kırılımı:

| Holding status | Count | Misclassified | Rate |
|---|---:|---:|---:|
| reverted | `446` | `138` | `0.3094` |
| holding | `190` | `47` | `0.2474` |
| mixed | `95` | `21` | `0.2211` |

Bu tablo şunu gösteriyor:

- hata oranı embedding view'dan bağımsız, çünkü classifier head aynı
- asıl karşılaştırma hata oranında değil, bu hatalı / sınır örneklerin NN uzayında hangi structural ailelere yakın düştüğünde yapılmalı

---

## Manual NN Family Comparison

### `z_long`

Query dağılımı:

- `7` x `borderline_intact_break_up / up_pressure / borderline / up / reverted`
- `3` x `borderline_intact_break_down / down_pressure / borderline / down / reverted`
- `2` x `borderline_intact_break_down / down_pressure / borderline / down / holding`

Neighbor family top counts:

| Neighbor family | Count |
|---|---:|
| `close_confirmed_break_up / bullish / confirmed / up / holding` | `22` |
| `close_confirmed_break_down / bearish / confirmed / down / holding` | `15` |
| `close_confirmed_break_down / bearish / confirmed / down / reverted` | `9` |
| `close_confirmed_break_up / bullish / confirmed / up / reverted` | `8` |
| intact wick/pressure families | `5` total |

Other:

- same-symbol neighbors: `21 / 60 = 0.3500`
- structural direction match: `60 / 60 = 1.0000`

Interpretation:

- `z_long` borderline intact örnekleri güçlü biçimde aynı yönlü confirmed break ailesine çekiyor
- bu, pressure direction açısından çok temiz
- fakat pressure vs confirmed state sınırı zayıf görünüyor

### `z_fused`

Query dağılımı:

- `5` x `borderline_intact_break_down / down_pressure / borderline / down / reverted`
- `3` x `borderline_intact_break_up / up_pressure / borderline / up / reverted`
- `2` x `borderline_intact_break_down / mixed_pressure / borderline / mixed / mixed`
- `1` x `borderline_intact_break_down / down_pressure / borderline / down / holding`
- `1` x `borderline_intact_break_up / up_pressure / borderline / up / holding`

Neighbor family top counts:

| Neighbor family | Count |
|---|---:|
| `close_confirmed_break_down / bearish / confirmed / down / reverted` | `20` |
| `close_confirmed_break_up / bullish / confirmed / up / reverted` | `12` |
| `close_confirmed_break_up / bullish / confirmed / up / holding` | `11` |
| `close_confirmed_break_down / bearish / confirmed / down / holding` | `9` |
| `clean_intact / intact / neutral / clean / none` | `4` |
| intact wick/pressure families | `4` total |

Other:

- same-symbol neighbors: `19 / 60 = 0.3167`
- structural direction match: `44 / 50 = 0.8800`
  - mixed-direction queries excluded from denominator

Interpretation:

- `z_fused` still keeps many borderline examples near confirmed break families
- however it introduces more intact-family neighbors than `z_long`
- direction alignment is slightly weaker than `z_long`
- same-symbol rate is not high, so this specific readout does not primarily indicate symbol shortcut

---

## What This Means

This review separates two facts:

1. `z_long` is very direction-aware.
   - Borderline up/down examples are almost perfectly near the same directional family.
   - But it tends to place borderline pressure near confirmed break states.

2. `z_fused` is more label-aware overall.
   - Overall NN label agreement is higher.
   - It pulls in some intact-family neighbors.
   - But it still does not fully separate pressure intact from confirmed break.

This supports the current diagnosis:

```text
ce_supcon_long_v1 learns useful directional pressure,
but confirmed_state vs near-confirmation pressure remains a boundary problem.
```

The issue does not look primarily like symbol shortcut in this slice:

- `z_long` same-symbol neighbor rate: `0.3500`
- `z_fused` same-symbol neighbor rate: `0.3167`

---

## Experiment Implication

The next experiment should not try to erase directionality from `z_long`.

Bad target:

```text
make borderline intact examples less directional
```

Better target:

```text
preserve pressure direction,
improve confirmed_state boundary,
avoid increasing symbol shortcut.
```

This points away from another broad symbol-aware SupCon attempt as the immediate next step.

More relevant directions:

1. `confirmed_state` boundary refinement
   - borderline-aware CE weighting
   - calibration / decision-threshold analysis

2. Factor-aware auxiliary objective
   - main head: `confirmed_state`
   - auxiliary head/probe: `pressure_state`
   - optional analysis target: `break_maturity`

3. Fusion/head diagnosis
   - if `z_fused` contains enough intact-family structure but classifier still predicts confirmed break, head-level intervention may be enough
   - if `z_fused` itself embeds borderline examples inside confirmed break clusters, representation objective needs refinement

---

## Decision

Before opening a new training experiment, keep this as the immediate decision frame:

- `z_long` is useful as a pressure/context representation
- `z_fused` remains the representation closest to classifier behavior
- the next readout should compare candidate interventions against both:
  - `z_long` pressure preservation
  - `z_fused` confirmed-state boundary quality

The success criterion should be:

```text
borderline pressure intact examples produce fewer early confirmed-state predictions,
while pressure information remains linearly/proximally readable,
and symbol agreement does not increase.
```

