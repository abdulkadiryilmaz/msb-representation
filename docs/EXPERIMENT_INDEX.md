# Experiment Index

Bu dosya, Stage 1A deneylerini yüksek seviyede takip etmek için kısa bir giriş noktasıdır.

## Nasıl Okunmalı

- araştırma framing / temel varsayımlar: `docs/foundations/`
- implementasyon ve karar notları: `docs/worklogs/`
- deney planı ve deney readout'ları: `docs/experiments/`
- ortak süreç: `docs/EXPERIMENT_WORKFLOW.md`

## Foundations

- `docs/foundations/market-structure-edge-program.md`
- `docs/foundations/stage1-msb-representation-framing.md`

`market-structure-edge-program.md`, repo üstü program hedefini, mission/thesis ayrımını ve nihai `TradePlan` çıktısını tanımlar.

`stage1-msb-representation-framing.md`, Stage 1A araştırma yönünü ve representation-learning çerçevesini tanımlar. Deney okumalarından önce bu iki foundations belgesiyle başlamak en doğru giriş olur.

## Current Active Experiment

- active run: yok
- latest completed:
  - `CE + SupCon | long aux v2`
  - readout: `docs/experiments/ce_supcon/2026-05-13-ce-supcon-long-aux-v2-readout.md`
- latest result:
  - `long aux v2` reddedildi
  - pressure probe ve F1 güçlü, ancak `z_long` / `z_long_proj` symbol shortcut guardrail'i bozuldu
- current best candidate:
  - `ce_supcon_long_v1`

## Current Summary

| Family | Variant | Status | Kısa Yorum |
|---|---|---|---|
| `CE-only` | core4 baseline | reference | güçlü ve stabil baseline |
| `CE + SupCon` | fused projection v1 | mixed | `z_long` iyileşti ama symbol bias büyüdü |
| `CE + SupCon` | long projection v1 | best current | en iyi latent geometry dengesi |
| `CE + SupCon` | long projection v2 | rejected | weight düşürmek fayda getirmedi |
| `CE + SupCon` | long symbol v1 | rejected | geometry yeniden symbol-heavy oldu |
| `CE + SupCon` | long symbol v2 | rejected | neutral same-symbol mask yönü düzeltmedi; geometry hâlâ symbol-heavy |
| `CE + SupCon` | long factor v1 | rejected | pressure probe iyileşti ama symbol shortcut geri döndü |
| `CE + SupCon` | long aux v1 | mixed | pressure ve borderline iyileşti; genel `z_long` geometry hafif geriledi |
| `CE + SupCon` | long aux v2 | rejected | pressure probe güçlü ama `z_long` yeniden symbol-heavy oldu |

## Metrics Snapshot

### Classifier — Val Macro F1

| Variant | Val macro F1 |
|---|---|
| ce_only | 0.8901 |
| ce_supcon_v1 | 0.8908 |
| ce_supcon_long_v1 | 0.8900 |
| ce_supcon_long_v2 | 0.8892 |
| ce_supcon_long_symbol_v1 | 0.8906 |
| ce_supcon_long_symbol_v2 | 0.8890 |
| ce_supcon_long_factor_v1 | 0.8874 |
| ce_supcon_long_aux_v1 | 0.8870 |
| ce_supcon_long_aux_v2 | 0.8898 |

Yorum: tüm koşularda F1 değerleri çok yakın. Bu metrik tek başına deney kararını belirleyemiyor.

### Latent Geometry — `z_long` NN Metrikleri

Karar metriği olarak `z_long` seçilmesinin nedeni: `z_short` tüm koşularda stabil kalıyor, asıl değişim ve risk buraya yansıyor.

**Val split:**

| Variant | label agree ↑ | symbol agree ↓ |
|---|---|---|
| ce_only | 0.4442 | 0.9093 |
| ce_supcon_v1 | 0.4746 | 0.9493 |
| **ce_supcon_long_v1** | **0.8096** | **0.3171** |
| ce_supcon_long_v2 | 0.7955 | 0.4543 |
| ce_supcon_long_symbol_v1 | 0.4741 | 0.9475 |
| ce_supcon_long_symbol_v2 | 0.4661 | 0.9233 |
| ce_supcon_long_factor_v1 | 0.8004 | 0.7466 |
| ce_supcon_long_aux_v1 | 0.7996 | 0.3221 |
| ce_supcon_long_aux_v2 | 0.7910 | 0.7839 |

**Test split:**

| Variant | label agree ↑ | symbol agree ↓ |
|---|---|---|
| ce_only | 0.4504 | 0.9115 |
| ce_supcon_v1 | 0.4838 | 0.9470 |
| **ce_supcon_long_v1** | **0.8080** | **0.2987** |
| ce_supcon_long_v2 | 0.7965 | 0.4543 |
| ce_supcon_long_symbol_v1 | 0.4840 | 0.9404 |
| ce_supcon_long_symbol_v2 | 0.4748 | 0.9150 |
| ce_supcon_long_factor_v1 | 0.7994 | 0.7250 |
| ce_supcon_long_aux_v1 | 0.7977 | 0.3141 |
| ce_supcon_long_aux_v2 | 0.7963 | 0.7807 |

### Latent Geometry — `z_fused` NN Metrikleri

**Val split:**

| Variant | label agree ↑ | symbol agree ↓ |
|---|---|---|
| ce_only | 0.8437 | 0.5739 |
| ce_supcon_v1 | 0.8431 | 0.6001 |
| **ce_supcon_long_v1** | **0.8424** | **0.2986** |
| ce_supcon_long_v2 | 0.8401 | 0.3530 |
| ce_supcon_long_symbol_v1 | 0.8451 | 0.6362 |
| ce_supcon_long_symbol_v2 | 0.8446 | 0.6111 |
| ce_supcon_long_factor_v1 | 0.8469 | 0.4987 |
| ce_supcon_long_aux_v1 | 0.8447 | 0.2831 |
| ce_supcon_long_aux_v2 | 0.8443 | 0.4197 |

**Test split:**

| Variant | label agree ↑ | symbol agree ↓ |
|---|---|---|
| ce_only | 0.8445 | 0.5970 |
| ce_supcon_v1 | 0.8397 | 0.6346 |
| **ce_supcon_long_v1** | **0.8425** | **0.2784** |
| ce_supcon_long_v2 | 0.8414 | 0.3613 |
| ce_supcon_long_symbol_v1 | 0.8348 | 0.6509 |
| ce_supcon_long_symbol_v2 | 0.8394 | 0.6595 |
| ce_supcon_long_factor_v1 | 0.8446 | 0.5026 |
| ce_supcon_long_aux_v1 | 0.8398 | 0.2816 |
| ce_supcon_long_aux_v2 | 0.8457 | 0.4094 |

### Pressure Probe — `intact` Altkümesi Linear F1 (test split)

Majority baseline: `0.3015`

| Variant | z_short | z_long | z_fused |
|---|---|---|---|
| ce_only | 0.5227 | 0.2067 | 0.5254 |
| **ce_supcon_long_v1** | **0.5257** | **0.4842** | **0.5407** |
| ce_supcon_long_symbol_v1 | 0.5411 | 0.1967 | 0.5348 |
| ce_supcon_long_symbol_v2 | 0.5476 | 0.2278 | 0.5408 |
| ce_supcon_long_factor_v1 | 0.5497 | 0.5547 | 0.5782 |
| ce_supcon_long_aux_v1 | 0.5639 | 0.5135 | 0.5686 |
| ce_supcon_long_aux_v2 | 0.5700 | 0.5103 | 0.5801 |

Not: pressure probe, eğitim hedefinin dışındaki `intact` içi yönsel baskı semantiğini ölçüyor. Label tanımı henüz validate edilmediğinden bu metrik teşhis aracı olarak okunmalı, doğrudan karar kriteri olarak değil.

## CE-Only

Klasör:

- `docs/experiments/ce_only/`

Önerilen okuma sırası:

1. `2026-04-13-ce-only-baseline-core4.md`
2. `2026-04-14-ce-only-embedding-view-readout.md`
3. `2026-04-14-ce-only-bucket-readout.md`
4. `2026-04-14-ce-only-test-split-readout.md`
5. `2026-04-14-ce-only-synthesis.md`

Kısa sonuç:

- güçlü ve stabil baseline
- `z_short` semantik olarak güçlü
- `z_long` symbol-heavy
- ana açık problem `borderline_intact_break`

## CE + SupCon

Klasör:

- `docs/experiments/ce_supcon/`

Önerilen okuma sırası:

1. `2026-04-14-ce-supcon-experiment-plan.md`
2. `2026-04-14-ce-supcon-v1-readout.md`
3. `2026-04-15-ce-supcon-long-v1-readout.md`
4. `2026-04-16-ce-supcon-long-v2-readout.md`
5. `2026-04-16-ce-supcon-long-symbol-v1-readout.md`
6. `2026-04-17-ce-supcon-long-symbol-v2-plan.md`
7. `2026-04-17-ce-supcon-long-symbol-v2-readout.md`
8. `2026-05-13-ce-supcon-long-factor-v1-plan.md`
9. `2026-05-13-ce-supcon-long-factor-v1-readout.md`
10. `2026-05-13-ce-supcon-long-aux-v1-plan.md`
11. `2026-05-13-ce-supcon-long-aux-v1-readout.md`

Kısa sonuç:

- fused-projection SupCon yön olarak karışık
- branch-aware long projection en iyi sonuç
- weight düşürme fayda getirmedi
- ilk symbol-aware deneme başarısız oldu
- `long symbol v2` reddedildi
- `long factor v1` reddedildi; pressure sinyali güçlendi ama symbol shortcut geri döndü
- `long aux v1` mixed; pressure ve borderline iyileşti ama `z_long` geometry hafif geriledi

## Worklogs

Bugün için `worklogs` altında kalan ana notlar:

- `2026-04-13-latent-export-and-analysis-infra.md`
- `2026-04-13-hard-case-domain-buckets.md`
- `2026-04-15-branch-aware-supcon-implementation.md`
- `2026-04-16-symbol-aware-supcon-implementation.md`

Bu notlar deney sonucu değil, tasarım / implementasyon / karar notlarıdır.
