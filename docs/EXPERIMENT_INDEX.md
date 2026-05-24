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
- `docs/foundations/stage1b-forward-labels.md`
- `docs/foundations/stage1b-output-contract.md`
- `docs/foundations/stage2-actionability-framing.md`
- `docs/foundations/stage2-actionability-labels.md`

`market-structure-edge-program.md`, repo üstü program hedefini, mission/thesis ayrımını ve nihai `TradePlan` çıktısını tanımlar.

`stage1-msb-representation-framing.md`, Stage 1A araştırma yönünü ve representation-learning çerçevesini tanımlar. Deney okumalarından önce bu iki foundations belgesiyle başlamak en doğru giriş olur.

`stage1b-forward-labels.md`, Stage 1B için ilk `future_break_direction_H` label sözleşmesini ve predictor öncesi guardrail'leri tanımlar.

`stage1b-output-contract.md`, seçili H8 predictor'ın Stage 2'ye hangi sinyal ve fiyat bağlamıyla aktarılacağını tanımlar.

`stage2-actionability-framing.md`, Stage 1B sinyalinden `no_trade / wait / actionable` kararına geçiş için ilk Stage 2 çerçevesini tanımlar.

`stage2-actionability-labels.md`, Stage 2 v1 için rule-based actionability label sözleşmesini tanımlar.

## Current Active Experiment

- active run:
  - Stage 1B event-sequence predictor design
  - selected Stage 1A checkpoint: `ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`
  - selected Stage 1B H8 checkpoint: `stage1b_h8_fresh_break_v3_z_fused_proximity/best.pt`
  - selected Stage 1B H16 checkpoint: `stage1b_h16_dominant_v3_z_fused_proximity/best.pt`
  - foundation: `docs/foundations/stage1b-output-contract.md`
- latest completed:
  - `Stage 1B event-sequence contract v2`
  - readout: `docs/experiments/stage1b/2026-05-24-stage1b-event-sequence-contract-v2-readout.md`
- latest result:
  - Stage 1B output contract `fresh break only` semantiğinden `event_type + event_direction` sözleşmesine taşındı
  - label generator artık `h{H}_event_type` ve `h{H}_event_direction` alanlarını üretiyor
  - index 453 artık `fresh_break=none` kalırken `event_type=reversal`, `event_direction=bearish` olarak ayrışıyor
  - index 454 `event_type=fresh_break`, `event_direction=bearish`
  - H8 reversal sınıfı seyrek: test `25`; H16 reversal daha kullanılabilir: test `112`
  - sonuç: H8 fresh-trigger için, H16 ise event-sequence / reversal-continuation çalışması için daha uygun aday
- current best candidate:
  - `ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`

## Current Summary

| Family | Variant | Status | Kısa Yorum |
|---|---|---|---|
| `CE-only` | core4 baseline | reference | güçlü ve stabil baseline |
| `CE + SupCon` | fused projection v1 | mixed | `z_long` iyileşti ama symbol bias büyüdü |
| `CE + SupCon` | long projection v1 | previous best | güçlü referans ama seed-sensitive |
| `CE + SupCon` | long projection v2 | rejected | weight düşürmek fayda getirmedi |
| `CE + SupCon` | long symbol v1 | rejected | geometry yeniden symbol-heavy oldu |
| `CE + SupCon` | long symbol v2 | rejected | neutral same-symbol mask yönü düzeltmedi; geometry hâlâ symbol-heavy |
| `CE + SupCon` | long factor v1 | rejected | pressure probe iyileşti ama symbol shortcut geri döndü |
| `CE + SupCon` | long aux v1 | mixed | pressure ve borderline iyileşti; genel `z_long` geometry hafif geriledi |
| `CE + SupCon` | long aux v2 | rejected | pressure probe güçlü ama `z_long` yeniden symbol-heavy oldu |
| `CE + SupCon` | long short-pressure aux v1 | rejected | pressure `z_short`ta iyileşti ama `z_long` geometry çöktü |
| `CE + SupCon` | long v1 seed stability | seed-sensitive | `seed_42` stabil, `seed_41` ve `seed_43` symbol-heavy / branch-usage collapse |
| `CE + SupCon` | long frontload v1 | useful / not fully seed-stable | `seed_42` güçlü, `seed_43` toparlandı, `seed_41` hâlâ symbol-heavy |
| `CE + SupCon` | long SupCon-only warmup v1 | diagnostic | `z_long` düzeldi ama `z_short` / `z_fused` branch dengesi bozuldu |
| `CE + SupCon` | long SupCon-dominant warmup v1 | best balanced schedule candidate | seed 41 ve seed 42'de `z_long` temizlendi, branch dengesi korundu |
| `CE + SupCon` | long branch CE aux v1 | selected for downstream transfer | `seed_41/epoch_020.pt` Stage 1A aktarım adayı seçildi; seed 42 bucket asimetrisi sonraki model iyileştirme konusu |
| `Stage 1B` | forward direction probe v1 | transfer signal found | H16/H32/H48 future break direction label'ları üretildi; `z_fused` linear probe H16 `0.6768`, H32 `0.6183`, H48 `0.5807` |
| `Stage 1B` | H16 predictor v1a | accepted baseline | two-head predictor; test reconstructed macro F1 `0.6731`, break F1 `0.7015`, direction F1 `0.8330` |
| `Stage 1B` | H16 + H32 predictor v1b | accepted multi-horizon baseline | H16 test reconstructed macro F1 `0.6765`; H32 direction signal useful, H32 break occurrence weak |
| `Stage 1B` | H16=None / H32=Break segment | diagnostic | delayed segment test support `%13.6`; precision `0.2451`, recall `0.4761`, H32 direction F1 `0.5416` |
| `Stage 1B` | forward label audit v1 | completed | H16/H32 consistency clean; labels are break-heavy, H32 test break rate `0.8210` |
| `Stage 1B` | short-horizon label + probe v1 | completed | H4/H8 more readable than H16; H8 selected as next predictor target |
| `Stage 1B` | H8 predictor v1a | accepted primary baseline | test reconstructed macro F1 `0.7307`, break F1 `0.7497`, direction F1 `0.8955` |
| `Stage 1B` | H8 + H16 predictor v1b | rejected | H8 test reconstructed macro F1 `0.7270`; H16 equal-weight auxiliary does not improve H8 |
| `Stage 1B` | H8 predictor error audit v1 | completed | high-confidence predictions reliable; 0.90-1.00 break confidence bucket accuracy `0.9532` |
| `Stage 1B` | H8 visual / proximity diagnostic v1 | completed | high-confidence signals are level-proximity driven; Stage 2 needs price/level context |
| `Stage 1B` | synthesis / close-out | completed | H8 predictor v1a selected for Stage 2 handoff |
| `Stage 1B` | H8 fresh break semantics audit v1 | completed | H8 directional label'ların `%66.00` kadarı already-broken; v2 label anchor-side ayrımı gerektiriyor |
| `Stage 1B` | H8 fresh break v2 | diagnostic | fresh-trigger target materially harder; balanced predictor test recon macro F1 `0.4472` at threshold `0.57` |
| `Stage 1B` | event sequence v2 | completed | `post_break_outcome` eklendi; H16 dominant direction test macro F1 `0.6181` |
| `Stage 1B` | Stage 1A latent proximity probe v1 | diagnostic | intact örneklerde nearest side okunuyor (`z_fused` linear F1 `0.7644`), distance bucket zayıf (`0.4295`) |
| `Stage 1B` | proximity-enriched predictor v3 | accepted diagnostic | `z_fused + explicit context`; H8 fresh recon F1 `0.5971`, direction F1 `0.8412`; index 454 hatası düzeldi |
| `Stage 1B` | event-sequence contract v2 | completed | output contract `event_type + event_direction`; H8 reversal seyrek, H16 event-sequence predictor sonraki aday |
| `Stage 2` | actionability label v1 | completed | rule-based labels generated; false-positive Stage 1B breaks mostly filtered to `no_trade` |
| `Stage 2` | actionability threshold sweep v1 | completed | selected `min_confidence=0.80`, `min_mfe_r=1.0`; false-positive actionable count drops to `0` |

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
| ce_supcon_long_short_pressure_aux_v1 | 0.8870 |
| ce_supcon_long_v1_seed_41 | 0.8880 |
| ce_supcon_long_v1_seed_42 | 0.8899 |
| ce_supcon_long_v1_seed_43 | 0.8881 |
| ce_supcon_long_frontload_v1_epoch_050 | 0.8893 |
| ce_supcon_long_supcon_dominant_warmup_v1_seed_41_epoch_010 | 0.8913 |
| ce_supcon_long_supcon_dominant_warmup_v1_seed_42_epoch_040 | 0.8902 |
| ce_supcon_long_branch_ce_aux_v1_seed_41_epoch_020 | 0.8927 |
| ce_supcon_long_branch_ce_aux_v1_seed_42_epoch_020 | 0.8911 |

Yorum: tüm koşularda F1 değerleri çok yakın. Bu metrik tek başına deney kararını belirleyemiyor.

### Latent Geometry — `z_long` NN Metrikleri

Karar metriği olarak `z_long` seçilmesinin nedeni: `z_short` tüm koşularda stabil kalıyor, asıl değişim ve risk buraya yansıyor.

**Val split:**

| Variant | label agree ↑ | symbol agree ↓ |
|---|---|---|
| ce_only | 0.4442 | 0.9093 |
| ce_supcon_v1 | 0.4746 | 0.9493 |
| ce_supcon_long_v1 | 0.8096 | 0.3171 |
| ce_supcon_long_v2 | 0.7955 | 0.4543 |
| ce_supcon_long_symbol_v1 | 0.4741 | 0.9475 |
| ce_supcon_long_symbol_v2 | 0.4661 | 0.9233 |
| ce_supcon_long_factor_v1 | 0.8004 | 0.7466 |
| ce_supcon_long_aux_v1 | 0.7996 | 0.3221 |
| ce_supcon_long_aux_v2 | 0.7910 | 0.7839 |
| ce_supcon_long_short_pressure_aux_v1 | 0.4599 | 0.9343 |
| ce_supcon_long_v1_seed_41 | 0.7613 | 0.8190 |
| ce_supcon_long_v1_seed_42 | 0.8032 | 0.2886 |
| ce_supcon_long_v1_seed_43 | 0.7887 | 0.7623 |
| ce_supcon_long_frontload_v1_epoch_050 | 0.8032 | 0.3408 |
| **ce_supcon_long_supcon_dominant_warmup_v1_seed_41_epoch_010** | **0.8132** | **0.2914** |
| **ce_supcon_long_supcon_dominant_warmup_v1_seed_42_epoch_040** | **0.8193** | **0.3077** |
| ce_supcon_long_branch_ce_aux_v1_seed_41_epoch_020 | 0.8227 | 0.2853 |
| ce_supcon_long_branch_ce_aux_v1_seed_42_epoch_020 | 0.8142 | 0.3024 |

**Test split:**

| Variant | label agree ↑ | symbol agree ↓ |
|---|---|---|
| ce_only | 0.4504 | 0.9115 |
| ce_supcon_v1 | 0.4838 | 0.9470 |
| ce_supcon_long_v1 | 0.8080 | 0.2987 |
| ce_supcon_long_v2 | 0.7965 | 0.4543 |
| ce_supcon_long_symbol_v1 | 0.4840 | 0.9404 |
| ce_supcon_long_symbol_v2 | 0.4748 | 0.9150 |
| ce_supcon_long_factor_v1 | 0.7994 | 0.7250 |
| ce_supcon_long_aux_v1 | 0.7977 | 0.3141 |
| ce_supcon_long_aux_v2 | 0.7963 | 0.7807 |
| ce_supcon_long_short_pressure_aux_v1 | 0.4742 | 0.9321 |
| ce_supcon_long_v1_seed_41 | 0.7604 | 0.8082 |
| ce_supcon_long_v1_seed_42 | 0.8019 | 0.2724 |
| ce_supcon_long_v1_seed_43 | 0.7934 | 0.7049 |
| ce_supcon_long_frontload_v1_epoch_050 | 0.8011 | 0.3225 |
| **ce_supcon_long_supcon_dominant_warmup_v1_seed_41_epoch_010** | **0.8102** | **0.2826** |
| **ce_supcon_long_supcon_dominant_warmup_v1_seed_42_epoch_040** | **0.8158** | **0.2787** |
| ce_supcon_long_branch_ce_aux_v1_seed_41_epoch_020 | 0.8220 | 0.2745 |
| ce_supcon_long_branch_ce_aux_v1_seed_42_epoch_020 | 0.8125 | 0.2754 |

### Latent Geometry — `z_fused` NN Metrikleri

**Val split:**

| Variant | label agree ↑ | symbol agree ↓ |
|---|---|---|
| ce_only | 0.8437 | 0.5739 |
| ce_supcon_v1 | 0.8431 | 0.6001 |
| ce_supcon_long_v1 | 0.8424 | 0.2986 |
| ce_supcon_long_v2 | 0.8401 | 0.3530 |
| ce_supcon_long_symbol_v1 | 0.8451 | 0.6362 |
| ce_supcon_long_symbol_v2 | 0.8446 | 0.6111 |
| ce_supcon_long_factor_v1 | 0.8469 | 0.4987 |
| ce_supcon_long_aux_v1 | 0.8447 | 0.2831 |
| ce_supcon_long_aux_v2 | 0.8443 | 0.4197 |
| ce_supcon_long_short_pressure_aux_v1 | 0.8399 | 0.5436 |
| ce_supcon_long_v1_seed_41 | 0.8428 | 0.4937 |
| ce_supcon_long_v1_seed_42 | 0.8438 | 0.2835 |
| ce_supcon_long_v1_seed_43 | 0.8419 | 0.3570 |
| ce_supcon_long_frontload_v1_epoch_050 | 0.8344 | 0.3117 |
| **ce_supcon_long_supcon_dominant_warmup_v1_seed_41_epoch_010** | **0.8395** | **0.3273** |
| **ce_supcon_long_supcon_dominant_warmup_v1_seed_42_epoch_040** | **0.8425** | **0.2961** |
| ce_supcon_long_branch_ce_aux_v1_seed_41_epoch_020 | 0.8429 | 0.2991 |
| ce_supcon_long_branch_ce_aux_v1_seed_42_epoch_020 | 0.8411 | 0.3101 |

**Test split:**

| Variant | label agree ↑ | symbol agree ↓ |
|---|---|---|
| ce_only | 0.8445 | 0.5970 |
| ce_supcon_v1 | 0.8397 | 0.6346 |
| ce_supcon_long_v1 | 0.8425 | 0.2784 |
| ce_supcon_long_v2 | 0.8414 | 0.3613 |
| ce_supcon_long_symbol_v1 | 0.8348 | 0.6509 |
| ce_supcon_long_symbol_v2 | 0.8394 | 0.6595 |
| ce_supcon_long_factor_v1 | 0.8446 | 0.5026 |
| ce_supcon_long_aux_v1 | 0.8398 | 0.2816 |
| ce_supcon_long_aux_v2 | 0.8457 | 0.4094 |
| ce_supcon_long_short_pressure_aux_v1 | 0.8424 | 0.5701 |
| ce_supcon_long_v1_seed_41 | 0.8359 | 0.4225 |
| ce_supcon_long_v1_seed_42 | 0.8374 | 0.2766 |
| ce_supcon_long_v1_seed_43 | 0.8392 | 0.3934 |
| ce_supcon_long_frontload_v1_epoch_050 | 0.8365 | 0.3009 |
| **ce_supcon_long_supcon_dominant_warmup_v1_seed_41_epoch_010** | **0.8361** | **0.3012** |
| **ce_supcon_long_supcon_dominant_warmup_v1_seed_42_epoch_040** | **0.8422** | **0.2839** |
| ce_supcon_long_branch_ce_aux_v1_seed_41_epoch_020 | 0.8418 | 0.2913 |
| ce_supcon_long_branch_ce_aux_v1_seed_42_epoch_020 | 0.8416 | 0.2842 |

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
| ce_supcon_long_short_pressure_aux_v1 | 0.5774 | 0.2621 | 0.5702 |

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
12. `2026-05-13-ce-supcon-long-aux-v2-plan.md`
13. `2026-05-13-ce-supcon-long-aux-v2-readout.md`
14. `2026-05-13-ce-supcon-long-short-pressure-aux-v1-plan.md`
15. `2026-05-13-ce-supcon-long-short-pressure-aux-v1-readout.md`
16. `2026-05-16-ce-supcon-long-v1-seed-stability-plan.md`
17. `2026-05-16-ce-supcon-long-v1-seed-stability-readout.md`
18. `2026-05-16-ce-supcon-long-frontload-v1-plan.md`
19. `2026-05-16-ce-supcon-long-frontload-v1-readout.md`
20. `2026-05-16-ce-supcon-long-supcon-warmup-v1-plan.md`
21. `2026-05-17-ce-supcon-long-supcon-warmup-v1-readout.md`
22. `2026-05-17-ce-supcon-long-supcon-dominant-warmup-v1-plan.md`
23. `2026-05-17-ce-supcon-long-supcon-dominant-warmup-v1-readout.md`
24. `2026-05-17-ce-supcon-long-branch-ce-aux-v1-plan.md`
25. `2026-05-17-ce-supcon-long-branch-ce-aux-v1-readout.md`

Kısa sonuç:

- fused-projection SupCon yön olarak karışık
- branch-aware long projection en iyi sonuç
- weight düşürme fayda getirmedi
- ilk symbol-aware deneme başarısız oldu
- `long symbol v2` reddedildi
- `long factor v1` reddedildi; pressure sinyali güçlendi ama symbol shortcut geri döndü
- `long aux v1` mixed; pressure ve borderline iyileşti ama `z_long` geometry hafif geriledi
- `long aux v2` reddedildi; pressure-only auxiliary symbol shortcut'ı geri getirdi
- `long short-pressure aux v1` reddedildi; direct pressure gradient izolasyonu `z_long` geometry'yi korumadı
- `long v1 seed stability`: `ce_supcon_long_v1` best observed checkpoint olarak kaldı ama setup seed-sensitive
- `long frontload v1`: SupCon'u erken aktive etti; `seed_42` güçlü kaldı, `seed_43` toparlandı, fakat `seed_41` hâlâ symbol-heavy olduğu için tam seed-stable değil
- `long SupCon-only warmup v1`: erken SupCon aktivasyonunu çözdü ama hard CE=0 warmup branch dengesini bozdu
- `long SupCon-dominant warmup v1`: şu ana kadarki en iyi dengeli schedule adayı; seed 41 ve seed 42'de `z_long` temizlendi, fakat seçilen epoch seed'e bağlı kaldığı için sonraki hat objective-level `z_long` constraint olmalı
- `long branch CE aux v1`: selected for downstream transfer; `seed_41/epoch_020.pt` Stage 1A aktarım adayı seçildi, seed 42 bucket asimetrisi sonraki model iyileştirme konusu

## Worklogs

Bugün için `worklogs` altında kalan ana notlar:

- `2026-04-13-latent-export-and-analysis-infra.md`
- `2026-04-13-hard-case-domain-buckets.md`
- `2026-04-15-branch-aware-supcon-implementation.md`
- `2026-04-16-symbol-aware-supcon-implementation.md`

Bu notlar deney sonucu değil, tasarım / implementasyon / karar notlarıdır.
