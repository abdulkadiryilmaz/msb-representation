# Experiment Index

Bu dosya, Stage 1A deneylerini yüksek seviyede takip etmek için kısa bir giriş noktasıdır.

## Nasıl Okunmalı

- araştırma framing / temel varsayımlar: `docs/foundations/`
- implementasyon ve karar notları: `docs/worklogs/`
- deney planı ve deney readout'ları: `docs/experiments/`
- ortak süreç: `docs/EXPERIMENT_WORKFLOW.md`

## Foundations

- `docs/foundations/stage1-msb-representation-framing.md`

Bu belge, Stage 1A araştırma yönünü ve representation-learning çerçevesini tanımlar. Deney okumalarından önce buradan başlamak en doğru giriş olur.

## Current Active Experiment

- active run: `CE + SupCon | long symbol v2`
- plan: `docs/experiments/ce_supcon/2026-04-17-ce-supcon-long-symbol-v2-plan.md`
- baseline to beat: `ce_supcon_long_v1`
- current hypothesis:
  - `same-label + different-symbol` çiftleri positive
  - `same-label + same-symbol` çiftleri neutral
  - bu mask, `v1`deki semantik kazancı korurken symbol shortcut etkisini daha bilinçli baskılayabilir
- success criteria:
  - `z_long` / `z_long_proj` label agreement yüksek kalmalı
  - `z_fused` symbol agreement `v1`e yakın veya daha düşük olmalı
  - classifier tarafı `v1`den belirgin kötü olmamalı

## Current Summary

| Family | Variant | Status | Kısa Yorum |
|---|---|---|---|
| `CE-only` | core4 baseline | reference | güçlü ve stabil baseline |
| `CE + SupCon` | fused projection v1 | mixed | `z_long` iyileşti ama symbol bias büyüdü |
| `CE + SupCon` | long projection v1 | best current | en iyi latent geometry dengesi |
| `CE + SupCon` | long projection v2 | rejected | weight düşürmek fayda getirmedi |
| `CE + SupCon` | long symbol v1 | rejected | geometry yeniden symbol-heavy oldu |
| `CE + SupCon` | long symbol v2 | in_progress | cross-symbol positive, same-symbol neutral mask test ediliyor |

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

Kısa sonuç:

- fused-projection SupCon yön olarak karışık
- branch-aware long projection en iyi sonuç
- weight düşürme fayda getirmedi
- ilk symbol-aware deneme başarısız oldu
- `long symbol v2` şu anda koşuyor

## Worklogs

Bugün için `worklogs` altında kalan ana notlar:

- `2026-04-13-latent-export-and-analysis-infra.md`
- `2026-04-13-hard-case-domain-buckets.md`
- `2026-04-15-branch-aware-supcon-implementation.md`
- `2026-04-16-symbol-aware-supcon-implementation.md`

Bu notlar deney sonucu değil, tasarım / implementasyon / karar notlarıdır.
