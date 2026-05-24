# 2026-05-24 - Stage 1A Latent Proximity Probe v1 Readout

## Amaç

Stage 1B görsel incelemesinde bazı pencerelerde fiyat structural level'a çok yakın olmasına rağmen predictor yönü ters okuyordu.

Bu diagnostik şu soruyu test eder:

> Seçili Stage 1A latent'i, mevcut `intact` pencerelerde bull/bear level proximity bilgisini taşıyor mu?

Bu çalışma yeni bir predictor eğitimi değildir. Frozen Stage 1A latent export'ları üzerinde linear/MLP probe çalıştırır.

## Kaynaklar

Stage 1A source:

```text
ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt
```

Latent exports:

```text
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_train_latents.npz
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_test_latents.npz
```

Label context:

```text
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_train_h4_h8_h16_v2.parquet
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_test_h4_h8_h16_v2.parquet
```

Script:

```text
scripts/probe_stage1a_proximity.py
```

## Probe Targets

Probe yalnızca `current_label = intact` olan örneklerde çalıştırıldı.

Hedefler:

| Target | Tanım |
|---|---|
| `nearest_level_side` | Mevcut close'a en yakın structural level `bull` mu `bear` mı? |
| `nearest_distance_bucket` | En yakın level'a uzaklık bucket'ı: `very_near / near / watch / far` |
| `wick_breach_side` | Recent zone içinde wick breach tarafı: `none / bull / bear / both` |

Distance bucket kuralı:

| Bucket | nearest distance |
|---|---:|
| `very_near` | `<= 0.10%` |
| `near` | `<= 0.20%` |
| `watch` | `<= 0.50%` |
| `far` | `> 0.50%` |

## Komutlar

Örnek komut:

```bash
PYTHONPATH=src python scripts/probe_stage1a_proximity.py \
  --train-latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_train_latents.npz \
  --train-label-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_train_h4_h8_h16_v2.parquet \
  --eval-latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/selected_epoch_020_test_latents.npz \
  --eval-label-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_labels_test_h4_h8_h16_v2.parquet \
  --embedding-key z_fused \
  --target nearest_level_side \
  --output-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_nearest_side_test_z_fused
```

Her target için `z_short`, `z_long`, `z_fused` ayrı çalıştırıldı.

## Sonuçlar

Test split sample count: `4753` intact örnek.

### `nearest_level_side`

Label counts:

```text
bear: 2400
bull: 2353
```

| Embedding | Majority macro F1 | Linear macro F1 | MLP macro F1 |
|---|---:|---:|---:|
| `z_short` | 0.3311 | 0.7406 | 0.7424 |
| `z_long` | 0.3311 | 0.7223 | 0.7193 |
| `z_fused` | 0.3311 | 0.7644 | 0.7560 |

Yorum:

Latent, en yakın level tarafını anlamlı biçimde taşıyor. `z_fused` en güçlü, fakat `z_short` ve `z_long` da çoğunluk baseline'ın çok üzerinde.

### `nearest_distance_bucket`

Label counts:

```text
very_near: 355
near: 399
watch: 1547
far: 2452
```

| Embedding | Majority macro F1 | Linear macro F1 | MLP macro F1 |
|---|---:|---:|---:|
| `z_short` | 0.1702 | 0.4175 | 0.3669 |
| `z_long` | 0.1702 | 0.3552 | 0.2964 |
| `z_fused` | 0.1702 | 0.4295 | 0.3841 |

Yorum:

Uzaklık bucket'ı okunabiliyor, fakat nearest side kadar güçlü değil. Bu, latent'in direction/proximity taraf bilgisini taşıdığını ama mesafe hassasiyetinin zayıf olduğunu gösteriyor.

### `wick_breach_side`

Label counts:

```text
none: 2915
bull: 805
bear: 870
both: 163
```

| Embedding | Majority macro F1 | Linear macro F1 | MLP macro F1 |
|---|---:|---:|---:|
| `z_short` | 0.1901 | 0.4803 | 0.5033 |
| `z_long` | 0.1901 | 0.5104 | 0.5410 |
| `z_fused` | 0.1901 | 0.5552 | 0.5953 |

Yorum:

Wick breach sinyali latentte orta güçte okunabiliyor. `z_fused` yine en güçlü; `z_long`un `z_short`tan daha iyi olması, broader context branch'in level-interaction sinyalini tamamen kaybetmediğini gösteriyor.

## Örnek 454 Bağlantısı

İncelenen BTC sample:

```text
index = 454
timestamp = 2026-01-18 23:00
current_label = intact
target true = bearish
predictor decision = bullish
bear_distance_pct = 0.087%
```

Bu probe sonuçlarına göre:

- latent genel olarak nearest side bilgisini taşıyor
- fakat precise distance / threshold-adjacent pressure bilgisi daha zayıf
- Stage 1B predictor, bu bilgiyi doğru kullanmamış olabilir
- ayrıca predictor input'una explicit proximity features eklemek hâlâ makul bir Stage 1B decision-module iyileştirmesi olabilir

## Karar

Bu diagnostik, Stage 1A latent'inin proximity bilgisini tamamen kaybetmediğini gösteriyor.

Ancak TradePlan'a giden Stage 1B için iki açık problem kalıyor:

1. `nearest_level_side` okunabilir olsa da `nearest_distance_bucket` zayıf.
2. Stage 1B predictor, latentte mevcut olan proximity bilgisini her zaman doğru yön kararına çeviremiyor.

Sonraki en mantıklı adım:

```text
Stage 1B predictor v3 = z_fused + explicit proximity / level-interaction features
```

Bu, Stage 1A temsil başarısını değil, Stage 1B karar modülünün daha fazla price/level context kullanmasını test eder.

## Artifacts

```text
scripts/probe_stage1a_proximity.py

data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_nearest_side_test_z_short/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_nearest_side_test_z_long/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_nearest_side_test_z_fused/

data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_distance_bucket_test_z_short/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_distance_bucket_test_z_long/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_distance_bucket_test_z_fused/

data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_wick_breach_test_z_short/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_wick_breach_test_z_long/
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/probes/proximity_wick_breach_test_z_fused/
```

## Verification

```bash
python -m py_compile scripts/probe_stage1a_proximity.py
ruff check scripts/probe_stage1a_proximity.py
```
