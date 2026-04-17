# Experiment Plan: 2026-04-17 — CE + SupCon Long Symbol V2

**Status**: In Progress  
**Goal**: `branch-aware SupCon` yapısını koruyup, symbol-aware positive tanımını `neutral same-symbol` mantığıyla yeniden deneyerek `ce_supcon_long_v1` üzerindeki shortcut temizliğini daha kontrollü biçimde geliştirmek

---

## Context

Şu ana kadarki tablo:

- `ce_supcon_long_v1` en iyi latent geometry dengesini verdi
- `ce_supcon_long_v2` (`weight=0.03`) bu dengeyi iyileştirmedi
- `ce_supcon_long_symbol_v1` ise classifier tarafında hafif artı verse de latent geometry'yi bozdu

Özellikle `long_symbol_v1` şunu gösterdi:

- `same-label + different-symbol` positive tanımı tek başına yeterli değil
- `z_long` ve `z_long_proj` yeniden symbol-heavy hale döndü

Dolayısıyla yeni aday, symbol-aware fikri koruyup maske mantığını daha dikkatli kurmak olmalı.

## Experiment Question

Şu soru test ediliyor:

- `same-label + different-symbol` çiftlerini positive,
- `same-label + same-symbol` çiftlerini neutral,
- `different-label` çiftlerini negative

olarak ele almak, `ce_supcon_long_v1`deki semantik kazancı korurken symbol shortcut etkisini daha da azaltabilir mi?

## Method

Bu koşuda kullanılan yapı:

- classifier head: `z_fused`
- contrastive target: `z_long_proj`
- batch sampling: symbol-balanced
- `supcon_weight = 0.05`
- `supcon_temperature = 0.1`

Loss mask mantığı:

- positive:
  - aynı label
  - farklı symbol
- neutral:
  - aynı label
  - aynı symbol
- negative:
  - farklı label

Buradaki kritik fark:

- `long_symbol_v1` aynı label + aynı symbol çiftlerini positive olmaktan çıkarmıştı
- bu yeni varyantta ise bu çiftler itilmiyor, yalnızca nötr bırakılıyor

## Command

```bash
source .venv/bin/activate
python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 64 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_symbol_v2 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode label_diff_symbol_neutral_same_symbol \
  --supcon-balance-symbols
```

## Success Criteria

Başarı için aranacak desen:

1. classifier

- `best_val_macro_f1`, `ce_supcon_long_v1`den belirgin kötü olmamalı

2. `z_long` / `z_long_proj`

- label agreement yüksek kalmalı
- symbol agreement `long_symbol_v1`e göre ciddi biçimde düşmeli

3. `z_fused`

- label agreement `ce_supcon_long_v1`e yakın kalmalı
- symbol agreement mümkünse `ce_supcon_long_v1`e yakın veya daha düşük olmalı

4. bucket readout

- özellikle `borderline_intact_break`
- `wick_sweep_*`
- `high_vol_intact_wick_sweep`

alanlarında cleaner neighbor structure aranmalı

## Baseline To Beat

Bu deneyin asıl referansı:

- `ce_supcon_long_v1`

Çünkü şu an en iyi branch-aware SupCon varyantı odur.

Yan referanslar:

- `ce_only`
- `ce_supcon_long_symbol_v1`

## Planned Readout

Koşu bittikten sonra üretilecek artefaktlar:

- val/test latent export
- val/test hard-case ve bucket analizi
- val/test embedding compare
- `ce_supcon_long_v1` ile yan yana karşılaştırma

## Post-Run Checklist

Koşu tamamlandığında şu sırayla ilerlenmeli:

1. Eğitim özeti kontrolü

- `metadata.json`
- `history.json`
- `best_val_macro_f1`
- early stopping epoch

2. Val artefaktları

- `export_stage1a_latents.py --split val`
- `analyze_stage1a_latents.py`
- `compare_stage1a_embedding_views.py`

3. Test artefaktları

- `export_stage1a_latents.py --split test`
- `analyze_stage1a_latents.py`
- `compare_stage1a_embedding_views.py`

4. Karşılaştırma

- `ce_supcon_long_v1` ile yan yana overall embedding metrikleri
- `z_long`, `z_long_proj`, `z_fused` üstünde label vs symbol agreement
- kritik bucket'larda `borderline_intact_break`, `wick_sweep_*`, `high_vol_intact_wick_sweep`

5. Dokümantasyon

- `2026-04-17-ce-supcon-long-symbol-v2-readout.md` oluştur
- `docs/EXPERIMENT_INDEX.md` durum satırını güncelle
- gerekiyorsa `docs/experiments/ce_supcon/README.md` timeline bölümünü güncelle

## Related

- `docs/experiments/ce_supcon/2026-04-14-ce-supcon-experiment-plan.md`
- `docs/experiments/ce_supcon/2026-04-15-ce-supcon-long-v1-readout.md`
- `docs/experiments/ce_supcon/2026-04-16-ce-supcon-long-symbol-v1-readout.md`
- `docs/worklogs/2026-04-16-symbol-aware-supcon-implementation.md`

---

**Status**: In Progress
