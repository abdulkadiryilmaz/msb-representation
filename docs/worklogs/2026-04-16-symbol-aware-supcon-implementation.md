# Worklog: 2026-04-16 — Symbol-Aware SupCon Implementation

**Status**: Complete ✓  
**Goal**: Branch-aware `SupCon` setup'ını, positive tanımı ve batch composition tarafında symbol-aware hale getirerek bir sonraki deney eksenini açmak

---

## Neden Bu Değişiklik Yapıldı

Önceki branch-aware sonuçlar iki şeyi netleştirdi:

- `ce_supcon_long_v1`, fused-projection `SupCon` varyantından daha doğru yönde
- `ce_supcon_long_v2` ile ağırlığı düşürmek beklenen iyileşmeyi getirmedi

Dolayısıyla yeni deney ekseni artık `lambda` aramak değil.

Asıl soru şu oldu:

- same-label positive tanımını symbol shortcut etkisini baskılayacak biçimde daraltırsak daha temiz bir geometry elde eder miyiz?

Bu worklog, bu amaçla eklenen `symbol-aware positive/sampling` yüzeyini kaydeder.

## Tasarım Kararı

İki yeni mekanizma eklendi:

1. symbol-aware positive mode

- aynı label
- ama farklı symbol

eşleşmelerini positive sayan contrastive mod

2. symbol-balanced batch sampling

- train batch'lerinin tek symbol çevresinde yoğunlaşmasını azaltan round-robin sampler

Bu iki mekanizma birlikte şu problemi hedefliyor:

- label-semantic attraction korunurken
- coin identity shortcut etkisi daha bilinçli biçimde baskılansın

## Kod Değişiklikleri

### 1. `SupConLoss`

`SupConLoss.forward(...)` artık opsiyonel olarak:

- `symbol_ids`
- `positive_mode`

alabiliyor.

Desteklenen modlar:

- `label`
- `label_diff_symbol`

`label_diff_symbol` modunda positive pair ancak şu durumda oluşuyor:

- label aynı
- symbol farklı

### 2. Trainer

Trainer config'e yeni alan eklendi:

- `supcon_positive_mode`

Eğer mod `label_diff_symbol` ise:

- batch içindeki `meta.symbol` alanlarından geçici `symbol_ids` üretilir
- loss buna göre hesaplanır

### 3. Train CLI

Yeni flag'ler:

- `--supcon-positive-mode {label,label_diff_symbol}`
- `--supcon-balance-symbols`

Bu sayede deney yüzeyi şu kombinasyonları destekler:

- klasik same-label positive
- same-label ama different-symbol positive
- istenirse symbol-balanced batch sampler

### 4. Dataset / Sampler

Yeni sınıf:

- `Stage1ASymbolBalancedBatchSampler`

Bu sampler:

- symbol havuzlarını round-robin tüketir
- her batch içinde symbol karışımını yüksek tutmaya çalışır

Amaç, contrastive objective'in tek symbol içi tekrarlarla domine olmasını azaltmaktır.

## Doğrulama

Çalıştırılan testler:

```bash
PYTHONPATH=src pytest -q tests/test_stage1a_losses.py tests/test_stage1a_dataset.py tests/test_stage1a_analysis.py
```

Sonuç:

```text
14 passed in 0.58s
```

## İlk Önerilen Deney

İlk symbol-aware koşu için önerilen komut:

```bash
source .venv/bin/activate
python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 64 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_symbol_v1 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --supcon-positive-mode label_diff_symbol \
  --supcon-balance-symbols
```

Bu koşuda:

- classifier yine `z_fused`
- contrastive target yine `z_long_proj`
- fakat positive eşleşmeler artık symbol-aware
- ve batch composition daha dengeli

## Beklenen Okuma

Bu varyanttan sonra asıl bakılacak şeyler:

- `z_long` ve `z_long_proj` label agreement korunuyor mu?
- `z_fused` symbol shortcut daha da azalıyor mu?
- `borderline_intact_break` ve `wick` bucket'larında cleaner neighbor structure geliyor mu?
- classifier tarafındaki trade-off büyüyor mu, küçülüyor mu?

Başarı için en kritik desen:

- `ce_supcon_long_v1` kadar güçlü veya ona yakın semantik kazanç
- ama daha da düşük symbol agreement

## Related

- `docs/worklogs/2026-04-15-ce-supcon-long-v1-readout.md`
- `docs/worklogs/2026-04-16-ce-supcon-long-v2-readout.md`

---

**Status**: Complete ✓
