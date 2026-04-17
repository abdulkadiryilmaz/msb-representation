# Worklog: 2026-04-16 — CE + SupCon Long Symbol V1 Readout

**Status**: Complete ✓  
**Goal**: İlk `symbol-aware positive/sampling` koşusunun, `ce_supcon_long_v1` branch-aware baseline'ına göre ne verdiğini kayda geçirmek

---

## Experiment Setup

Bu koşu şu ayarla çalıştırıldı:

- classifier: `z_fused`
- contrastive target: `z_long_proj`
- `supcon_weight = 0.05`
- `supcon_temperature = 0.1`
- `supcon_positive_mode = label_diff_symbol`
- `supcon_balance_symbols = true`
- checkpoint dir: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_symbol_v1`

Amaç şuydu:

- branch-aware geometry kazancını korumak
- ama positive tanımını ve batch composition'ı symbol-aware yaparak shortcut etkisini daha da azaltmak

## Classifier Summary

En iyi val macro-F1:

- `CE-only`: `0.8901`
- `ce_supcon_long_v1`: `0.8900`
- `ce_supcon_long_symbol_v1`: `0.8906`

Classifier tarafında ilk bakışta küçük bir artı var.

Ancak bu deney için ana karar ekseni classifier metriği değil, latent geometry.

## Main Readout

Bu koşunun ana sonucu nettir:

- classifier tarafındaki küçük artıya rağmen latent geometry ciddi biçimde kötüleşti

Özellikle:

- `z_long`
- `z_long_proj`
- `z_fused`

temsil uzaylarında symbol clustering yeniden çok baskın hale geldi.

Bu nedenle deney, araştırma hedefi açısından başarısız sayılmalıdır.

## Val Readout

### `z_long`

Val tarafında:

- `ce_supcon_long_v1`: label `0.8096`, symbol `0.3171`
- `ce_supcon_long_symbol_v1`: label `0.4741`, symbol `0.9475`

Bu çok sert bir bozulma.

Yani `z_long`, `v1`de elde edilen temiz semantic manifold'u kaybedip yeniden coin kimliğine yaslanan bir uzaya dönmüş.

### `z_long_proj`

Asıl contrastive hedef uzay da başarısız:

- `v1`: label `0.8079`, symbol `0.2661`
- `symbol_v1`: label `0.3696`, symbol `0.7700`

Bu, symbol-aware positive tanımının hedeflediğimiz etkiyi üretmediğini açık biçimde gösteriyor.

### `z_fused`

Fused latent tarafında da shortcut temizliği kaybedildi:

- `v1`: label `0.8424`, symbol `0.2986`
- `symbol_v1`: label `0.8451`, symbol `0.6362`

Bu noktada semantik kalite yaklaşık korunuyor gibi görünse de, symbol shortcut çok sert biçimde geri dönüyor.

Bu nedenle toplam okuma pozitife dönmüyor.

## Test Readout

Test split val bulgusunu doğruladı.

### `z_long`

- `v1`: label `0.8080`, symbol `0.2987`
- `symbol_v1`: label `0.4840`, symbol `0.9404`

### `z_long_proj`

- `v1`: label `0.8067`, symbol `0.2621`
- `symbol_v1`: label `0.3843`, symbol `0.7722`

### `z_fused`

- `v1`: label `0.8425`, symbol `0.2784`
- `symbol_v1`: label `0.8438`, symbol `0.6589`

Yani test tarafında da tablo aynı:

- semantic gain korunmuyor
- symbol shortcut çok belirgin biçimde geri geliyor

Bu, val tarafındaki bozulmanın tesadüfi olmadığını gösteriyor.

## Why This Failed

Bu koşunun neden başarısız göründüğüne dair en olası açıklama şu:

- `label_diff_symbol` positive tanımı çok sert bir kısıt getirdi
- aynı label + aynı symbol örnekleri artık positive olmaktan çıktı
- ama bu örnekler explicit olarak neutral da yapılmadı

Sonuçta model, batch içindeki yapıyı beklediğimiz gibi kullanmak yerine, daha sert symbol kümelenmesine geri dönmüş olabilir.

Ek bir sinyal:

- `val_supcon_loss` çok düşük ve neredeyse düz seyrediyor

Bu da contrastive objective'in pratikte yeterince zengin positive sinyal üretmediğini düşündürüyor.

## Overall Decision

Bu koşu için en dürüst karar şu:

- `ce_supcon_long_symbol_v1` devam ettirilecek yön değil

Çünkü:

- küçük classifier artısı var
- ama latent geometry açık biçimde daha kötü

Stage 1A için bu kabul edilebilir bir trade değil.

## What This Means

Bu deney sonrası daha net biliyoruz:

- `symbol-aware` fikir tamamen yanlış olmayabilir
- ama `same-label + different-symbol` çiftlerini positive yapmak tek başına yeterli değil
- hatta mevcut formuyla geometry'yi bozuyor

Dolayısıyla bir sonraki symbol-aware varyantta doğru soru şu:

- `same-label + same-symbol` çiftlerini negative gibi davranmaktan çıkarıp neutral yapabilir miyiz?

Yani bir sonraki aday:

- only cross-symbol positives
- same-symbol same-label pairs neutral

şeklinde daha kontrollü bir loss tasarımı olmalı.

## Recommended Next Direction

En mantıklı sonraki adım:

- `ce_supcon_long_v1`i ana referans olarak korumak
- symbol-aware v2'yi loss mask mantığını değiştirerek kurmak

Bu noktada `v1` hâlâ en iyi SupCon varyantı olarak duruyor.

## Related

- `docs/experiments/ce_supcon/2026-04-15-ce-supcon-long-v1-readout.md`
- `docs/experiments/ce_supcon/2026-04-16-ce-supcon-long-v2-readout.md`
- `docs/worklogs/2026-04-16-symbol-aware-supcon-implementation.md`

---

**Status**: Complete ✓
