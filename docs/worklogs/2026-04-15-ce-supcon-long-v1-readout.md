# Worklog: 2026-04-15 — CE + SupCon Long V1 Readout

**Status**: Complete ✓  
**Goal**: `branch-aware SupCon` varyantının `CE-only` ve ilk fused-projection `CE + SupCon` koşusuna göre ne getirdiğini kayda geçirmek

---

## Experiment Setup

Bu koşu şu ayarla çalıştırıldı:

- dataset: `core4`
- timeframe: `15m`
- short window: `48`
- long window: `288`
- objective: `CE + 0.05 * SupCon(z_long_proj, y)`
- temperature: `0.1`
- batch size: `64`
- checkpoint dir: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1`

Model tarafında:

- classifier yine `z_fused` üstünde kaldı
- contrastive loss yalnızca `z_long_proj` üstünde hesaplandı

Yani bu koşu, fused projection yerine long branch projection'ı contrastive olarak düzenlemeyi hedefledi.

## Classifier Summary

En iyi val macro-F1:

- `CE-only`: `0.8901`
- `CE + SupCon v1`: `0.8908`
- `CE + SupCon long v1`: `0.8900`

Dolayısıyla classifier tarafında tablo şu:

- `ce_supcon_long_v1`, `CE-only`e çok yakın
- ama ilk `ce_supcon_v1` koşusundan hafif zayıf

Bu fark küçük. Bu yüzden bu deneyin asıl değeri classifier skorundan çok latent geometry tarafında aranmalı.

## Main Readout

Bu koşunun en güçlü bulgusu şudur:

- `z_long` branch ilk kez hem çok daha label-aware
- hem de çok daha az symbol-heavy hale geldi

Bu, ilk `ce_supcon_v1` koşusundan temel olarak farklıdır.

İlk fused-projection varyantında:

- `z_long` label agreement artıyordu
- ama symbol shortcut da büyüyordu

Branch-aware varyantta ise:

- label agreement çok güçlü biçimde artıyor
- symbol agreement ise dramatik biçimde düşüyor

Bu nedenle araştırma açısından bu koşu, ilk `SupCon` varyantından daha doğru yönde bir deneydir.

## Val Readout

### `z_long`

Val tarafında:

- mean NN label agreement: `0.4442 -> 0.8096`
- top-1 label match: `0.5692 -> 0.8140`
- mean NN symbol agreement: `0.9093 -> 0.3171`
- top-1 symbol match: `0.9678 -> 0.4006`

Bu çok güçlü bir yeniden düzenlenme demektir.

Yani long branch artık coin kimliğine yaslanan bir context taşıyıcı olmaktan çıkıp, yapısal state semantiğine çok daha yakın bir temsil üretmektedir.

### `z_fused`

Val tarafında `z_fused` için:

- mean NN label agreement: `0.8437 -> 0.8424`
- top-1 label match: `0.8466 -> 0.8441`
- mean NN symbol agreement: `0.5739 -> 0.2986`
- top-1 symbol match: `0.6906 -> 0.3413`

Bu nokta kritiktir.

Çünkü burada tablo şunu söylüyor:

- fused latent'in semantik gücü büyük ölçüde korunuyor
- ama symbol shortcut etkisi belirgin biçimde azalıyor

Bu, Stage 1A araştırma hedefi açısından çok değerli bir sonuçtur.

### `z_short`

Val tarafında `z_short` belirgin biçimde zayıfladı:

- mean NN label agreement: `0.8436 -> 0.7874`
- top-1 label match: `0.8442 -> 0.7929`

Bu olumsuz bir sinyal.

Ancak burada kritik nokta şudur:

- asıl araştırma hedefi `z_short`'ı izole korumak değil
- fused latent'i daha semantik ve daha az shortcut-bound hale getirmek

Dolayısıyla bu düşüş not edilmeli, ama deneyin toplam değerini tek başına belirlememelidir.

## Test Readout

Test split val bulgusunu büyük ölçüde doğruladı.

### `z_long`

- mean NN label agreement: `0.4504 -> 0.8080`
- top-1 label match: `0.5689 -> 0.8112`
- mean NN symbol agreement: `0.9115 -> 0.2987`
- top-1 symbol match: `0.9707 -> 0.3696`

Bu, long branch'teki yeniden düzenlenmenin yalnızca val içi bir artefakt olmadığını gösteriyor.

### `z_fused`

- mean NN label agreement: `0.8445 -> 0.8425`
- top-1 label match: `0.8511 -> 0.8488`
- mean NN symbol agreement: `0.5970 -> 0.2784`
- top-1 symbol match: `0.7025 -> 0.3051`

Bu da yine aynı sonucu destekliyor:

- fused latent semantik olarak büyük ölçüde korunuyor
- ama coin identity clustering ciddi biçimde kırılıyor

### `z_short`

Testte de `z_short` hafif değil, belirgin bir geri çekilme gösteriyor:

- mean NN label agreement: `0.8438 -> 0.8027`
- top-1 label match: `0.8438 -> 0.8116`

Bu, branch-aware baskının dolaylı etkisinin short branch'te de hissedildiğini düşündürüyor.

## `z_long_proj` Readout

Yeni projection uzayı olan `z_long_proj` kendi başına da çok güçlü çıktı.

Val:

- mean NN label agreement: `0.8079`
- mean NN symbol agreement: `0.2661`

Test:

- mean NN label agreement: `0.8067`
- mean NN symbol agreement: `0.2621`

Bu iki sayı birlikte okunduğunda, branch-aware contrastive objective'in doğrudan hedeflediği uzayın gerçekten temiz bir semantik düzen kurduğu söylenebilir.

## Bucket Readout

Asıl izlenen bucket'lar:

- `borderline_intact_break`
- `wick_sweep_up`
- `wick_sweep_down`
- `high_vol_intact_wick_sweep`
- `close_confirmed_break_up`
- `close_confirmed_break_down`

### 1. `z_long` bucket düzeyinde dramatik biçimde toparlandı

Hem val hem test'te:

- `borderline_intact_break`
- `wick_sweep_*`
- `close_confirmed_break_*`
- `true_break_*`

bucket'larında `z_long` label agreement ve top-1 label match ciddi biçimde yükseldi.

Aynı anda symbol agreement sert biçimde düştü.

Bu, long branch'in batch contrastive objective altında artık "coin shortcut" değil "state manifold" öğrenmeye başladığını düşündürüyor.

### 2. `z_fused` tarafında ana kazanç shortcut temizliği

Fused latent bucket bazında genellikle:

- label agreement'i CE-only seviyesine çok yakın tutuyor
- ama symbol agreement'i neredeyse tüm bucket'larda ciddi biçimde düşürüyor

Bu, ilk `ce_supcon_v1` koşusundan daha iyi bir sonuçtur.

İlk varyantta:

- fused latent'te symbol bias büyüyordu

Bu varyantta ise:

- fused latent'te symbol bias keskin biçimde düşüyor

### 3. `borderline_intact_break` hâlâ tam çözülmüş değil

En kritik sınır bucket'ı olan `borderline_intact_break` için:

- `z_long` çok daha iyi
- ama `z_fused` tarafında tam bir semantik sıçrama yok
- misclassified rate de düşmüyor

Yani bu deney, problemi çözme yönünde daha doğru bir adım olsa da, en zor sınır bucket'ı henüz tamamen açılmış değil.

## Overall Decision

Bu koşu için en dürüst karar şu:

- classifier tarafında küçük bir trade-off var
- ama latent geometry tarafında açık bir ilerleme var
- özellikle `z_long` branch semantik olarak dramatik biçimde iyileşti
- en önemlisi, fused latent'te symbol shortcut ciddi biçimde kırıldı

Bu nedenle araştırma hedefi açısından:

- `ce_supcon_long_v1`, ilk `ce_supcon_v1` koşusundan daha doğru yönde bir deneydir

## What This Means

Bu not sonrası artık daha net biliyoruz:

- `SupCon` sinyali tutulmalı
- ama fused projection üstünde değil, hedefli branch üstünde uygulandığında daha iyi davranıyor
- özellikle `z_long` branch, contrastive düzenleme için doğru müdahale noktası gibi görünüyor

Başka deyişle:

- problem `SupCon` fikrinde değil
- problem ilk varyantta `SupCon`'un uygulama noktasındaydı

## Recommended Next Step

Bu koşudan sonra en mantıklı küçük varyant:

- aynı branch-aware yapı
- daha düşük contrastive ağırlık

Önerilen ilk takip koşusu:

- `supcon_weight = 0.03`

Amaç:

- fused latent'te elde edilen shortcut temizliğini korumak
- classifier tarafındaki küçük trade-off'u biraz azaltmak
- özellikle `z_short` üzerindeki dolaylı baskıyı hafifletmek

## Related

- `docs/worklogs/2026-04-15-branch-aware-supcon-implementation.md`
- `docs/worklogs/2026-04-14-ce-supcon-v1-readout.md`
- `docs/worklogs/2026-04-14-ce-only-synthesis.md`

---

**Status**: Complete ✓
