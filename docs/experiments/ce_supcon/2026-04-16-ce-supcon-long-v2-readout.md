# Worklog: 2026-04-16 — CE + SupCon Long V2 Readout

**Status**: Complete ✓  
**Goal**: `supcon_weight = 0.03` ile çalışan ikinci branch-aware koşunun, `ce_supcon_long_v1`e göre daha iyi bir denge verip vermediğini kayda geçirmek

---

## Experiment Setup

Bu koşu, `ce_supcon_long_v1` ile aynı mimariyi kullandı:

- classifier: `z_fused`
- contrastive target: `z_long_proj`
- objective: `CE + lambda * SupCon(z_long_proj, y)`

Tek anlamlı değişiklik:

- `lambda_supcon = 0.03`

Checkpoint:

- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v2`

## Classifier Summary

En iyi val macro-F1:

- `CE-only`: `0.8901`
- `ce_supcon_long_v1`: `0.8900`
- `ce_supcon_long_v2`: `0.8892`

Bu nedenle classifier tarafında ilk sonuç nettir:

- `v2`, `v1`e göre iyileşme getirmedi
- küçük de olsa geri gitti

## Main Readout

Asıl soru şuydu:

- `SupCon` ağırlığını düşürmek, branch-aware geometry kazancını korurken classifier trade-off'unu azaltacak mı?

Bu koşunun cevabı:

- hayır

Çünkü `v2`'de:

- classifier tarafı toparlanmadı
- latent geometry tarafında da `v1`in en güçlü etkileri zayıfladı

## Overall Embedding Readout

### `z_long`

`v1`'in ana gücü şuydu:

- `z_long` çok daha label-aware oluyordu
- aynı anda symbol shortcut sert biçimde kırılıyordu

`v2`'de bu etkinin daha zayıf bir versiyonu görüldü.

Val:

- `v1`: label `0.8096`, symbol `0.3171`
- `v2`: label `0.7955`, symbol `0.4543`

Test:

- `v1`: label `0.8080`, symbol `0.2987`
- `v2`: label `0.7965`, symbol `0.4543`

Yani:

- label-aware etki azaldı
- symbol shortcut geri geldi

### `z_long_proj`

Projection uzayında da aynı zayıflama var.

Val:

- `v1`: label `0.8079`, symbol `0.2661`
- `v2`: label `0.7919`, symbol `0.3624`

Test:

- `v1`: label `0.8067`, symbol `0.2621`
- `v2`: label `0.7938`, symbol `0.3819`

Bu, contrastive hedefin kendisinin de daha az temiz bir uzay ürettiğini gösteriyor.

### `z_fused`

En kritik temsil `z_fused` için de `v2`, `v1`den daha zayıf.

Val:

- `v1`: label `0.8424`, symbol `0.2986`
- `v2`: label `0.8401`, symbol `0.3530`

Test:

- `v1`: label `0.8425`, symbol `0.2784`
- `v2`: label `0.8414`, symbol `0.3613`

Bu şu anlama geliyor:

- fused latent'te `v1`in sağladığı shortcut temizliği korunamadı
- semantik kalite de hafifçe geriledi

## Bucket Readout

Bucket bazında `v2` tamamen kötü değil.

Bazı dilimlerde:

- `close_confirmed_break_up`
- `close_confirmed_break_down`
- bazı test `wick` örnekleri

için küçük yerel artılar var.

Ancak genel tablo açısından iki kritik gerçek değişmedi:

1. `borderline_intact_break` hâlâ açık problem
2. `v1`in genel shortcut temizliği + semantik denge kombinasyonu `v2`de daha zayıf

Özellikle:

- `high_vol_intact_wick_sweep`
- fused symbol agreement
- long branch symbol agreement

taraflarında `v1` daha güçlü.

## Overall Decision

Bu koşu için en dürüst karar şudur:

- `supcon_weight = 0.03` yönü ilerletilmemeli

Çünkü:

- classifier trade-off anlamlı biçimde düzelmedi
- contrastive etkinin asıl değerli tarafı seyreldi

Başka deyişle:

- daha düşük ağırlık, daha iyi denge getirmedi
- aksine `v1`in en iyi özelliğini zayıflattı

## What This Means

Bu not sonrası karar netleşti:

- branch-aware SupCon doğru yönde
- ama `0.05` civarı etki, `0.03`ten daha iyi
- sıradaki deney ekseni ağırlığı küçültmek olmamalı

Dolayısıyla yeni soru şu olmalı:

- aynı branch-aware setup korunurken, positive/negative tanımı ve batch composition nasıl daha symbol-aware hale getirilir?

## Recommended Next Direction

Bir sonraki mantıklı deney:

- `symbol-aware positive/sampling`

Yani:

- positive eşleşmeleri yalnızca aynı label ile değil, mümkünse farklı symbol kısıtıyla kurmak
- batch içi symbol karışımını daha dengeli hale getirmek

Amaç:

- `v1`de gördüğümüz güçlü semantik düzenlemeyi korumak
- shortcut etkisini daha da bilinçli biçimde baskılamak
- classifier tarafındaki gereksiz yan etkileri azaltmak

## Related

- `docs/experiments/ce_supcon/2026-04-15-ce-supcon-long-v1-readout.md`
- `docs/worklogs/2026-04-15-branch-aware-supcon-implementation.md`

---

**Status**: Complete ✓
