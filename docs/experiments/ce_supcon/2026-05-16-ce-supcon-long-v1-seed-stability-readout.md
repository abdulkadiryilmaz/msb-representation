# Experiment Readout: 2026-05-16 — CE + SupCon Long V1 Seed Stability

**Status**: Completed  
**Family**: `CE + SupCon`  
**Reference checkpoint**: `ce_supcon_long_v1`  
**Runs**: `ce_supcon_long_v1_seed_41`, `ce_supcon_long_v1_seed_42`, `ce_supcon_long_v1_seed_43`  
**Decision**: seed-sensitive / not stable enough

---

## Soru

`ce_supcon_long_v1` ile görülen güçlü `z_long` / `z_long_proj` geometry tek koşuya özgü mü, yoksa farklı seed'lerde tekrar ediyor mu?

Bu deney yeni objective denemesi değildir. Mevcut current-best setup'ın stabilitesini ölçer.

---

## Eğitim Özeti

| Variant | Seed | Device | Best val macro F1 |
|---|---:|---|---:|
| `ce_supcon_long_v1` | unknown | mps | 0.8900 |
| `ce_supcon_long_v1_seed_41` | 41 | mps | 0.8880 |
| `ce_supcon_long_v1_seed_42` | 42 | mps | 0.8899 |
| `ce_supcon_long_v1_seed_43` | 43 | mps | 0.8881 |

Classifier metriği tek başına sorun göstermiyor. Üç seed de val macro F1 tarafında referans bandına yakın.

---

## Test Embedding Geometry

### `z_long`

| Variant | label agreement | symbol agreement | top1 label | top1 symbol |
|---|---:|---:|---:|---:|
| `ce_supcon_long_v1` | 0.8080 | 0.2987 | 0.8112 | 0.3696 |
| `seed_41` | 0.7604 | 0.8082 | 0.7954 | 0.9087 |
| `seed_42` | 0.8019 | 0.2724 | 0.8003 | 0.3040 |
| `seed_43` | 0.7934 | 0.7049 | 0.8097 | 0.8508 |

Okuma:

- `seed_42`, referans koşuya yakın ve hatta symbol agreement tarafında biraz daha iyi.
- `seed_41` ve `seed_43`, classifier F1'i korumasına rağmen `z_long` uzayında belirgin symbol-heavy hale geliyor.
- Bu failure classifier seviyesinde görünmüyor; latent geometry guardrail ile yakalanıyor.

### `z_fused`

| Variant | label agreement | symbol agreement | top1 label | top1 symbol |
|---|---:|---:|---:|---:|
| `ce_supcon_long_v1` | 0.8425 | 0.2784 | 0.8488 | 0.3051 |
| `seed_41` | 0.8359 | 0.4225 | 0.8394 | 0.5302 |
| `seed_42` | 0.8374 | 0.2766 | 0.8370 | 0.3011 |
| `seed_43` | 0.8392 | 0.3934 | 0.8434 | 0.4808 |

`z_fused` tarafında da aynı desen var: `seed_42` stabil, `seed_41` ve `seed_43` daha symbol-bound.

### `z_long_proj`

| Variant | label agreement | symbol agreement | top1 label | top1 symbol |
|---|---:|---:|---:|---:|
| `ce_supcon_long_v1` | 0.8067 | 0.2621 | 0.7985 | 0.2749 |
| `seed_41` | 0.7500 | 0.6505 | 0.7518 | 0.7385 |
| `seed_42` | 0.7988 | 0.2531 | 0.8037 | 0.2533 |
| `seed_43` | 0.7830 | 0.5517 | 0.7814 | 0.6255 |

`z_long_proj` asıl SupCon hedefi olduğu için bu tablo kritik. `seed_41` ve `seed_43`, projection geometry tarafında da referans davranışı koruyamıyor.

---

## Branch Usage Diagnostics

| Variant | `z_long/z_short` norm | effective long/short | `z_long` symbol/label centroid | `z_long_proj` symbol/label centroid |
|---|---:|---:|---:|---:|
| `ce_supcon_long_v1` | 0.6424 | 0.5467 | 0.0230 | 0.0192 |
| `seed_41` | 0.0536 | 0.0333 | 1.5365 | 0.0550 |
| `seed_42` | 0.5092 | 0.4606 | 0.0420 | 0.0430 |
| `seed_43` | 0.1136 | 0.1022 | 0.3711 | 0.0592 |

Okuma:

- `seed_42`, long branch'i aktif tutuyor.
- `seed_41` açık long-branch collapse: norm ratio ve effective contribution CE-only benzeri seviyeye düşüyor.
- `seed_43` tam collapse değil ama effective long contribution belirgin düşük ve symbol/label centroid ratio yüksek.

Bu, önceki diagnosis sonucunu güçlendiriyor:

> `ce_supcon_long_v1` başarısı sadece SupCon objective varlığından değil, long branch'in fused training içinde aktif kalabilmesinden geliyor.

---

## Bucket Geometry

Seçilmiş test split cosine centroid distances:

| Variant | `z_long` clean vs borderline up | `z_long` borderline up vs bullish confirmed | `z_long_proj` clean vs borderline up | `z_long_proj` borderline up vs bullish confirmed |
|---|---:|---:|---:|---:|
| `ce_supcon_long_v1` | 0.1216 | 0.4199 | 0.0323 | 0.0623 |
| `seed_41` | 0.0295 | 0.0161 | 0.0347 | 0.0202 |
| `seed_42` | 0.0950 | 0.1872 | 0.0289 | 0.0728 |
| `seed_43` | 0.2041 | 0.1219 | 0.0444 | 0.0446 |

Okuma:

- Referans koşuda borderline up ile bullish confirmed arasında `z_long` mesafesi belirgin.
- `seed_41` bu ayrımı neredeyse tamamen kaybediyor.
- `seed_42` ayrımı kısmen koruyor.
- `seed_43` clean/borderline ayrımını büyütse de borderline/confirmed mesafesi referansa göre zayıf.

Bucket geometry de seed stability'nin zayıf olduğunu doğruluyor.

---

## Best vs Final Checkpoint Kontrolü

`seed_41` ve `seed_43` için SupCon loss geç aktive olduğu için şu soru ayrıca kontrol edildi:

> Final checkpoint'te geometry toparlanıyor mu, yoksa failure kalıcı mı?

### Test embedding geometry

| Variant | Checkpoint | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol |
|---|---|---:|---:|---:|---:|
| `seed_41` | best | 0.7604 | 0.8082 | 0.7500 | 0.6505 |
| `seed_41` | final | 0.7992 | 0.5569 | 0.7974 | 0.3578 |
| `seed_43` | best | 0.7934 | 0.7049 | 0.7830 | 0.5517 |
| `seed_43` | final | 0.8013 | 0.6866 | 0.7945 | 0.5059 |

Final checkpoint, özellikle `seed_41` için projection geometry'yi belirgin toparlıyor. Ancak symbol agreement hâlâ referans ve `seed_42` seviyesine inmiyor.

### Branch usage

| Variant | Checkpoint | `z_long/z_short` norm | effective long/short | `z_long` symbol/label centroid | `z_long_proj` symbol/label centroid |
|---|---|---:|---:|---:|---:|
| `seed_41` | best | 0.0536 | 0.0333 | 1.5365 | 0.0550 |
| `seed_41` | final | 0.1251 | 0.1249 | 0.1177 | 0.0383 |
| `seed_43` | best | 0.1136 | 0.1022 | 0.3711 | 0.0592 |
| `seed_43` | final | 0.2088 | 0.2451 | 0.1942 | 0.0395 |

Okuma:

- Bu sadece checkpoint selection problemi değil.
- Final checkpoint, geç gelen SupCon etkisiyle geometry'yi kısmen toparlıyor.
- Fakat long branch contribution hâlâ referanstan düşük:
  - reference effective long/short: `0.5467`
  - `seed_42`: `0.4606`
  - `seed_41 final`: `0.1249`
  - `seed_43 final`: `0.2451`

Bu nedenle failure'ın ana nedeni “best checkpoint yanlış seçildi” değil; SupCon'un geç aktive olması ve classifier/branch dynamics'in erken fazda kısa yola yerleşmesi.

---

## Seed 43 Extended Run Kontrolü

`seed_43` için ek bir kontrol yapıldı:

- `epochs=40`
- `patience=100`
- aynı seed ve aynı objective
- checkpoint dir: `ce_supcon_long_v1_seed_43_e40`
- karar için `final.pt` test edildi

Amaç:

> SupCon'a daha uzun süre verilirse geç başlayan geometry toparlanıyor mu?

### SupCon trend

| Variant | Epochs | train SupCon @ epoch 27 | train SupCon final | val SupCon @ epoch 27 | val SupCon final |
|---|---:|---:|---:|---:|---:|
| `seed_43_e40` | 40 | 2.9789 | 2.9584 | 2.9887 | 2.9822 |

SupCon loss epoch 27 sonrasında da düşmeye devam ediyor. Düşüş büyük değil, ama temsil metriklerine yansıyor.

### Test embedding geometry

| Variant | Checkpoint | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol |
|---|---|---:|---:|---:|---:|
| `seed_43` | best | 0.7934 | 0.7049 | 0.7830 | 0.5517 |
| `seed_43` | final @ 27 | 0.8013 | 0.6866 | 0.7945 | 0.5059 |
| `seed_43_e40` | final @ 40 | 0.8018 | 0.5245 | 0.7989 | 0.3798 |

Okuma:

- Label agreement neredeyse aynı kalıyor.
- Symbol agreement belirgin düşüyor.
- En büyük kazanım `z_long_proj` tarafında: 0.5059 -> 0.3798.

### Branch usage

| Variant | Checkpoint | `z_long/z_short` norm | effective long/short | `z_long` symbol/label centroid | `z_long_proj` symbol/label centroid |
|---|---|---:|---:|---:|---:|
| `seed_43` | best | 0.1136 | 0.1022 | 0.3711 | 0.0592 |
| `seed_43` | final @ 27 | 0.2088 | 0.2451 | 0.1942 | 0.0395 |
| `seed_43_e40` | final @ 40 | 0.2858 | 0.3752 | 0.1069 | 0.0326 |

Bu sonuç önceki teşhisi rafine ediyor:

- Geç aktive olan SupCon, yeterli süre verilirse geometry'yi kısmen değil, belirgin biçimde toparlayabiliyor.
- Ancak `seed_43_e40` hâlâ referans / `seed_42` seviyesinde değil:
  - reference effective long/short: `0.5467`
  - `seed_42`: `0.4606`
  - `seed_43_e40`: `0.3752`
  - reference `z_long` symbol agreement: `0.2987`
  - `seed_43_e40`: `0.5245`

Dolayısıyla problem yalnızca early stopping veya best checkpoint selection değil. Ama daha uzun SupCon training, kötü attractor'ı kısmen terk ettirebiliyor.

Yeni hipotez:

> İyi geometry için SupCon'un erken aktive olması en güçlü yol; geç aktive olursa daha uzun eğitimle kısmi toparlanma mümkün, fakat branch usage dengesi referans seviyesine tam dönemeyebilir.

Bu da sonraki çözüm yönünü değiştiriyor:

- sadece checkpoint selection değil
- sadece daha uzun epoch da değil
- SupCon'un erken fazda etkili olmasını sağlayan schedule / warmup / branch usage constraint daha olası çözüm

---

## Decision

`ce_supcon_long_v1` setup'ı mevcut haliyle seed-stable kabul edilmemeli.

Plan dokümanındaki karar kuralına göre en az iki yeni seed'in core geometry'yi koruması gerekiyordu. Sonuç:

- stable: `seed_42`
- unstable / rejected geometry: `seed_41`, `seed_43`

Bu yüzden:

- `ce_supcon_long_v1`, best observed checkpoint olarak kalabilir
- fakat Stage 1B veya Stage 2 varsayımlarını bu tek checkpoint'e dayandırmak riskli
- yeni model/objective tasarımında ilk hedef pressure/maturity kazanımı değil, branch usage stability olmalı

---

## Next Direction

Kısa vadede yeni auxiliary head denemelerine dönmeden önce şu seçenekler daha anlamlı:

1. Long branch contribution guardrail'i eğitim objective veya early selection kriterine dahil etmek.
2. Best checkpoint seçiminde sadece `val_loss` yerine branch/geometry proxy metric kullanmak.
3. SupCon batch composition veya positive mining'i seed sensitivity azaltacak şekilde yeniden tasarlamak.
4. `seed_42` ve referans koşuyu yan yana inceleyip stable koşuların ortak branch dynamics özelliğini çıkarmak.

Bu sonuç, `ce_supcon_long_v1`in gösterdiği edge'in hayali olmadığını ama henüz güvenilir biçimde üretilemediğini gösteriyor.
