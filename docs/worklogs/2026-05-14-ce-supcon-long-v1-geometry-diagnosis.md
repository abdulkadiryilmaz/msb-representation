# Worklog: 2026-05-14 — CE + SupCon Long V1 Geometry Diagnosis

**Status**: Diagnosis complete  
**Reference checkpoint**: `ce_supcon_long_v1`  
**Compared against**: `ce_only`, `ce_supcon_long_aux_v2`, `ce_supcon_long_short_pressure_aux_v1`

---

## Soru

`ce_supcon_long_v1` neden `z_long` geometry'yi koruyabiliyor?

Auxiliary deneyleri pressure sinyalini öğrenebildi, fakat `z_long` / `z_long_proj` geometry'yi bozdu. Bu yüzden asıl soru pressure objective'den çok branch dynamics tarafına kaydı:

> `long_v1`te long branch nasıl aktif ve semantik kalıyor, aux koşularında neden tekrar shortcut veya branch collapse davranışına dönüyor?

---

## Ana Bulgular

### 1. `long_v1`, long branch'i gerçek anlamda aktive ediyor

Test latent norm ve feature std:

| Variant | `z_short` norm | `z_long` norm | `z_long / z_short` | `z_long` feature std |
|---|---:|---:|---:|---:|
| `ce_only` | 22.155 | 2.578 | 0.116 | 0.341 |
| `ce_supcon_long_v1` | 12.056 | 7.744 | 0.642 | 0.995 |
| `ce_supcon_long_aux_v2` | 17.269 | 3.253 | 0.188 | 0.471 |
| `ce_supcon_long_short_pressure_aux_v1` | 19.793 | 2.094 | 0.106 | 0.224 |

Okuma:

- CE-only'de `z_long` fiilen küçük ölçekli / zayıf kullanılan branch.
- `long_v1`, `z_long`u hem büyütüyor hem varyansını artırıyor.
- Aux koşuları bu aktivasyonu koruyamıyor; özellikle short-pressure aux koşusu CE-only benzeri long-branch collapse'a dönüyor.

Bu, `long_v1`in başarısının sadece projection uzayında değil, encoder output seviyesinde de olduğunu gösteriyor.

---

### 2. Classifier etkili katkısı `long_v1`te dengeleniyor

Classifier ilk katmanında short/long weight normları tek başına yeterli değil; latent norm ile birlikte okunmalı.

Yaklaşık etkili katkı:

```text
effective_long/short =
  (classifier_long_weight_norm * z_long_norm)
  /
  (classifier_short_weight_norm * z_short_norm)
```

| Variant | classifier long/short | latent long/short | effective long/short |
|---|---:|---:|---:|
| `ce_only` | 0.595 | 0.116 | 0.069 |
| `ce_supcon_long_v1` | 0.851 | 0.642 | 0.547 |
| `ce_supcon_long_aux_v2` | 0.841 | 0.188 | 0.158 |
| `ce_supcon_long_short_pressure_aux_v1` | 0.539 | 0.106 | 0.057 |

Okuma:

- CE-only'de fused classifier neredeyse tamamen `z_short` üzerinden çalışıyor.
- `long_v1`, long branch'in classifier'a etkili katkısını ciddi biçimde artırıyor.
- `aux_v2` classifier weight oranını koruyor gibi görünse de latent long/short oranı düştüğü için efektif katkı düşüyor.
- short-pressure aux koşusunda hem classifier weight oranı hem latent oranı düşüyor; long branch tekrar devre dışı kalıyor.

Bu bulgu, `long_v1`in `z_long` geometry'yi korumasında **branch usage balance** etkisinin kritik olduğunu gösteriyor.

---

### 3. SupCon loss davranışı ayırt edici

History özeti:

| Variant | train SupCon first | train SupCon best epoch | train SupCon last |
|---|---:|---:|---:|
| `ce_supcon_long_v1` | 4.1442 | 3.6642 | 3.6346 |
| `ce_supcon_long_aux_v2` | 3.4353 | 3.0203 | 2.9740 |
| `ce_supcon_long_short_pressure_aux_v1` | 3.4350 | 3.4340 | 3.4338 |

Not: `long_v1` eski trainer ile çalıştığı için absolute SupCon loss seviyesi yeni koşularla birebir karşılaştırılmamalı. Fakat trend anlamlı:

- `long_v1`: SupCon loss belirgin düşüyor.
- `aux_v2`: SupCon loss düşüyor ama symbol shortcut yine artıyor.
- short-pressure aux: SupCon loss neredeyse hiç düşmüyor; `z_long_proj` geometry kurulamıyor.

Bu, short-pressure izolasyon koşusunda failure'ın sadece classifier tarafında değil, doğrudan SupCon/projection learning tarafında da olduğunu gösteriyor.

---

### 4. Centroid ayrışması `long_v1`te label-dominant

Cosine-normalized centroid ayrışması, test split:

| Variant | Embedding | between label | between symbol | symbol/label |
|---|---|---:|---:|---:|
| `ce_supcon_long_v1` | `z_long` | 0.7145 | 0.0177 | 0.02 |
| `ce_supcon_long_aux_v2` | `z_long` | 0.5429 | 0.1992 | 0.37 |
| `ce_supcon_long_short_pressure_aux_v1` | `z_long` | 0.0386 | 0.4836 | 12.54 |
| `ce_supcon_long_v1` | `z_long_proj` | 0.3744 | 0.0076 | 0.02 |
| `ce_supcon_long_aux_v2` | `z_long_proj` | 0.3543 | 0.0257 | 0.07 |
| `ce_supcon_long_short_pressure_aux_v1` | `z_long_proj` | 0.0016 | 0.0066 | 4.23 |

Okuma:

- `long_v1`: `z_long` ve `z_long_proj` label centroid ayrışmasını taşırken symbol centroid ayrışmasını bastırıyor.
- `aux_v2`: label ayrışması kısmen duruyor ama symbol ayrışması büyüyor.
- short-pressure aux: `z_long` label ayrışması neredeyse yok oluyor; symbol ayrışması baskın hale geliyor.

Bu da `long_v1`in başarısının “symbol shortcut azalması” değil, **label-dominant / symbol-suppressed geometry** olduğunu netleştiriyor.

---

## Tanı

`ce_supcon_long_v1`in koruduğu şey:

```text
branch-aware SupCon
  -> z_long aktivasyonunu büyütüyor
  -> z_long_proj label-dominant manifold kuruyor
  -> fused classifier long branch'i gerçekten kullanıyor
  -> symbol centroid ayrışması baskılanıyor
```

Auxiliary koşularda bozulma iki şekilde geliyor:

1. `aux_v2`: pressure sinyali okunabilir hale geliyor ama `z_long` aktivasyonu küçülüyor ve symbol ayrışması büyüyor.
2. short-pressure aux: pressure short branch'e izole ediliyor ama fused model short branch'e daha fazla yaslanıyor; `z_long` etkili katkısı CE-only seviyesine dönüyor ve SupCon projection geometry kurulamıyor.

Bu yüzden önceki hipotez eksikti:

> Sorun yalnızca pressure gradient'in `z_long`a doğrudan akması değil. Sorun, auxiliary supervision'ın fused training dynamics içinde branch usage balance'ı bozması.

---

## Sonuç

Yeni auxiliary objective tasarlamadan önce korunması gereken invariant şu olmalı:

```text
z_long effective contribution must not collapse
```

Sadece NN label/symbol agreement'e bakmak geç kalmış bir ölçüm. Yeni deneylerde training sonrası en az şu guardrail'ler raporlanmalı:

- `z_long_norm / z_short_norm`
- classifier effective `long/short` contribution
- `z_long_proj` SupCon loss trend
- `z_long` centroid `symbol/label` ratio

Bu guardrail'ler bozulursa pressure probe iyileşse bile deney reddedilmeli.

---

## Next Direction

Kısa vadede yeni pressure/maturity auxiliary denemesi açmak yerine iki tanı adımı daha yüksek bilgi değerine sahip:

1. `ce_supcon_long_v1`i aynı ayarlarla seed kontrollü tekrar koşmak.
   - Amaç: `long_v1` başarısının stabil mi, yoksa seed-sensitive mı olduğunu görmek.

2. Branch usage diagnostics script'ini standart readout aracına çevirmek.
   - Her yeni koşuda norm ratio, effective classifier ratio, centroid ratio ve SupCon trend otomatik raporlanmalı.

Eğer yeni model tasarımı yapılacaksa, hedef pressure kazanımından önce branch usage balance'ı korumak olmalı.

---

## 2026-05-15 Follow-up: Standard Diagnostics

Bu tanıdan çıkan iki kontrol standart post-run araca çevrildi.

### 1. Branch usage diagnostics

Yeni script:

```text
scripts/diagnose_stage1a_branch_usage.py
```

Raporlanan ana guardrail'ler:

- `z_long_norm / z_short_norm`
- classifier first-layer `long/short` weight ratio
- effective classifier `long/short` contribution
- `z_long` / `z_long_proj` / `z_fused` centroid `symbol/label` ratio
- SupCon loss trend

Amaç, yeni bir deneyin `z_long` branch'ini gerçekten kullanıp kullanmadığını readout öncesinde otomatik görmek.

### 2. Bucket geometry diagnostics

Yeni script:

```text
scripts/diagnose_stage1a_bucket_geometry.py
```

Bu script, notebook'ta manuel yapılan bucket centroid distance kontrollerini standartlaştırır.

Raporlanan çiftler:

- `clean_intact` vs `borderline_up/down`
- `borderline_up/down` vs confirmed bullish/bearish break
- `wick_sweep_up/down` vs confirmed bullish/bearish break
- `borderline_*_holding/reverted` vs confirmed break

Bu kontrolün amacı, aggregate `borderline_intact_break` başarısını tek başına okumak yerine borderline örneklerin latent uzayda clean intact'a mı, yoksa confirmed break tarafına mı yaklaştığını ölçmek.

### Workflow update

`docs/EXPERIMENT_WORKFLOW.md` post-run checklist'e iki adım eklendi:

```text
Run branch usage diagnostics
Run bucket geometry diagnostics
```

Böylece seed tekrarından veya yeni objective denemelerinden sonra bu iki guardrail readout'un zorunlu parçası olacak.

---

## 2026-05-16 Follow-up: Frontload ve Checkpoint Selection

Seed stability okumasından sonra `seed_43` üzerinde `CE + SupCon long frontload v1` denendi.

Amaç, SupCon gradient'inin erken fazda etkili olmasını sağlamak ve classifier'ın önce short-branch shortcut'a yerleşmesini zorlaştırmaktı.

Uygulanan schedule:

```text
epoch 1-8:
  loss = CE + 0.10 * SupCon

epoch 9+:
  loss = CE + 0.05 * SupCon
```

### Ana bulgu

Frontload, SupCon'u gerçekten daha erken aktive etti:

| Run | SupCon loss ilk belirgin düşüş |
|---|---|
| `ce_supcon_long_v1_seed_43` | yaklaşık epoch 12-13 |
| `ce_supcon_long_frontload_v1_seed_43_e70` | yaklaşık epoch 4-5 |

Bu, problemin en azından kısmen erken training dynamics ile ilgili olduğunu doğruluyor.

### Epoch sweep sonucu

`seed_43` frontload koşusu 70 epoch'a uzatıldı ve `epoch_005/010/020/030/040/050/060/070` checkpoint'leri önce test split üzerinde retrospektif olarak incelendi. Daha sonra aynı checkpoint'ler validation split üzerinde de değerlendirildi; gerçek seçim kuralı validation geometry olmalı.

Validation sweep:

| Epoch | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.7796 | 0.5335 | 0.7715 | 0.3434 | 0.8202 | 0.2960 | 0.0692 |
| 10 | 0.7949 | 0.4391 | 0.7874 | 0.2739 | 0.8265 | 0.3031 | 0.2159 |
| 20 | 0.7955 | 0.3521 | 0.7947 | 0.2717 | 0.8397 | 0.3101 | 0.3465 |
| 30 | 0.8005 | 0.3407 | 0.7988 | 0.2703 | 0.8317 | 0.3191 | 0.4692 |
| 40 | 0.8023 | 0.3442 | 0.8015 | 0.2728 | 0.8356 | 0.3219 | 0.5130 |
| 50 | 0.8032 | 0.3408 | 0.8007 | 0.2713 | 0.8344 | 0.3117 | 0.5852 |
| 60 | 0.8065 | 0.3668 | 0.8044 | 0.2799 | 0.8335 | 0.3145 | 0.6187 |
| 70 | 0.8001 | 0.3817 | 0.7994 | 0.2894 | 0.8351 | 0.3171 | 0.6283 |

Validation okuması:

- Epoch 40 ve 50 birbirine yakın.
- Validation protokolüyle seçilecek checkpoint `epoch_050.pt` olur: `z_long` label agreement epoch 40'tan biraz daha yüksek, symbol agreement biraz daha düşük, `z_fused` symbol daha düşük ve effective long/short daha sağlıklı.
- Epoch 60 label agreement'i daha yüksek olsa da symbol agreement artışı nedeniyle daha zayıf aday.
- Epoch 70 effective long/short'u artırıyor ama symbol agreement bozuluyor.

Retrospektif test sweep özeti:

| Epoch | val F1 | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.8717 | 0.7776 | 0.4770 | 0.7727 | 0.3247 | 0.0695 |
| 10 | 0.8778 | 0.7908 | 0.4054 | 0.7897 | 0.2773 | 0.2159 |
| 20 | 0.8893 | 0.7998 | 0.3334 | 0.7985 | 0.2746 | 0.3475 |
| 30 | 0.8748 | 0.7999 | 0.3336 | 0.8000 | 0.2743 | 0.4732 |
| 40 | 0.8827 | 0.8051 | 0.3158 | 0.8024 | 0.2613 | 0.5250 |
| 50 | 0.8823 | 0.8011 | 0.3225 | 0.8019 | 0.2682 | 0.5909 |
| 60 | 0.8756 | 0.8025 | 0.3385 | 0.7998 | 0.2725 | 0.6238 |
| 70 | 0.8803 | 0.8005 | 0.3489 | 0.7970 | 0.2733 | 0.6360 |

Test sweep'te epoch 40 en iyi dengeyi verdi:

- `z_long` label agreement en yüksek noktaya çıktı.
- `z_long_proj` symbol agreement referans `ce_supcon_long_v1` seviyesine çok yaklaştı.
- effective long/short katkısı referans koşuya yaklaştı.
- 50-70 aralığında effective long/short yükselmeye devam etti, fakat symbol agreement da kötüleşti.

Bu yüzden effective long/short tek başına daha yüksek diye daha iyi sayılmamalı. Long branch'in aktif olması gerekli ama yeterli değil; long branch'in symbol-dominant hale gelmemesi gerekiyor.

### Checkpoint selection kararı

Bu bulgu, `best.pt` seçiminin Stage 1A representation hedefi için yetersiz olduğunu gösteriyor.

Mevcut trainer `best.pt` dosyasını val loss'a göre seçiyor. Ancak frontload koşusunda:

- best val F1 yaklaşık epoch 20'de,
- best val loss yaklaşık epoch 23'te,
- en iyi latent geometry dengesi yaklaşık epoch 40'ta görüldü.

Bu nedenle Stage 1A için yeni seçim kuralı:

```text
Training fixed-budget çalışır.
Ara checkpoint'ler düzenli kaydedilir.
Representation checkpoint'i post-run val geometry diagnostics ile seçilir.
Test split sadece seçilmiş checkpoint'in final readout'u için kullanılır.
```

Not: Bu follow-up'taki ilk epoch sweep test split üzerinde retrospektif tanı olarak yapıldı. Validation sweep eklendikten sonra seçim protokolünün neden validation tarafında olması gerektiği daha netleşti: test retrospektifi epoch 40'ı, validation selection ise epoch 50'yi işaret ediyor. Bundan sonraki gerçek checkpoint selection test split üzerinden yapılmamalı; selection val geometry üzerinde, final raporlama test split üzerinde yapılmalı.

### Yeni invariant

Yeni deneylerde korunması gereken daha güçlü invariant:

```text
z_long effective contribution must rise
AND
z_long / z_long_proj symbol agreement must remain suppressed
```

Bu nedenle tek başına `val_loss`, `val_macro_f1`, `train_supcon_loss` veya `effective_long_short_ratio` karar metriği olmamalı.

### Multi-seed follow-up

Aynı frontload schedule daha sonra `seed_41` ve `seed_42` için 50 epoch koşuldu.

Validation-selected checkpoint sonuçları:

| Run | Selected checkpoint | Split | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `frontload_seed41` | `epoch_040.pt` | val | 0.8024 | 0.4888 | 0.7980 | 0.3395 | 0.8423 | 0.3412 | 0.2893 |
| `frontload_seed41` | `epoch_040.pt` | test | 0.8068 | 0.4860 | 0.8002 | 0.3217 | 0.8413 | 0.3054 | 0.2863 |
| `frontload_seed42` | `epoch_040.pt` | val | 0.8107 | 0.3062 | 0.8086 | 0.2690 | 0.8390 | 0.2962 | 0.6414 |
| `frontload_seed42` | `epoch_040.pt` | test | 0.8089 | 0.3001 | 0.8067 | 0.2607 | 0.8361 | 0.2861 | 0.6489 |
| `frontload_seed43` | `epoch_050.pt` | val | 0.8032 | 0.3408 | 0.8007 | 0.2713 | 0.8344 | 0.3117 | 0.5852 |
| `frontload_seed43` | `epoch_050.pt` | test | 0.8011 | 0.3225 | 0.8019 | 0.2682 | 0.8365 | 0.3009 | 0.5909 |

Okuma:

- `seed_42`: frontload altında çok güçlü kaldı; iyi seed bozulmadı.
- `seed_43`: frontload ile büyük ölçüde toparlandı.
- `seed_41`: label geometry toparlandı ama `z_long` symbol agreement hâlâ yüksek kaldı.

Bu yüzden frontload için karar:

```text
useful, but not seed-stable enough
```

Frontload erken SupCon aktivasyonunu iyileştiriyor, fakat bazı seed'lerde symbol-heavy attractor hâlâ mümkün. Bundan sonraki tasarım CE-delay / warm-start veya explicit branch usage regularization tarafına bakmalı.

---

## 2026-05-17 Follow-up: SupCon-only Warmup

Frontload sonucundan sonra `seed_41` failure mode'una daha doğrudan müdahale etmek için hard SupCon-only warmup denendi.

Schedule:

```text
epoch 1-5:
  loss = 0.00 * CE + 0.10 * SupCon

epoch 6+:
  loss = 1.00 * CE + 0.05 * SupCon
```

### Training dynamics

SupCon-only warmup iki seed'de de erken SupCon platosunu kırdı.

| Run | epoch 1 train SupCon | epoch 5 train SupCon | epoch 10 train SupCon | epoch 50 train SupCon |
|---|---:|---:|---:|---:|
| `warmup_seed41` | 3.1653 | 2.9739 | 2.9101 | 2.7873 |
| `warmup_seed42` | 3.2156 | 2.9763 | 2.9167 | 2.7932 |

Bu, `seed_41` failure'ının önemli bir kısmının erken SupCon aktivasyonuyla ilgili olduğunu doğruluyor.

### Selected checkpoint test readout

| Run | Checkpoint | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `warmup_seed41` | `epoch_050.pt` | 0.5174 | 0.7091 | 0.8380 | 0.2701 | 0.8382 | 0.2562 | 0.8438 | 0.5384 | 7.8831 |
| `warmup_seed42` | `epoch_040.pt` | 0.6137 | 0.5776 | 0.8364 | 0.2771 | 0.8358 | 0.2592 | 0.8397 | 0.4446 | 5.4168 |

### Diagnosis

Hard SupCon-only warmup solves one problem and creates another:

- `z_long` / `z_long_proj` geometry becomes very strong.
- `seed_41` symbol-heavy `z_long` failure disappears.
- `z_short` geometry degrades sharply.
- `z_fused` becomes more symbol-heavy.
- effective long/short becomes excessively large.

Bu yüzden karar:

```text
SupCon-only warmup = diagnostic success, not final candidate
```

Asıl öğrenim:

```text
Early SupCon activation matters,
but CE=0 for 5 epochs is too aggressive for balanced fused representation learning.
```

### Next direction

Hard warmup yerine SupCon-dominant warmup denenmeli:

```text
epoch 1-5:
  loss = 0.10 * CE + 0.10 * SupCon

epoch 6+:
  loss = 1.00 * CE + 0.05 * SupCon
```

Bu tasarımın hedefi:

- SupCon platosunu erken kırmak
- `z_long` symbol-heavy attractor'ı engellemek
- `z_short` / `z_fused` dengesini hard warmup'a göre daha iyi korumak

---

## 2026-05-17 Follow-up: SupCon-dominant Warmup Seed 41

Hard SupCon-only warmup fazla agresif olduğu için ara schedule denendi:

```text
epoch 1-5:
  loss = 0.10 * CE + 0.10 * SupCon

epoch 6+:
  loss = 1.00 * CE + 0.05 * SupCon
```

Seed 41 training dynamics:

| Epoch | CE weight | SupCon weight | train CE | train SupCon | val F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.10 | 0.10 | 0.4417 | 3.1707 | 0.8360 |
| 5 | 0.10 | 0.10 | 0.3365 | 2.9716 | 0.8713 |
| 10 | 1.00 | 0.05 | 0.2932 | 2.9494 | 0.8813 |
| 20 | 1.00 | 0.05 | 0.2496 | 2.9127 | 0.8859 |
| 30 | 1.00 | 0.05 | 0.2250 | 2.8914 | 0.8863 |
| 40 | 1.00 | 0.05 | 0.2072 | 2.8762 | 0.8870 |
| 50 | 1.00 | 0.05 | 0.1948 | 2.8650 | 0.8819 |

Seed 41 validation sweep:

| Epoch | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.8052 | 0.3353 | 0.8062 | 0.4409 | 0.8050 | 0.2808 | 0.8254 | 0.3335 | 0.4353 |
| 10 | 0.7992 | 0.3175 | 0.8132 | 0.2914 | 0.8095 | 0.2544 | 0.8395 | 0.3273 | 0.5427 |
| 20 | 0.7788 | 0.3383 | 0.8210 | 0.3204 | 0.8166 | 0.2852 | 0.8435 | 0.3359 | 0.6788 |
| 30 | 0.7772 | 0.3296 | 0.8211 | 0.3506 | 0.8193 | 0.2804 | 0.8444 | 0.3287 | 0.7854 |
| 40 | 0.7624 | 0.3031 | 0.8198 | 0.3759 | 0.8160 | 0.3023 | 0.8443 | 0.3271 | 0.9059 |
| 50 | 0.7551 | 0.3205 | 0.8204 | 0.3024 | 0.8192 | 0.2693 | 0.8396 | 0.3272 | 0.8714 |

Validation-selected checkpoint: `epoch_010.pt`.

Test readout:

| Embedding | label agree | symbol agree |
|---|---:|---:|
| `z_short` | 0.7964 | 0.3034 |
| `z_long` | 0.8102 | 0.2826 |
| `z_fused` | 0.8361 | 0.3012 |
| `z_long_proj` | 0.8084 | 0.2542 |

Branch usage:

| Metric | Value |
|---|---:|
| `z_long / z_short` norm ratio | 0.6016 |
| classifier long/short weight ratio | 0.9085 |
| effective long/short ratio | 0.5465 |
| `z_long` centroid symbol/label ratio | 0.0196 |
| `z_long_proj` centroid symbol/label ratio | 0.0179 |

Ara karar:

```text
seed 41 için en iyi dengeli aday: SupCon-dominant warmup v1, epoch_010.pt
```

Ancak en iyi checkpoint'in epoch 10'da gelmesi önemli bir uyarı:

- warmup iyi başlangıç kuruyor
- post-warmup schedule bu dengeyi uzun süre korumuyor olabilir
- seed 42 ile erken optimum davranışı tekrar ediyor mu kontrol edilmeli

Eğer seed 42 de aynı deseni gösterirse, sıradaki tasarım staged post-warmup olmalı:

```text
epoch 1-5:
  0.10 * CE + 0.10 * SupCon

epoch 6-15:
  0.50 * CE + 0.075 * SupCon

epoch 16+:
  1.00 * CE + 0.05 * SupCon
```

### Seed 42 result

Seed 42 aynı schedule ile koşuldu. Validation-selected checkpoint `epoch_040.pt`.

Seed 42 test readout:

| Embedding | label agree | symbol agree |
|---|---:|---:|
| `z_short` | 0.7711 | 0.2771 |
| `z_long` | 0.8158 | 0.2787 |
| `z_fused` | 0.8422 | 0.2839 |
| `z_long_proj` | 0.8143 | 0.2548 |

Branch usage:

| Metric | Value |
|---|---:|
| `z_long / z_short` norm ratio | 0.9151 |
| classifier long/short weight ratio | 0.9384 |
| effective long/short ratio | 0.8587 |
| `z_long` centroid symbol/label ratio | 0.0236 |
| `z_long_proj` centroid symbol/label ratio | 0.0212 |

Multi-seed selected checkpoints:

| Run | Split | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dominant_seed41_e10` | test | 0.7964 | 0.3034 | 0.8102 | 0.2826 | 0.8084 | 0.2542 | 0.8361 | 0.3012 | 0.5465 |
| `dominant_seed42_e40` | test | 0.7711 | 0.2771 | 0.8158 | 0.2787 | 0.8143 | 0.2548 | 0.8422 | 0.2839 | 0.8587 |

Son karar:

```text
SupCon-dominant warmup v1 = best balanced candidate so far
```

Bu schedule:

- frontload seed 41'deki `z_long` symbol-heavy failure'ı düzeltti
- hard warmup'taki `z_short` / `z_fused` collapse'ı üretmedi
- seed 42'yi bozmadı
- effective long/short'u sağlıklı seviyeye taşıdı

Ancak asıl öğrenim schedule'dan daha genel:

```text
z_long activity / usefulness objective'de doğrudan yok.
Schedule bunu teşvik ediyor ama garanti etmiyor.
```

Bu yüzden sonraki deney hattı schedule tuning yerine objective-level constraint olmalı:

```text
fused CE
+ SupCon(z_long_proj)
+ auxiliary CE head on z_long
```
