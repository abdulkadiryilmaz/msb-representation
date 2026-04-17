# Worklog: 2026-04-14 — CE + SupCon Experiment Plan for Stage 1A

**Status**: Proposed ✓  
**Goal**: `CE-only` baseline bulgularını kullanarak ilk `CE + SupCon` deneyini açık ve uygulanabilir biçimde tanımlamak

---

## Neden Şimdi

`CE-only` koşusu sonrası artık üç kritik şey netleşti:

- classifier performansı zaten güçlü
- `z_short` yapısal state semantiğini iyi taşıyor
- `z_long` ve kısmen `z_fused`, symbol/context bilgisine fazla yaslanıyor

Bucket tarafında da sorun alanı netleşti:

- `true_break_*` örnekleri büyük ölçüde temiz
- asıl açık problem `borderline_intact_break`
- ikinci açık problem `high_vol_intact_wick_sweep` ve genel `wick_sweep_*` alanı

Dolayısıyla `SupCon` için gerekçe artık genel accuracy artırmak değil.

Asıl amaç:

- latent komşuluğunu daha semantik hale getirmek
- symbol shortcut etkisini azaltmak
- sınır örneklerinde geometry'yi iyileştirmek

## Experiment Question

İlk `CE + SupCon` koşusu şu soruya cevap vermeli:

- `projection head` üstünde contrastive supervision eklemek, `z_fused` ve mümkünse `z_long` tarafında label-aware komşuluğu artırıyor mu?

Alt soru:

- bu iyileşme, classifier metriğini bozmadan geliyor mu?

## Baseline To Beat

Referans çizgi:

- objective: `CE-only`
- dataset: `core4`
- short window: `48`
- long window: `288`
- `z_short = 64`
- `z_long = 32`
- fusion: `concat`
- projection head: mevcut

Bu baseline'da:

- val macro-F1 güçlü
- `z_short` NN label agreement yüksek
- `z_long` NN symbol agreement çok yüksek

Bu yüzden başarı ölçütü yalnızca macro-F1 değil.

## Proposed Objective

İlk deneme:

- ana loss = `CE + lambda_supcon * SupCon(z_proj, y)`

Burada:

- `CE`, classifier head üstünde kalır
- `SupCon`, `z_proj` üstünde çalışır
- encoder'lar ve fusion katmanı iki loss'u birlikte görür

Neden `z_proj`:

- mevcut mimaride projection head zaten var
- contrastive objective'i ana classifier uzayına doğrudan bindirmeden eklemek daha güvenli
- ilk iterasyon için en düşük mimari risk bu

## First Hyperparameter Proposal

İlk sweep çok dar tutulmalı.

Önerilen başlangıç:

- `temperature = 0.1`
- `lambda_supcon = 0.05`
- ikinci küçük varyant: `lambda_supcon = 0.1`

Sebep:

- `CE-only` zaten güçlü
- agresif contrastive ağırlık classifier çözümünü bozabilir
- ilk amaç geometry'yi hafifçe itmek, eğitimi yeniden tanımlamak değil

## Success Criteria

Başarı kriteri çok eksenli olmalı.

### 1. Classifier tarafı

Korunması gerekenler:

- val macro-F1, `CE-only` baseline'dan belirgin kötü olmamalı
- confusion yapısı büyük regression göstermemeli

Pratik eşik:

- macro-F1 düşüşü `~0.01` üzerindeyse şüpheli say

### 2. Embedding tarafı

Asıl hedefler:

- `z_fused` NN label agreement artmalı
- `z_fused` top-1 label match artmalı
- `z_fused` symbol agreement düşmeli veya en azından artmamalı
- `z_long` label agreement anlamlı biçimde iyileşmeli

Özellikle şu desen aranmalı:

- `z_short` korunuyor
- `z_long` daha semantik oluyor
- `z_fused` short kadar label-aware kalıp symbol bias'tan biraz uzaklaşıyor

### 3. Bucket tarafı

Asıl izlenecek bucket'lar:

- `borderline_intact_break`
- `high_vol_intact_wick_sweep`
- `wick_sweep_up`
- `wick_sweep_down`
- `close_confirmed_break_up`
- `close_confirmed_break_down`

Pozitif sinyal:

- bu bucket'larda NN label agreement artışı
- `borderline_intact_break` hata oranında düşüş
- `high_vol_intact_wick_sweep` ve `wick_sweep_*` bucket'larında daha temiz komşuluk

## Minimum Implementation Plan

1. Trainer config'e contrastive parametreleri ekle

- `use_supcon`
- `supcon_weight`
- `supcon_temperature`

2. `SupConLoss` yardımcı modülü ekle

- batch içi supervised contrastive loss
- label aynıysa positive pair

3. Eğitim loop'unu güncelle

- `loss = ce_loss + supcon_weight * supcon_loss`
- history içine `train_ce_loss`, `train_supcon_loss`, `val_ce_loss` benzeri alanlar ekle

4. Ayrı checkpoint klasörü kullan

- örnek: `checkpoints/ce_supcon_v1`

5. Aynı latent analysis hattını çalıştır

- export
- bucket readout
- embedding compare

## Evaluation Contract

İlk `CE + SupCon` koşusu sonunda mutlaka şu artefaktlar üretilmeli:

- training history
- val confusion
- latent export
- bucket full distribution
- embedding overall summary
- embedding bucket comparison

Karşılaştırma mutlaka şu eksende yapılmalı:

- `CE-only` vs `CE + SupCon`

yalnızca accuracy değil.

## Decision Rule

İlk `CE + SupCon` koşusu için basit karar kuralı:

- eğer macro-F1 benzer kalır ve `z_long` / `z_fused` label agreement iyileşirse: devam et
- eğer macro-F1 bozulur ve geometry kazanımı zayıf kalırsa: weight düşür veya positive tanımını değiştir
- eğer sadece symbol agreement artarsa: bu deney başarısız sayılmalı

## Open Risks

- batch içi positive diversity yetersiz olabilir
- symbol imbalance, contrastive objective'i yanlış yöne çekebilir
- `SupCon`, zaten temiz olan `z_short` uzayını bozabilir
- `intact` sınıfının daha büyük olması positive/negative dengesini etkileyebilir

Bu yüzden ilk iterasyon küçük ve kontrollü tutulmalı.

## Recommended Next Command Set

İmplementasyon sonrası hedef akış:

```bash
python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 32 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_v1 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1
```

Sonrasında:

```bash
PYTHONPATH=src python scripts/export_stage1a_latents.py \
  --dataset-root data/stage1a/binance/15m \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_v1 \
  --split val

PYTHONPATH=src python scripts/analyze_stage1a_latents.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_v1/analysis/val_latents.npz

PYTHONPATH=src python scripts/compare_stage1a_embedding_views.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_v1/analysis/val_latents.npz
```

## Conclusion

`CE + SupCon` için artık gerekçe nettir:

- hedef classifier'ı kurtarmak değil
- latent geometry'yi özellikle `z_long` ve `z_fused` tarafında daha semantik yapmak
- `borderline_intact_break` ve failed-break bucket'larında ayrımı iyileştirmek

Bu yüzden ilk `CE + SupCon` koşusu küçük, kontrollü ve analiz-heavy bir deney olarak ele alınmalı.

---

**Status**: Proposed ✓
