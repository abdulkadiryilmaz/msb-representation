# CE + SupCon Long Branch CE Aux v1 Plan

**Date**: 2026-05-17  
**Status**: Planned  
**Family**: `CE + SupCon`  
**Variant**: `ce_supcon_long_branch_ce_aux_v1`

## Motivation

Son schedule deneyleri şunu gösterdi:

- `z_long` doğru attractor'a düşebiliyor
- fakat bu davranış seed ve epoch seçimine duyarlı kalıyor
- schedule, `z_long` aktivitesini teşvik ediyor ama garanti etmiyor

Bu deneyde hedef, `z_long`un label-dominant bilgi taşımasını dolaylı bir yan etki olmaktan çıkarıp objective'e doğrudan eklemek.

## Change

Mevcut objective:

```text
fused CE
+ SupCon(z_long_proj)
```

Yeni objective:

```text
fused CE
+ SupCon(z_long_proj)
+ long_aux_weight * CE(long_aux_head(z_long), label)
```

`long_aux_head` yalnızca `z_long` üzerinden sınıflandırma yapar. Böylece ana fused classifier `z_short`a yaslansa bile `z_long` label semantiğini taşımaya devam etmek zorunda kalır.

## Initial Configuration

İlk koşu schedule içermeden yapılacak:

```text
ce_weight = 1.00
supcon_weight = 0.05
long_aux_loss_weight = 0.10
supcon_embedding_key = z_long_proj
epochs = 50
patience = 100
epoch_checkpoints = 5 10 20 30 40 50
seeds = 41, 42
```

Rationale:

- `0.10` auxiliary CE, `z_long` için doğrudan ama ana CE'yi bastırmayacak kadar sınırlı bir corrective force sağlar
- warmup kullanılmayarak objective değişikliğinin tek başına yeterli olup olmadığı test edilir
- checkpoint seçimi val split üzerinde geometry-first yapılır

## Decision Metrics

Primary:

- `z_long` mean NN label agreement
- `z_long` mean NN symbol agreement
- `z_fused` mean NN label/symbol agreement
- branch usage diagnostics

Guardrails:

- `z_long` symbol agreement `0.35` civarında veya altında kalmalı
- `z_fused` symbol agreement `0.33` civarında veya altında kalmalı
- effective long/short ratio sağlıklı aralıkta kalmalı; tek başına yüksek olması başarı sayılmamalı
- seçilen checkpoint yalnızca test split'e bakılarak seçilmemeli

## Candidate Commands

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --patience 100 \
  --batch-size 32 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --long-aux-loss-weight 0.10 \
  --seed 41 \
  --epoch-checkpoints 5 10 20 30 40 50 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50
```

```bash
PYTHONPATH=src python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --patience 100 \
  --batch-size 32 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --long-aux-loss-weight 0.10 \
  --seed 42 \
  --epoch-checkpoints 5 10 20 30 40 50 \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_42_e50
```

## Expected Readout

Bu deney başarılı sayılmak için dominant warmup kadar iyi veya daha iyi geometry vermek zorunda değil; öncelikle daha stabil bir mekanizma göstermeli:

```text
z_long active + label-dominant kalıyor mu?
Bu davranış seed değişiminde schedule'a göre daha az kırılgan mı?
```
