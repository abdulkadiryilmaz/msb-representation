# Experiment Plan: 2026-05-16 — CE + SupCon Long V1 Seed Stability

**Status**: Completed  
**Family**: `CE + SupCon`  
**Baseline to validate**: `ce_supcon_long_v1`  
**Question**: `ce_supcon_long_v1` başarısı seed-stable mı?

**Execution note**: İlk `seed_41` denemesi 2026-05-16'da local CPU üzerinde başlatıldı, ancak epoch süresi yaklaşık 15.5 dakika olduğu için 2 epoch sonrasında durduruldu. Sonrasında seed sweep MPS üzerinde yeniden koşuldu ve `seed_41`, `seed_42`, `seed_43` tamamlandı.

---

## Motivation

`ce_supcon_long_v1`, şu ana kadar Stage 1A için en iyi latent geometry dengesini verdi:

- `z_long` label agreement yüksek
- `z_long` symbol agreement düşük
- `z_fused` genel classifier performansını koruyor
- branch-aware SupCon, long branch'i aktif tutuyor

Ancak mevcut checkpoint seed kaydı taşımıyor. Bu yüzden açık kalan soru:

> Bu geometry kazanımı tek koşuya / seed'e özgü bir sonuç mu, yoksa aynı setup farklı seed'lerde de tekrar ediyor mu?

Bu deney yeni objective denemesi değildir. Mevcut current-best setup'ın stabilitesini ölçen kontrol deneyidir.

---

## Hypothesis

Eğer `ce_supcon_long_v1` gerçek bir representation kazanımı yakalıyorsa, farklı seed'lerde de aşağıdaki invariant'lar korunmalı:

- `z_long` NN label agreement yüksek kalmalı
- `z_long` NN symbol agreement düşük kalmalı
- `z_fused` test geometry kabul edilebilir aralıkta kalmalı
- `z_long_norm / z_short_norm` collapse etmemeli
- effective classifier `long/short` contribution CE-only seviyesine düşmemeli
- `z_long_proj` SupCon loss anlamlı biçimde düşmeli

Başarısızlık sinyali:

- classifier F1 makul kalsa bile `z_long` veya `z_long_proj` symbol-heavy hale gelirse seed stability yoktur.

---

## Runs

Yeni seed'ler:

| Variant | Seed | Checkpoint dir |
|---|---:|---|
| `ce_supcon_long_v1_seed_41` | 41 | `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1_seed_41` |
| `ce_supcon_long_v1_seed_42` | 42 | `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1_seed_42` |
| `ce_supcon_long_v1_seed_43` | 43 | `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1_seed_43` |

Existing `ce_supcon_long_v1` checkpoint will remain the reference run, but it should not be treated as a known-seed member because its metadata did not record the seed.

---

## Training Command Template

```bash
python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 32 \
  --use-supcon \
  --supcon-weight 0.05 \
  --supcon-temperature 0.1 \
  --supcon-embedding-key z_long_proj \
  --seed <seed> \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1_seed_<seed>
```

---

## Post-Run Checklist

For each seed:

1. Export val latents
2. Run val latent analysis
3. Run val embedding compare against `ce_supcon_long_v1`
4. Export test latents
5. Run test latent analysis
6. Run test embedding compare against `ce_supcon_long_v1`
7. Run branch usage diagnostics
8. Run bucket geometry diagnostics

Required diagnostic commands:

```bash
python scripts/diagnose_stage1a_branch_usage.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1_seed_<seed>/analysis/test_latents.npz \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1_seed_<seed>

python scripts/diagnose_stage1a_bucket_geometry.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1_seed_<seed>/analysis/test_latents.npz
```

---

## Decision Rule

Accept `ce_supcon_long_v1` as seed-stable if at least two of three new seed runs preserve the core geometry:

- `z_long` test label agreement remains close to reference
- `z_long` test symbol agreement stays materially below symbol-heavy rejected runs
- branch usage diagnostics do not show long-branch collapse
- bucket geometry does not collapse borderline examples into noisy / symbol-driven structure

If only one seed preserves this behavior, keep `ce_supcon_long_v1` as best observed checkpoint but mark the setup seed-sensitive.

If none preserve it, do not build Stage 1B assumptions on this representation without first stabilizing the objective.
