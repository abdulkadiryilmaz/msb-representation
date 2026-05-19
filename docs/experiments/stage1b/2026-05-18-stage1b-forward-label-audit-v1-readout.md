# Stage 1B Forward Label Audit v1 Readout

**Date**: 2026-05-18
**Status**: completed
**Scope**: distribution and horizon-consistency audit for `future_break_direction_H` labels
**Artifact**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_label_audit_h16_h32_h48.json`

## Goal

Stage 1B predictor sonuçlarında görülen `break` / `none` bulanıklığının label üretiminden kaynaklanıp kaynaklanmadığını ilk seviyede kontrol etmek.

Bu audit iki şeyi test eder:

1. Label implementation mantıksal olarak tutarlı mı?
2. Label distribution predictor için sağlıklı mı, yoksa break-heavy bir prior mı üretiyor?

Bu çalışma görsel/manual label audit değildir. OHLC plot üzerinden örnek doğrulama sonraki adımdır.

## Checks

Kontrol edilenler:

- H16 / H32 / H48 label dağılımları
- H16 -> H32 cross-tab consistency
- `H16 break`, `H32 none` gibi mantıksal olarak şüpheli geçişler
- `H16 bullish`, `H32 bearish` veya tersi yön flip durumları
- `H16 none`, `H32 bullish/bearish` delayed segment support
- time-to-break dağılımı

## Distribution

### Train

| Horizon | none | bullish | bearish | ambiguous | break rate |
|---|---:|---:|---:|---:|---:|
| H16 | 41,967 | 45,457 | 43,608 | 8 | 0.6797 |
| H32 | 24,263 | 54,622 | 52,145 | 10 | 0.8148 |
| H48 | 15,446 | 59,058 | 56,526 | 10 | 0.8821 |

### Val

| Horizon | none | bullish | bearish | ambiguous | break rate |
|---|---:|---:|---:|---:|---:|
| H16 | 2,587 | 3,100 | 3,143 | 2 | 0.7071 |
| H32 | 1,470 | 3,640 | 3,720 | 2 | 0.8336 |
| H48 | 945 | 3,897 | 3,988 | 2 | 0.8930 |

### Test

| Horizon | none | bullish | bearish | ambiguous | insufficient_future | break rate |
|---|---:|---:|---:|---:|---:|---:|
| H16 | 3,025 | 3,115 | 3,547 | 1 | 16 | 0.6871 |
| H32 | 1,706 | 3,689 | 4,276 | 1 | 32 | 0.8210 |
| H48 | 1,082 | 3,940 | 4,633 | 1 | 48 | 0.8835 |

Main observation:

`future_break_direction_H` is strongly break-heavy, especially for H32/H48. This is expected as horizon grows, but it also makes `none` a harder and smaller class.

## H16 -> H32 Consistency

Test split cross-tab:

Rows: H16
Columns: H32

```text
H16 none        -> H32 none=1706, bullish=588, bearish=729, insufficient_future=2
H16 bullish     -> H32 bullish=3101, insufficient_future=14
H16 bearish     -> H32 bearish=3547
H16 ambiguous   -> H32 ambiguous=1
H16 insufficient -> H32 insufficient_future=16
```

Suspicious transitions:

| Split | H16 break -> H32 none | Rate | H16 direction flip in H32 | Rate |
|---|---:|---:|---:|---:|
| train | 0 | 0.0000 | 0 | 0.0000 |
| val | 0 | 0.0000 | 0 | 0.0000 |
| test | 0 | 0.0000 | 0 | 0.0000 |

Implementation-level consistency looks clean:

- If a break exists in H16, it remains present in H32.
- H16 bullish does not become H32 bearish.
- H16 bearish does not become H32 bullish.
- No impossible `short break / long none` transitions appear.

## Delayed Segment

Definition:

```text
H16 = none
H32 in {bullish, bearish}
```

| Split | Support | Rate | Bullish | Bearish |
|---|---:|---:|---:|---:|
| train | 17,702 | 0.1351 | 9,165 | 8,537 |
| val | 1,117 | 0.1265 | 540 | 577 |
| test | 1,317 | 0.1357 | 588 | 729 |

The delayed segment is real and stable across splits. Its support is enough for targeted modeling, but the v1b predictor did not identify it cleanly.

## Time-To-Break

Test split:

| Horizon | support | min | q25 | median | q75 | max |
|---|---:|---:|---:|---:|---:|---:|
| H16 | 6,663 | 2 | 2 | 2 | 7 | 16 |
| H32 | 7,966 | 2 | 2 | 4 | 12 | 32 |
| H48 | 8,574 | 2 | 2 | 4 | 15 | 48 |

Important observation:

For H16, median time-to-break is only 2 bars. A large portion of `break` labels are very near-term confirmations. This can make the target closer to immediate level proximity / near-touch behavior than a broader structure evolution target.

## Interpretation

No obvious implementation bug appears in the horizon labels.

The label contract is internally consistent, but it creates a break-heavy target:

- H16 test break rate: `0.6871`
- H32 test break rate: `0.8210`
- H48 test break rate: `0.8835`

This explains part of the predictor behavior:

- `break` is the majority class.
- `none` is the harder minority class.
- Raising the break threshold improves `none` recall but costs true break recall.
- H32 naturally becomes a broad "break likely within horizon" signal, not a clean delayed setup signal.

The weak `H16=none, H32=break` precision is therefore not surprising. H32 is too break-heavy to act as a clean early setup signal by itself.

## Decision

Label implementation is acceptable for Stage 1B v1 experiments.

However, this label alone is not enough for TradePlan-oriented setup quality. Before using H32 as early setup signal, we need either:

1. direct delayed setup target:

```text
delayed_break_16_32 = H16 none and H32 bullish/bearish
```

2. break-quality / maturity labels:

```text
time_to_break bucket
break_strength
close_count / persistence
distance_to_level
```

3. visual/manual audit of false positives and false negatives to distinguish label noise from genuinely hard structure.

## Next Step

Run a visual/manual audit on sampled examples:

- true `none`, predicted `break`
- true `break`, predicted `none`
- true `H16 none`, `H32 break`
- high-confidence predicted delayed segment

The goal is to decide whether the current break labels are structurally meaningful enough, or whether Stage 1B should move toward richer targets before another predictor iteration.

## Artifacts

```text
scripts/audit_stage1b_forward_labels.py
data/stage1a/binance/15m/checkpoints/ce_supcon_long_branch_ce_aux_v1_seed_41_e50/analysis/stage1b_forward_label_audit_h16_h32_h48.json
```
