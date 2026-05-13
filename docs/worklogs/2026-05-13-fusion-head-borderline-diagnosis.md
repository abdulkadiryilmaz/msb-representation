# Worklog: 2026-05-13 — Fusion / Head Borderline Diagnosis

**Status**: Complete  
**Scope**: `ce_supcon_long_v1`, `z_fused`, borderline intact diagnosis

---

## Why This Diagnosis Exists

`z_long` vs `z_fused` structural review sonrası ana soru şuydu:

```text
Borderline intact hataları head boundary problemi mi,
yoksa z_fused latent geometry zaten confirmed break ailesine mi gömüyor?
```

Bu ayrım yeni deney yönü için kritik:

- latent ayrıştırıyor ama head yanlışsa → calibration / CE weighting / head-level müdahale yeterli olabilir
- latent de confirmed break ailesine gömüyorsa → representation objective veya factor-aware auxiliary hedef gerekir

Bu diagnosis yeni training veya probe çalıştırmaz. Mevcut `test_latents.npz` üstünden:

- classifier probability / margin
- correct vs misclassified borderline subset
- `z_fused` nearest-neighbor family dağılımı

okur.

---

## Artifact

Script:

- `scripts/diagnose_stage1a_fusion_head.py`

Output:

- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/fusion_head_diagnosis/z_fused_test_borderline/`

Files:

- `summary.json`
- `borderline_samples.csv`
- `head_group_summary.csv`
- `neighbor_family_summary.csv`

Command:

```bash
PYTHONPATH=src python scripts/diagnose_stage1a_fusion_head.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/test_latents.npz \
  --embedding-key z_fused \
  --top-k 10 \
  --output-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/fusion_head_diagnosis/z_fused_test_borderline
```

---

## Population

Subset:

```text
confirmed_state = intact
break_maturity = borderline
```

Summary:

| Metric | Value |
|---|---:|
| count | `731` |
| misclassified | `206` |
| misclassified rate | `0.2818` |
| avg confidence | `0.8033` |
| avg prob margin | `0.6117` |
| avg z_fused NN intact rate | `0.6847` |
| avg z_fused NN confirmed rate | `0.3153` |
| avg z_fused NN same-symbol rate | `0.2814` |

---

## Correct vs Misclassified

### Correct Borderline Intact

Count:

- `525`

Averages:

| Metric | Value |
|---|---:|
| confidence | `0.8329` |
| prob margin | `0.6718` |
| `p_intact` | `0.8329` |
| `p_bullish` | `0.0682` |
| `p_bearish` | `0.0989` |
| NN intact rate | `0.8339` |
| NN confirmed rate | `0.1661` |
| NN same-direction rate | `0.4577` |
| NN same-symbol rate | `0.2815` |

Correct örneklerde `z_fused` komşuluğu çoğunlukla intact-family tarafında kalıyor. Head de bunu `intact` olarak okuyor.

Top neighbor family:

- `intact / neutral / clean / none / none`: `281 / 525 = 0.5352`
- intact wick / pressure / borderline families toplamda belirgin pay taşıyor
- confirmed break families sınırlı kalıyor

### Misclassified Borderline Intact

Count:

- `206`

Averages:

| Metric | Value |
|---|---:|
| confidence | `0.7280` |
| prob margin | `0.4586` |
| `p_intact` | `0.2695` |
| `p_bullish` | `0.3262` |
| `p_bearish` | `0.4043` |
| NN intact rate | `0.3044` |
| NN confirmed rate | `0.6956` |
| NN same-direction rate | `0.8238` |
| NN same-symbol rate | `0.2811` |

Misclassified örneklerde tablo farklı:

- head confidence correct örneklere göre daha düşük
- probability margin daha düşük
- ama `p_intact` de oldukça düşük
- `z_fused` NN geometry çoğunlukla confirmed break ailesine kaymış
- komşuların yönü query pressure direction ile çoğunlukla aynı
- same-symbol oranı correct örneklerle neredeyse aynı

Bu nedenle ana problem symbol shortcut gibi görünmüyor.

---

## Head Group Readout

Down-pressure misclassified:

| Holding | Count | Pred | Confidence | `p_intact` | NN confirmed rate | NN same-direction |
|---|---:|---|---:|---:|---:|---:|
| reverted | `78` | bearish | `0.7158` | `0.2829` | `0.7103` | `0.9295` |
| holding | `30` | bearish | `0.7186` | `0.2809` | `0.7100` | `0.9333` |

Up-pressure misclassified:

| Holding | Count | Pred | Confidence | `p_intact` | NN confirmed rate | NN same-direction |
|---|---:|---|---:|---:|---:|---:|
| reverted | `60` | bullish | `0.7333` | `0.2661` | `0.6400` | `0.9067` |
| holding | `17` | bullish | `0.7608` | `0.2376` | `0.7118` | `0.8706` |

Correct examples:

- down-pressure reverted correct: NN confirmed rate `0.1899`
- down-pressure holding correct: NN confirmed rate `0.1364`
- up-pressure reverted correct: NN confirmed rate `0.1617`
- up-pressure holding correct: NN confirmed rate `0.1468`

The separation is large:

```text
correct borderline:      NN confirmed rate ~0.14-0.19
misclassified borderline: NN confirmed rate ~0.64-0.71
```

This suggests the problem is not only the classifier head. The misclassified subset is already geometrically close to confirmed break examples in `z_fused`.

---

## Interpretation

This diagnosis points to:

```text
z_fused geometry and classifier head agree on the hard errors.
```

Correct borderline examples:

- latent neighbors mostly intact-family
- head predicts intact with high confidence

Misclassified borderline examples:

- latent neighbors mostly confirmed-family
- head predicts the same directional confirmed class
- confidence is moderate, not extreme

So this is not a clean "latent is fine, head is bad" case.

More precise diagnosis:

```text
ce_supcon_long_v1 preserves useful pressure direction,
but for a subset of borderline intact examples, z_fused itself places them in the confirmed directional family.
The classifier follows that geometry.
```

This matches the manual examples:

- `1567`: `intact / down_pressure / borderline / reverted`, predicted bearish
- nearest neighbors are same-direction confirmed bearish examples

### Bimodal Borderline Population

The average borderline intact numbers hide a sharper split.

This subset is not uniformly weak. It behaves like two populations:

1. Correct borderline intact examples:
   - count: `525`
   - NN intact rate: `0.8339`
   - NN confirmed rate: `0.1661`
   - head predicts `intact`

2. Misclassified borderline intact examples:
   - count: `206`
   - NN intact rate: `0.3044`
   - NN confirmed rate: `0.6956`
   - head predicts same-direction `bullish` / `bearish`

So the problem is not:

```text
borderline intact representation is generally weak
```

The sharper diagnosis is:

```text
a specific borderline intact subpopulation is embedded inside the same-direction confirmed break family
```

This matters for the next experiment. We should not blindly push all borderline intact examples away from directionality, because most are already correctly placed near intact-family structure.

The target subset is narrower:

```text
confirmed_state = intact
break_maturity = borderline
pressure_state = up/down
NN confirmed rate high
pred = same-direction confirmed class
```

### Calibration Ceiling

The misclassified confidence is moderate:

- avg confidence: `0.7280`
- avg probability margin: `0.4586`

This leaves room for calibration or abstention-style decision rules.

However, because the same examples are already close to confirmed break families in `z_fused`, calibration alone has a limited ceiling. In other words:

```text
representation düzelmeden calibration, core geometry problemini çözmez
```

Calibration can be a secondary diagnostic or downstream thresholding tool, but it should not be treated as the primary fix unless the goal is simply to abstain on uncertain borderline cases.

### Open Question — CE-only Baseline mı, SupCon Yan Etkisi mi?

This diagnosis was run on `ce_supcon_long_v1`.

It does not yet answer whether the 206-example confirmed-family drift is:

1. already present in `CE-only`
2. amplified by branch-aware SupCon
3. partly caused by the state label design itself

This is the next diagnosis to run.

If `CE-only` shows the same drift:

```text
the issue is likely inherent to the current state supervision / feature geometry
```

If `ce_supcon_long_v1` increases the drift:

```text
SupCon may preserve pressure direction but over-pull borderline pressure toward confirmed break clusters
```

That distinction changes the next experiment design.

---

## Implication for Next Experiment

Head-only calibration may reduce some borderline mistakes, because confidence is not extreme:

- misclassified avg confidence: `0.7280`
- misclassified avg margin: `0.4586`

However, head-only calibration probably will not solve the core geometry issue because:

- misclassified examples have `z_fused` NN confirmed rate `0.6956`
- correct examples have `z_fused` NN confirmed rate `0.1661`

Therefore the next experiment should likely be representation-aware, but scoped:

Target:

```text
preserve pressure direction,
separate near-confirmation intact from confirmed_state,
avoid increasing symbol shortcut.
```

Most relevant next directions:

1. Factor-aware auxiliary objective
   - main target: `confirmed_state`
   - auxiliary analysis/training target: `pressure_state`
   - optional target: `break_maturity`

2. Borderline-aware contrastive or weighting
   - treat `borderline intact pressure` as near but not identical to confirmed break
   - avoid pushing all same-direction pressure directly into confirmed state clusters

3. Calibration as secondary check
   - useful for thresholding
   - not enough as primary solution unless downstream decision can tolerate abstention / uncertainty

---

## Decision

This diagnosis shifts the next experiment decision away from pure head-level fixes.

Recommended next planning step:

```text
design a small factor-aware Stage 1A objective that keeps CE confirmed_state,
adds pressure/break-maturity awareness,
and evaluates both z_long pressure preservation and z_fused confirmed-state boundary.
```
