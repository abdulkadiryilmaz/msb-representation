# Worklog: 2026-05-13 — Pressure Label Validation Readout

**Status**: Open  
**Scope**: Stage 1A analysis-only `pressure_labels_v1` / structural maturity factors  
**Purpose**: Model objective değişikliğinden önce pressure/maturity label'larının güvenilirliğini değerlendirmek

---

## Question

Yeni model denemesi için önerilen yön:

```text
pressure-aware but maturity-sensitive representation learning
```

Bu yön, `confirmed_state` ana label'ına ek olarak analysis-only factor'ları kullanmayı gündeme getiriyor:

- `pressure_state`
- `break_maturity`
- `structural_direction`
- `holding_status`

Bu yüzden model değişikliğinden önce cevaplanması gereken soru:

```text
Pressure/maturity label'ları objective'a girecek kadar güvenilir mi?
```

Buradaki önemli ayrım:

- `confirmed_state` ana Stage 1A label sözleşmesidir
- `pressure_state` ve `break_maturity` şu anda analysis/readout amaçlı rule-derived label'lardır
- bu label'lar henüz independent human-validated ground truth değildir

Bu nedenle ilk kullanımda bunları ana supervision gibi değil, düşük ağırlıklı auxiliary / geometry regularizer adayı gibi düşünmek gerekir.

---

## Artifacts

Pressure label artefact'ları:

- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/val_latents/pressure_labels_v1.parquet`
- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/test_latents/pressure_labels_v1.parquet`

Pressure probe readout:

- `docs/worklogs/2026-04-18-intact-pressure-probe-readout.md`

Structural probe readout:

- `docs/worklogs/2026-05-13-ce-only-vs-supcon-borderline-diagnosis.md`

Relevant scripts:

- `scripts/generate_stage1a_pressure_labels.py`
- `scripts/probe_stage1a_pressure.py`
- `scripts/probe_stage1a_structural_state.py`
- `scripts/analyze_stage1a_latents.py`

---

## Label Contract Under Review

Current `pressure_labels_v1` scope:

```text
base label = intact
```

Classes:

- `neutral`
- `up_pressure`
- `down_pressure`
- `non_intact` for non-intact base labels

Current structural factorization adds:

- `confirmed_state`
- `confirmed_direction`
- `pressure_state`
- `break_maturity`
- `structural_direction`
- `holding_status`

This label family should not be interpreted as future forecast.

It describes rule-based current-window structure:

```text
within the observed Stage 1A window,
is this intact example clean, under directional pressure, wick-swept, or borderline?
```

---

## Distribution Sanity

Test split intact pressure distribution:

| Pressure label | Count |
|---|---:|
| `neutral` | `3969` |
| `up_pressure` | `413` |
| `down_pressure` | `425` |
| total intact | `4807` |

Observation:

- pressure classes are minority classes
- `up_pressure` and `down_pressure` are balanced against each other
- `neutral` is dominant, so accuracy-based validation is misleading

This supports using:

- macro F1 for probes
- stratified manual review for label validation
- class-balanced or low-weight auxiliary objective if used in training

---

## Symbol Distribution Sanity

Test split intact pressure distribution by symbol:

| Symbol | Down | Neutral | Up |
|---|---:|---:|---:|
| BTC | `0.0823` | `0.8281` | `0.0895` |
| ETH | `0.0776` | `0.8280` | `0.0943` |
| SOL | `0.0911` | `0.8196` | `0.0893` |
| XRP | `0.1030` | `0.8266` | `0.0704` |

Readout:

- no obvious single-symbol pressure shortcut is visible
- class shares are not identical, but they are in a narrow band
- this does not prove semantic correctness, but it reduces the risk that `pressure_labels_v1` is mostly coin identity leakage

---

## Rule Consistency Checks

Structural factor consistency on `ce_supcon_long_v1` test split:

| Check | Count |
|---|---:|
| `up_pressure` but direction not `up/mixed` | `0` |
| `down_pressure` but direction not `down/mixed` | `0` |
| `neutral` with active structural direction | `0` |
| `borderline` without `up/down/mixed` direction | `0` |

Readout:

- current pressure label, structural direction, and maturity fields are internally consistent
- this validates the analysis contract mechanically
- it does not yet validate whether a human chart read agrees with the rule

---

## Maturity Distribution

Test split intact maturity distribution:

| Break maturity | Count |
|---|---:|
| `clean` | `2554` |
| `wick_sweep` | `939` |
| `borderline` | `731` |
| `high_vol_intact` | `313` |
| `high_vol_intact_wick_sweep` | `270` |

Borderline direction / holding breakdown:

| Direction / Holding | Count |
|---|---:|
| down / holding | `96` |
| down / reverted | `237` |
| mixed / mixed | `95` |
| up / holding | `94` |
| up / reverted | `209` |

Readout:

- borderline population is large enough to diagnose and train against
- reverted borderline examples are more common than holding examples
- this matters because `pressure_state` should not be read as "will break"; many pressure examples already reverted within the current window

---

## Probe Evidence

Pressure probe evidence:

| Variant | Embedding | Linear macro F1 |
|---|---|---:|
| `ce_supcon_long_v1` | `z_short` | `0.5257` |
| `ce_supcon_long_v1` | `z_long` | `0.4842` |
| `ce_supcon_long_v1` | `z_fused` | `0.5407` |
| majority baseline | n/a | `0.3015` |

Maturity binary probe evidence:

```text
borderline_intact vs confirmed_break
```

| Variant | Embedding | Linear macro F1 | MLP macro F1 |
|---|---|---:|---:|
| `ce_supcon_long_v1` | `z_long` | `0.6995` | `0.7134` |
| `ce_supcon_long_v1` | `z_fused` | `0.7158` | `0.7529` |
| majority baseline | n/a | `0.4653` | n/a |

Readout:

- pressure and maturity labels are not random/noisy enough to be unreadable
- existing latent already carries both signals above majority baseline
- probe success supports diagnostic usefulness
- probe success alone does not prove label correctness

---

## Current Trust Level

Current evidence supports:

```text
pressure_labels_v1 can be trusted as a diagnostic analysis label
```

Current evidence does not yet support:

```text
pressure_labels_v1 should be treated as high-confidence ground truth
```

Reason:

- distribution sanity passes
- symbol shortcut sanity passes at a coarse level
- rule consistency passes
- probe readability passes
- but manual semantic validation is still incomplete

---

## Required Manual Validation

Before using pressure/maturity labels as a training objective, run a stratified manual review:

| Group | Suggested samples |
|---|---:|
| `clean / neutral` | `10` |
| `wick_sweep_up` | `10` |
| `wick_sweep_down` | `10` |
| `borderline up / holding` | `10` |
| `borderline up / reverted` | `10` |
| `borderline down / holding` | `10` |
| `borderline down / reverted` | `10` |
| `mixed_pressure` or mixed borderline | `10` |

Manual review question:

```text
Does the visible price window support the assigned pressure/maturity label?
```

For each sample, record:

- accept
- questionable
- reject
- reason

Minimum acceptance bar before training usage:

- clear majority of reviewed labels should be `accept`
- rejected examples should reveal either edge-case noise or a specific rule bug
- if a systematic mismatch appears, create `pressure_labels_v2` before training

---

## Training Implication

Until manual validation is complete:

- do not make pressure/maturity the primary target
- do not give pressure/maturity auxiliary losses high weight
- use them as diagnostic labels and candidate low-weight regularizers

Safe first model direction:

```text
confirmed_state CE remains primary
pressure/maturity auxiliary losses are low-weight
success is judged on latent geometry and borderline behavior, not auxiliary accuracy alone
```

If manual validation fails:

```text
do not train with pressure_labels_v1
revise rules and create pressure_labels_v2
```

If manual validation passes:

```text
use pressure/maturity as factor-aware auxiliary supervision
but keep loss weights conservative in the first experiment
```

---

## Decision

Current decision:

```text
Do not block the next experiment forever,
but add one manual semantic validation pass before committing to a model run.
```

Recommended next step:

```text
Generate a stratified manual pressure/maturity review set from test split.
```

This review should reuse the existing manual NN/price-window plotting path, but the sampling criterion should be pressure/maturity strata rather than nearest-neighbor hard cases.

---

## Stratified Review Set

Generated artefact:

- `data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/manual_pressure_review/pressure_labels_v1_test_stratified/`

Script:

- `scripts/inspect_stage1a_pressure_review.py`

Command:

```bash
PYTHONPATH=src python scripts/inspect_stage1a_pressure_review.py \
  --dataset-root data/stage1a/binance/15m \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/test_latents.npz \
  --split test \
  --output-dir data/stage1a/binance/15m/checkpoints/ce_supcon_long_v1/analysis/manual_pressure_review/pressure_labels_v1_test_stratified \
  --per-stratum 10 \
  --seed 42
```

Selection summary:

| Stratum | Available | Selected |
|---|---:|---:|
| `clean_neutral` | `2554` | `10` |
| `wick_sweep_up` | `439` | `10` |
| `wick_sweep_down` | `451` | `10` |
| `borderline_up_holding` | `94` | `10` |
| `borderline_up_reverted` | `209` | `10` |
| `borderline_down_holding` | `96` | `10` |
| `borderline_down_reverted` | `237` | `10` |
| `mixed_borderline` | `95` | `10` |

Outputs:

- `review_notes.md`
- `candidates.csv`
- `80` short/long price-window PNG panels

`candidates.csv` includes empty manual review columns:

- `manual_decision`
- `manual_reason`

Use the same decision vocabulary:

- `accept`
- `questionable`
- `reject`

The next decision should be based on the manual acceptance pattern:

- if most strata are mostly `accept`, `pressure_labels_v1` can be used as low-weight auxiliary supervision
- if rejections cluster in one stratum, revise that rule and create `pressure_labels_v2`
- if rejections are broad, do not train with pressure/maturity labels yet

---

## Manual Spot Review Result

Manual spot review was performed over the stratified PNG panels.

Readout:

- `clean_neutral` examples looked semantically consistent
- `wick_sweep_up/down` examples generally matched the expected wick-without-close-confirmation pattern
- `borderline_*_holding` examples showed near-confirmation pressure that was still active near the anchor
- `borderline_*_reverted` examples showed pressure attempts that had moved back inside the relevant level by the anchor
- `mixed_borderline` examples looked appropriate as two-sided pressure / non-single-direction cases

Observed caveats:

- `holding_status` is threshold-sensitive; some `reverted` cases are visually close to the level and should be treated as lower-confidence than the core `pressure_state`
- some `mixed_borderline` rows have legacy bucket names ending in `_up` or `_down`, while factorized state is `mixed`; this is not a rule contradiction, but the factorized fields should be preferred for model design

Manual conclusion:

```text
pressure_labels_v1 is acceptable for a conservative auxiliary objective,
but should not be treated as primary high-confidence ground truth.
```

Training implication:

- use `pressure_state` and/or `break_maturity` with low loss weight
- keep `confirmed_state` CE as the primary target
- avoid making `holding_status` a primary auxiliary target in the first model experiment
- evaluate whether the intervention improves borderline intact vs confirmed break separation without suppressing useful pressure readability
