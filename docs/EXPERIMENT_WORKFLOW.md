# Experiment Workflow

This document defines the standard working process for Stage 1A experiments.

Goals:
- standardize plan, run, analysis, and documentation steps
- reduce context loss across sessions
- make required artifacts explicit

## 1. When to Open a New Experiment

Open a new experiment plan when:

- a new objective variant is being tested
- a new masking, sampling, or batch strategy is introduced within the same objective
- a different checkpoint family is created
- the success criteria or baseline to beat changes

Minor repeats within the same experiment chain can update the existing plan.

## 2. Where Experiment Documents Live

Experiment plans and readout notes:

- `docs/experiments/<family>/`

Examples:

- `docs/experiments/ce_only/`
- `docs/experiments/ce_supcon/`

Decision and implementation notes:

- `docs/worklogs/`

## 3. Naming Convention

Plan:

```text
YYYY-MM-DD-<variant>-plan.md
```

Readout:

```text
YYYY-MM-DD-<variant>-readout.md
```

Examples:

- `2026-04-17-ce-supcon-long-symbol-v2-plan.md`
- `2026-04-17-ce-supcon-long-symbol-v2-readout.md`

## 4. Pre-Run Checklist

Before every new run:

1. Is the baseline to beat clearly identified?
2. Is the experiment hypothesis written in one sentence?
3. Is exactly one thing changing relative to the baseline?
4. Is the checkpoint name unambiguous?
5. Does the plan note include:
   - method
   - command
   - success criteria
   - planned readout
6. Is this run marked as `in_progress` in `docs/EXPERIMENT_INDEX.md`?

## 5. During Training

Minimum information to record while the run is active:

- training device
- early stopping epoch
- best val macro F1
- any notable training behavior

These notes will be summarized in the readout.

## 6. Required Post-Run Artifacts

After a run completes, the minimum required artifacts are:

1. Training summary
   - `metadata.json`
   - `history.json`

2. Val artifacts
   - latent export
   - analyze
   - embedding compare

3. Test artifacts
   - latent export
   - analyze
   - embedding compare

4. Notebook or manual inspection if needed

## 7. Standard Post-Run Checklist

Default order after a run finishes:

1. Read training summary
2. Export val latents
3. Run val analyze
4. Run val embedding compare
5. Export test latents
6. Run test analyze
7. Run test embedding compare
8. Compare side-by-side with baseline to beat
9. Write readout document
10. Update `EXPERIMENT_INDEX.md`:
    - Add row to Current Summary table
    - Add numerical values to all Metrics Snapshot tables (see Section 10)
11. Update family README timeline if needed

## 8. Standard Comparison Axes

For every readout, cover the following axes:

### Classifier

- best val macro F1
- test generalization signal

### Embedding

- `z_short`
- `z_long`
- `z_fused`
- branch-specific projection if present (e.g. `z_long_proj`)

Key metrics to track:

- mean NN label agreement
- mean NN symbol agreement
- top-1 label match
- top-1 symbol match

### Bucket

Priority buckets:

- `borderline_intact_break`
- `wick_sweep_up`
- `wick_sweep_down`
- `high_vol_intact_wick_sweep`
- `close_confirmed_break_up`
- `close_confirmed_break_down`

## 9. Readout Writing Rules

A readout note must include:

- experiment setup
- classifier summary
- val readout
- test readout
- bucket readout
- overall decision
- next direction

Critical rule:

- decisions are never made on macro F1 alone
- latent geometry is the primary decision axis

## 10. When to Update the Experiment Index

The index is updated when:

- a new run starts: add or update an `in_progress` row
- a readout is complete: set status to one of:
  - `reference`
  - `best current`
  - `mixed`
  - `rejected`

### Metrics Snapshot Tables

`EXPERIMENT_INDEX.md` maintains four metric tables:

- Classifier — Val Macro F1
- Latent Geometry — `z_long` NN metrics (val + test)
- Latent Geometry — `z_fused` NN metrics (val + test)
- Pressure Probe — `intact` subset linear F1 (test)

Adding a new row to each table after every completed readout is **required**.

Values to add:

- val macro F1
- `z_long` / `z_fused` label agreement and symbol agreement (val and test)
- pressure probe z_short / z_long / z_fused linear F1 if the probe was run

The pressure probe is not run for every experiment — it is a selective diagnostic tool. If run, add to the table; if not run, leave the row empty.

The current best candidate is always shown in bold (`**`) in all tables.

## 11. When to Update Family README

Update the family README when:

- the narrative of the experiment chain changes after a new readout
- a new variant is added to the timeline
- the "best current" variant changes

## 12. When a Worklog Is Required

Open a worklog when:

- a new objective, masking, or sampling mechanism is implemented
- the latent analysis infrastructure changes
- a design decision is made that changes the research direction

Summary:

- implementation / decision → `worklog`
- experiment plan / experiment result → `experiments`

## 13. Minimum Reading Order

To get up to speed quickly in a new session:

1. `docs/EXPERIMENT_INDEX.md`
2. relevant family README
3. active plan note
4. baseline to beat readout
5. most recent rejected and most recent accepted variant readouts
