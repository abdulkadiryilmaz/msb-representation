## Overview
- Project: msb-representation
- Focus: multi-scale representation learning for Market Structure Break semantics
- Primary goal: structural latent learning, not live trading pipeline development
- Primary timeframe: 15m candles
- Core setup: short window `48`, long window `288`, shared-anchor dual-window dataset
- Default Stage 1A dataset universe: `core4` = BTC, ETH, SOL, XRP

## Language and Documentation
- Agent instruction files (`AGENTS.md`, `CLAUDE.md`, `EXPERIMENT_WORKFLOW.md`) are in English
- Research content (worklogs, experiment readouts, framing docs) is written in Turkish
- Technical terms may remain in English regardless of document language
- Research framing and conceptual foundation documents: `docs/foundations/`
- Experiment plans and readouts: `docs/experiments/`
- Architecture changes, data contract changes, and direction-setting decisions: `docs/worklogs/`
- Experiment workflow: `docs/EXPERIMENT_WORKFLOW.md`
- Top-level experiment summary: `docs/EXPERIMENT_INDEX.md`
- Worklog guide: `docs/WORKLOG_GUIDE.md`

## Research Framing
- Stage 1A is `MSB-oriented representation learning`
- `bullish / bearish / intact` labels supervise latent geometry
- classification is auxiliary; the main artifact is the latent representation
- success is measured with latent separability, probe quality, hard-case behavior, and downstream usefulness

## Architecture Summary
- Pipeline: Raw OHLCV -> FeatureBuilder -> Stage1A Dataset Builder -> Normalizer -> Dual-window Dataset -> Temporal CNN Encoders -> Fused Latent -> Aux Class Head / Projection Head
- Short branch: local structure, wick/body behavior, displacement, recent swing logic
- Long branch: broader structural context at the same anchor
- Fusion baseline: concat
- V1 objective: `CE only`
- Current best SupCon variant: branch-aware `CE + SupCon` on `z_long_proj`
- Current active experiment should be checked in `docs/EXPERIMENT_INDEX.md`

## Data Contract
- Raw OHLCV: `data/raw/<exchange>/<symbol>_<timeframe>.parquet`
- Stage 1A dataset root: `data/stage1a/<exchange>/<timeframe>/`
- Each Stage 1A symbol file stores:
  - `short_windows`
  - `long_windows`
  - `labels`
  - `timestamps`
- Dataset builds must use an explicit symbol universe:
  - either a named dataset profile
  - or an explicit allowlist
- Never silently train on “all available parquet files”
- Default profile for baseline experiments: `core4`
- If a different symbol universe is used, document why in the worklog and dataset spec

## Feature Principles
- Prefer price-structure-first features
- Shared-core short/long feature family is the default
- Do not add scale-specific extras before the shared-core baseline is stable
- If features change, update:
  - code
  - tests
  - docs
  - dataset spec assumptions

## Do Not
- Don’t mix live trading pipeline logic into this repo
- Don’t optimize for replay PnL before representation quality is understood
- Don’t couple Stage 1A labels to future trade outcome labels
- Don’t let coin identity shortcuts dominate the latent space
- Don’t bulk-import old pipeline code unless Stage 1A directly depends on it
- Don’t bring `signals/`, replay/backtest state logic, or production checkpoint conventions into this repo by default

## Repository Boundary
- This is a research repository
- `coin-oracle` remains the production / replay / signal execution side
- Before porting shared utilities, verify that Stage 1A actually depends on them
- Do not copy old modules on a "might be useful" basis

## Experiment Workflow
- Before starting a new experiment, follow the pre-run checklist in `docs/EXPERIMENT_WORKFLOW.md`
- After a run completes, apply the post-run checklist in full
- After each readout, add a new row to all four Metrics Snapshot tables in `docs/EXPERIMENT_INDEX.md`:
  - Classifier — Val Macro F1
  - `z_long` NN metrics (val + test: label agreement, symbol agreement)
  - `z_fused` NN metrics (val + test: label agreement, symbol agreement)
  - Pressure Probe result (if run)
- Always read the current best candidate from `docs/EXPERIMENT_INDEX.md`; the static reference in AGENTS.md may be stale

## Evaluation Principles
- Val macro F1 alone is not a decision criterion
- The primary decision axis is latent geometry: `mean_nn_label_agreement` and `mean_nn_symbol_agreement`
- Every readout must compare at minimum three representations: `z_short`, `z_long`, `z_fused`
- Accept/reject decisions are based primarily on `z_long` and `z_fused` NN metrics
- High symbol agreement (>0.5) indicates the latent is anchored to coin identity rather than structure; this alone can justify rejection

## Common Workflows
- Build dataset:
  - `python scripts/build_stage1a_dataset.py --data-root data --exchange binance --timeframe 15m --bars-per-day 96 --dataset-profile core4 --train-end 2025-09-30 --val-end 2025-12-31`
- Train CE-only baseline:
  - `python scripts/train_stage1a.py --dataset-root data/stage1a/binance/15m --epochs 50 --batch-size 32`
- Latent export (val split):
  - `python scripts/export_stage1a_latents.py --dataset-root data/stage1a/binance/15m --checkpoint-dir data/stage1a/binance/15m/checkpoints/<variant> --split val`
- Analyze latents:
  - `python scripts/analyze_stage1a_latents.py --latent-path <export.npz> --output-dir <checkpoint>/analysis/<split>_latents`
- Embedding compare (baseline to beat vs. current):
  - `python scripts/analyze_stage1a_latents.py --compare --baseline-latent <baseline.npz> --candidate-latent <candidate.npz>`

## Documentation
- `docs/foundations/stage1-msb-representation-framing.md`
- `docs/EXPERIMENT_INDEX.md`
- `docs/EXPERIMENT_WORKFLOW.md`
- `docs/STAGE1A_IMPLEMENTATION_PLAN.md`
- `docs/MULTI_SCALE_MSB_INTERPRETATION.md`
- `docs/TEMPORAL_CNN_BASICS.md`
- `docs/experiments/`
- `docs/worklogs/`

## Worklogs
- Create a worklog for architecture changes, migration decisions, or experiment-enabling implementation changes
- Keep design rationale close to the code that depends on it
- Dataset profile changes, label prior changes, feature changes, and objective changes require worklog coverage
- Do not put routine experiment readouts in `docs/worklogs/`; use `docs/experiments/`
