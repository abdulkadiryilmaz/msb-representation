## Overview
- Project: msb-representation
- Focus: multi-scale representation learning for Market Structure Break semantics
- Primary goal: structural latent learning, not live trading pipeline development
- Primary timeframe: 15m candles
- Core setup: short window `48`, long window `288`, shared-anchor dual-window dataset
- Default Stage 1A dataset universe: `core4` = BTC, ETH, SOL, XRP

## Dil ve Dokümantasyon
- README ve repo içi ana dokümantasyon Türkçe tutulur
- Teknik terimler gerekiyorsa İngilizce bırakılabilir, ama açıklama dili Türkçe olmalı
- Araştırma framing / kavramsal temel belgeleri `docs/foundations/` altında tutulur
- Deney planları ve deney sonuç okumaları `docs/experiments/` altında tutulur
- Mimari değişiklik, veri sözleşmesi değişikliği ve yön değiştirici kararlar `docs/worklogs/` altında belgelenir
- Genel deney süreci: `docs/EXPERIMENT_WORKFLOW.md`
- Üst seviye deney özeti: `docs/EXPERIMENT_INDEX.md`
- Worklog rehberi: `docs/WORKLOG_GUIDE.md`

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

## Repo Sınırı
- Bu repo araştırma reposudur
- `coin-oracle` production / replay / signal execution tarafı olarak kalır
- Ortak utility taşınacaksa önce Stage 1A bağımlılığı gerçekten var mı kontrol edilir
- "Belki lazım olur" diye eski modül kopyalanmaz

## Common Workflows
- Build dataset:
  - `python scripts/build_stage1a_dataset.py --data-root data --exchange binance --timeframe 15m --bars-per-day 96 --dataset-profile core4 --train-end 2025-09-30 --val-end 2025-12-31`
- Train CE-only baseline:
  - `python scripts/train_stage1a.py --dataset-root data/stage1a/binance/15m --epochs 50 --batch-size 32`

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
