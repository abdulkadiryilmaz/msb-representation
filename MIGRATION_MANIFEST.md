# Migration Manifest

Bu manifest, yeni repo oluşturulurken ilk aşamada hangi dosyaların kopyalanacağını listeler.

Not:

- Aşağıdaki `coin_oracle` path'leri kaynak repo içindeki orijinal konumları gösterir
- yeni repoda karşılıkları `src/msb_repr/...` altına taşınmıştır

## Copy First

Kod:

- `src/coin_oracle/stage1a/__init__.py`
- `src/coin_oracle/stage1a/config.py`
- `src/coin_oracle/stage1a/dataset.py`
- `src/coin_oracle/stage1a/features.py`
- `src/coin_oracle/stage1a/labels.py`
- `src/coin_oracle/stage1a/model.py`
- `src/coin_oracle/stage1a/profiles.py`
- `src/coin_oracle/stage1a/trainer.py`
- `src/coin_oracle/data/features.py`
- `src/coin_oracle/data/ingestion.py`
- `src/coin_oracle/data/normalizer.py`
- `src/coin_oracle/data/windows.py`
- `scripts/build_stage1a_dataset.py`
- `scripts/train_stage1a.py`
- `scripts/fetch_ohlcv.py`

Testler:

- `tests/conftest.py`
- `tests/test_features.py`
- `tests/test_normalizer.py`
- `tests/test_windows.py`
- `tests/test_stage1a_dataset.py`
- `tests/test_stage1a_labels.py`
- `tests/test_stage1a_profiles.py`

Dokümantasyon:

- `docs/STAGE1A_IMPLEMENTATION_PLAN.md`
- `docs/MULTI_SCALE_MSB_INTERPRETATION.md`
- `docs/TEMPORAL_CNN_BASICS.md`
- `docs/FEATURES.md`
- `docs/worklogs/2026-04-12-stage1-msb-representation-framing.md`

## Leave Behind

- signal pipeline scripts
- replay / backtest scripts
- predictor / old AE training scripts
- signal state modules
- position / cooldown logic

## Refactor Immediately After Copy

- package import paths:
  - `coin_oracle.*` -> `msb_repr.*`
- docs:
  - remove live pipeline framing
  - move Stage 1A docs to top-level relevance
- tests:
  - drop signal-specific fixtures
