# msb-representation

`msb-representation`, Market Structure Break semantiği için çok ölçekli temsil öğrenmesine odaklanan araştırma reposudur.

## Kapsam

Bu repo bilinçli olarak `coin-oracle`'dan daha dar bir alana odaklanır.

Merkezde olan başlıklar:

- Stage 1A dataset üretimi
- yapısal latent öğrenimi
- short / long encoder ve fusion deneyleri
- latent evaluation araçları

Başlangıç kapsamına bilinçli olarak alınmayanlar:

- canlı sinyal pipeline'ı
- pozisyon yönetimi
- replay / execution çerçevesi
- production anomaly gate mantığı

## İlk Milestone

İlk hedef temiz ve tekrar üretilebilir bir Stage 1A baseline kurmaktır:

- named dataset profile'lar
- dual-window Stage 1A dataset contract
- `CE-only` baseline training
- temel test paketi

## Kurulum

Önerilen ortam kurulumu `uv` ile:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

Geliştirme araçlarıyla birlikte:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Görselleştirme araçları da gerekiyorsa:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev,viz]"
```

Notebook tabanlı latent inceleme de gerekiyorsa:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e ".[dev,viz,notebook]"
```

## İlk Çalıştırma Komutları

Dataset build:

```bash
python scripts/build_stage1a_dataset.py \
  --data-root data \
  --exchange binance \
  --timeframe 15m \
  --bars-per-day 96 \
  --dataset-profile core4 \
  --train-end 2025-09-30 \
  --val-end 2025-12-31
```

CE-only baseline training:

```bash
python scripts/train_stage1a.py \
  --dataset-root data/stage1a/binance/15m \
  --epochs 50 \
  --batch-size 32
```

Latent export:

```bash
python scripts/export_stage1a_latents.py \
  --dataset-root data/stage1a/binance/15m \
  --checkpoint-dir data/stage1a/binance/15m/checkpoints/ce_only \
  --split val
```

Val latent analizi:

```bash
python scripts/analyze_stage1a_latents.py \
  --latent-path data/stage1a/binance/15m/checkpoints/ce_only/analysis/val_latents.npz
```

Temel test paketi:

```bash
PYTHONPATH=src pytest -q \
  tests/test_stage1a_dataset.py \
  tests/test_stage1a_labels.py \
  tests/test_stage1a_profiles.py \
  tests/test_normalizer.py \
  tests/test_windows.py
```

## Paket Yapısı

```text
src/msb_repr/
  data/
  stage1a/
```

## Temel Dokümanlar

- `docs/foundations/market-structure-edge-program.md`
- `docs/foundations/stage1-msb-representation-framing.md`
- `docs/foundations/stage1a-latent-concepts.md`
- `docs/STAGE1A_IMPLEMENTATION_PLAN.md`
- `docs/MULTI_SCALE_MSB_INTERPRETATION.md`
- `docs/TEMPORAL_CNN_BASICS.md`
- `docs/WORKLOG_GUIDE.md`
- `docs/worklogs/`
