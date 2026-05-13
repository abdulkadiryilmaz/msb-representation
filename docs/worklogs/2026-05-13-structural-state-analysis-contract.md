# Worklog: 2026-05-13 — Structural State Analysis Contract

**Status**: Complete  
**Scope**: Stage 1A analysis / readout contract

---

## Problem

Stage 1A ana label'ı hâlâ tek eksenli:

- `intact`
- `bullish`
- `bearish`

Bu label, current structural state için doğru ilk çerçeve olsa da readout tarafında fazla bilgi sıkıştırıyordu.

Özellikle `intact` örneklerde:

- clean intact
- up/down pressure
- wick sweep
- borderline break

aynı base label altında kalıyordu.

Benzer şekilde `bullish` / `bearish` örneklerde:

- confirmed break hâlâ holding mi
- yoksa recent zone sonunda reverted mı
- strong displacement var mı

ayrımı görünür değildi.

Bu durum `1567` gibi örnekleri yorumlarken kafa karışıklığı oluşturdu:

- base label `intact`
- model tahmini `bearish`
- fakat örnek aslında `down_pressure + borderline` state taşıyor

Yani modelin hatası yalnızca "yanlış sınıf" olarak değil, pressure ile confirmed state sınırını erken geçme davranışı olarak okunmalı.

---

## Change

Model, dataset label'ı ve training objective değiştirilmedi.

Yalnızca analysis/readout çıktılarına factorized structural state alanları eklendi:

- `confirmed_state`
  - `intact`
  - `bullish`
  - `bearish`
- `confirmed_direction`
  - `none`
  - `up`
  - `down`
- `pressure_state`
  - `neutral`
  - `up_pressure`
  - `down_pressure`
  - `mixed_pressure`
  - `non_intact`
- `break_maturity`
  - `clean`
  - `wick_sweep`
  - `borderline`
  - `confirmed`
  - `strong_confirmed`
  - high-vol variants where applicable
- `structural_direction`
  - `none`
  - `up`
  - `down`
  - `mixed`
- `holding_status`
  - `holding`
  - `reverted`
  - `mixed`
  - `none`
  - `unknown`

Bu alanlar şu çıktılara eklendi:

- `hard_cases.csv`
- `symbol_bucket_full_distribution.csv`
- manual NN review `candidates.csv`

---

## Rationale

Amaç Stage 1A'yı hemen yeni bir supervised task'a çevirmek değil.

Amaç:

- mevcut latent davranışını daha doğru okumak
- classifier hatalarının hangi structural axis üzerinde oluştuğunu görmek
- sonraki deneyi daha hedefli tasarlamak

Özellikle şu ayrım kritik:

```text
confirmed_state = intact
pressure_state = down_pressure
break_maturity = borderline
```

Bu örnek base label olarak `intact` kalmalıdır. Ancak clean intact ile aynı readout kategorisinde okunmamalıdır.

Bu sayede şu yorum mümkün olur:

```text
Model confirmed_state'i yanlış sınıflıyor olabilir,
ama pressure direction'ı doğru yakalıyor olabilir.
```

Bu, Stage 1A representation hedefi açısından daha doğru bir teşhis üretir.

---

## Program-Level Meaning

Bu değişiklik `TradePlan` üretmez.

Ancak `Market Structure Edge Program` hedefindeki `structural_context` alanını daha iyi tarif etmeye yarar.

Stage 2 ileride şuna ihtiyaç duyacaktır:

- clean intact mı?
- pressure intact mı?
- confirmed break mi?
- confirmed break hâlâ holding mi?
- yoksa reverted mı?

Bu ayrımlar `no_trade` / `TradePlan` ayrımı için base `bullish/bearish/intact` label'ından daha bilgilendiricidir.

---

## Verification

Çalıştırılan test:

```bash
PYTHONPATH=src pytest -q tests/test_stage1a_analysis.py
```

Beklenen davranış:

- directional borderline intact alanları korunur
- confirmed break örnekleri `holding_status` ile split edilebilir
- structural state alanları analysis-only olarak üretilir

