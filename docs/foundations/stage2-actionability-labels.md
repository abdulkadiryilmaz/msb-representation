# Stage 2 Actionability Labels

**Status**: Draft v1, Stage 1B event-sequence contract sonrası revizyon gerekli
**Scope**: Rule-based Stage 2 actionability label contract
**Depends on**:

- `docs/foundations/stage1b-output-contract.md`
- `docs/foundations/stage2-actionability-framing.md`

Not: Bu label sözleşmesi ilk H8 `break + direction` Stage 1B sinyaline göre tasarlandı. Stage 1B output contract v2 ile `event_type` ve `event_direction` eklendiği için Stage 2 v2 label seti, break adaylarının yanında reversal ve continuation adaylarını da ayrı ele almalıdır.

## Amaç

Bu belge Stage 2 v1 için üretilecek actionability label'larının sözleşmesini tanımlar.

Label'ın yanıtladığı soru:

> Stage 1B'nin işaret ettiği H8 break sinyali, fiyat yolu ve seviye bağlamı açısından TradePlan adayına dönüşebilir mi?

Bu label PnL label'ı değildir. Amaç, ilk aşamada yapısal olarak planlanabilir setup'ları ayrıştırmaktır.

## Label Set

```text
0 = no_trade
1 = wait_for_break
2 = wait_for_retest
3 = actionable_break_candidate
```

Sınıflar birbirini dışlar. Bir örnek yalnızca tek label alır.

## Input Contract

Her anchor için gerekli alanlar:

```text
symbol
timestamp
current_price
stage1b_side: long | short | none
p_break
p_direction
break_confidence
direction_confidence
bull_level
bear_level
effective_break_pct
distance_to_bull_level
distance_to_bear_level
future_ohlcv_path
```

İlk v1 için Stage 2 label üretimi, Stage 1B predictor çıktıları ve raw OHLCV future path üzerinden yapılır.

## Horizon Policy

Stage 1B primary signal H8'dir:

```text
H8 = 8 bars = 2 hours
```

Stage 2 v1 iki pencere kullanır:

```text
decision_horizon = H8
observation_horizon = H16
```

Neden:

- H8, yakın break / immediate actionability için ana penceredir.
- `wait_for_retest` sınıfı için break sonrası kısa bir ek gözlem alanı gerekir.
- H16, retest/hold davranışını görmek için ilk makul üst sınırdır.

İlk label üretiminde H16 yalnızca path gözlemi için kullanılır; Stage 1B'nin ana tahmin hedefi H8 olarak kalır.

## Side ve Trigger Level

Stage 1B side label kaynağıdır:

```text
side = stage1b_side
```

Side `none` ise:

```text
label = no_trade
reason_code = stage1b_no_break
```

Long için:

```text
trigger_level = bull_level
opposite_level = bear_level
```

Short için:

```text
trigger_level = bear_level
opposite_level = bull_level
```

Eksik veya geçersiz trigger level varsa:

```text
label = no_trade
reason_code = missing_trigger_level
```

## Distance Buckets

Distance, anchor fiyatının trigger seviyesine uzaklığıdır:

```text
distance_to_trigger = abs(trigger_level - current_price) / current_price
```

Başlangıç bucket'ları:

```text
at_or_beyond_trigger:
  long: current_price >= trigger_level
  short: current_price <= trigger_level

near_trigger:
  distance_to_trigger <= 0.0050

watchable:
  0.0050 < distance_to_trigger <= 0.0150

far:
  distance_to_trigger > 0.0150
```

Bu eşikler ilk v1 değerleridir. Audit sonrası sweep edilebilir.

## Break Confirmation

Expected-side close-confirmed break:

```text
long_break:
  close > trigger_level * (1 + effective_break_pct)

short_break:
  close < trigger_level * (1 - effective_break_pct)
```

Alanlar:

```text
break_in_h8: bool
break_in_h16: bool
break_time_bars: first break bar offset, 1-indexed
break_price: close at first confirmed break
```

## Invalidation Proxy

Stage 2 v1 için invalidation rule-based tanımlanır.

Long için:

```text
primary invalidation = min(anchor recent lows within short window)
fallback invalidation = bear_level
```

Short için:

```text
primary invalidation = max(anchor recent highs within short window)
fallback invalidation = bull_level
```

Eğer primary ve fallback geçersizse:

```text
label = no_trade
reason_code = missing_invalidation
```

Not: İlk implementasyonda recent structural low/high hesaplamak zorlaşırsa, v1.0 fallback olarak opposite level ile başlayabilir. Bu durumda readout'ta açıkça belirtilmelidir.

## R Proxy

Risk:

```text
long risk = trigger_level - invalidation_level
short risk = invalidation_level - trigger_level
```

Geçerli risk koşulu:

```text
risk > 0
```

Favorable excursion:

```text
long MFE = max(high_after_anchor) - trigger_level
short MFE = trigger_level - min(low_after_anchor)
```

Adverse excursion:

```text
long MAE = trigger_level - min(low_after_anchor)
short MAE = max(high_after_anchor) - trigger_level
```

R multiple:

```text
mfe_r = MFE / risk
mae_r = MAE / risk
```

Başlangıç target eşiği:

```text
min_mfe_r = 1.0
```

Audit sweep adayları:

```text
1.0R, 1.5R, 2.0R
```

## Execution Cost / Net Target Guardrails

Stage 2 v1 `mfe_r` ile R bazlı potansiyeli ölçer, fakat fee/slippage sonrası hedefin anlamlı kalıp kalmadığını açıkça kontrol etmez.

Stage 2 v2 için ek guardrail gerekir:

```text
stop_pct = risk / entry_price
gross_target_pct = abs(target - entry_price) / entry_price
net_target_pct = gross_target_pct - estimated_cost_pct
```

Başlangıç değerleri:

```text
min_stop_pct = 0.0015        # 0.15%
max_stop_pct = 0.0080        # 0.80%
estimated_cost_pct = 0.0008  # 0.08% round-trip fee + slippage proxy
min_net_target_pct = 0.0030  # 0.30%
primary_target_r = 2.0
```

Yorum:

- `min_stop_pct`: Çok dar stop'ları eler. Bu tip işlemler R olarak iyi görünse bile fee/slippage ve spread tarafından ezilebilir.
- `max_stop_pct`: Çok geniş stop'ları eler. Bu tip işlemlerde 2R hedef fiyat olarak fazla uzaklaşabilir.
- `estimated_cost_pct`: İlk sabit maliyet modeli. Exchange fee, spread ve slippage için round-trip kaba proxy'dir.
- `min_net_target_pct`: Hedefe ulaşılsa bile maliyetlerden sonra anlamlı fiyat hareketi kalmasını zorlar.
- `primary_target_r`: Stage 2 v2 için ilk ana hedef. `1.0R` v1 label'ı geniş tutmak için kullanışlıydı; TradePlan aday kalitesi için `2.0R` daha anlamlı başlangıçtır.

Valid trade-plan candidate için önerilen ek koşullar:

```text
min_stop_pct <= stop_pct <= max_stop_pct
net_target_pct >= min_net_target_pct
```

Örnek:

```text
entry = 94607
stop = 94990
risk = 383
target_2R = 93840

stop_pct = 383 / 94607 ~= 0.405%
gross_target_pct = 766 / 94607 ~= 0.810%
net_target_pct = 0.810% - 0.080% = 0.730%
```

Bu örnek guardrail'lerden geçer:

```text
0.15% <= 0.405% <= 0.80%
0.730% >= 0.30%
```

Dar stop örneği:

```text
entry = 95521
stop = 95531
risk ~= 0.010%
target_2R ~= 0.020%
net_target_pct ~= 0.020% - 0.080% = -0.060%
```

Bu örnek R olarak hedefe ulaşsa bile pratikte anlamlı değildir ve v2 guardrail tarafından elenmelidir.

## Path Outcome

Future path için sıralı olaylar hesaplanır:

```text
first_break_time
first_target_time
first_invalidation_time
first_retest_time
```

Target:

```text
long target = trigger_level + min_mfe_r * risk
short target = trigger_level - min_mfe_r * risk
```

Invalidation:

```text
long invalidated if low <= invalidation_level
short invalidated if high >= invalidation_level
```

Retest / hold proxy:

Long için:

```text
after break, low touches or approaches trigger_level
and closes back above trigger_level
```

Short için:

```text
after break, high touches or approaches trigger_level
and closes back below trigger_level
```

Retest tolerance:

```text
retest_tolerance = 0.0025
```

## Label Priority

Label üretimi deterministik ve öncelikli olmalıdır.

Önerilen sıra:

```text
1. hard no_trade filters
2. actionable_break_candidate
3. wait_for_retest
4. wait_for_break
5. no_trade fallback
```

### 1. Hard no_trade filters

Şu koşullardan biri varsa `no_trade`:

```text
side == none
break_confidence < 0.70
missing trigger level
missing invalidation
risk <= 0
distance bucket == far
no expected-side break in H16
opposite invalidation occurs before expected-side break
```

Reason code örnekleri:

```text
stage1b_no_break
low_confidence
missing_trigger_level
missing_invalidation
invalid_risk
too_far_from_trigger
no_break_in_observation_horizon
invalidated_before_break
```

### 2. actionable_break_candidate

Koşullar:

```text
break_confidence >= 0.70
distance bucket in {at_or_beyond_trigger, near_trigger}
expected-side break in H8
target reached before invalidation
mfe_r >= min_mfe_r
```

Bu sınıf, immediate veya yakın breakout planı üretmeye en uygun örnekleri temsil eder.

### 3. wait_for_retest

Koşullar:

```text
break_confidence >= 0.70
expected-side break in H8
price at anchor is at_or_beyond_trigger
or break displacement is large
retest/hold observed within H16
target not reached cleanly before retest
```

Bu sınıf, break yönü doğru olsa bile immediate chase yerine retest beklemenin daha makul olduğu örnekleri temsil eder.

Başlangıç displacement proxy:

```text
break_displacement = abs(break_price - trigger_level) / trigger_level
large_displacement = break_displacement >= 0.0050
```

### 4. wait_for_break

Koşullar:

```text
break_confidence >= 0.70
distance bucket in {near_trigger, watchable}
no expected-side break at anchor
expected-side break occurs within H16
not invalidated before break
```

Bu sınıf, anchor anında entry üretmeyen ama takip edilmeye değer setup'ları temsil eder.

### 5. no_trade fallback

Yukarıdaki sınıflardan hiçbirine girmeyen örnek `no_trade` olur.

## Output Fields

Label üretim script'i en az şu alanları yazmalıdır:

```text
symbol
timestamp
label
label_name
side
reason_code
current_price
trigger_level
invalidation_level
target_level
distance_to_trigger
distance_bucket
break_confidence
direction_confidence
break_in_h8
break_in_h16
break_time_bars
first_target_time
first_invalidation_time
mfe
mae
mfe_r
mae_r
```

## Audit Requirements

İlk readout şu kontrolleri içermelidir:

- class distribution
- class distribution by symbol
- class distribution by confidence bucket
- class distribution by distance bucket
- MFE/MAE/R summary by class
- `actionable_break_candidate` path sanity examples
- `wait_for_retest` path sanity examples
- false positive Stage 1B break segmentinde label dağılımı
- high-confidence Stage 1B segmentinde actionable oranı

## Open Questions

- `break_confidence < 0.70` hard filter mı kalmalı, yoksa sadece feature olarak mı kullanılmalı?
- `distance_threshold` değerleri symbol-volatility adjusted olmalı mı?
- Invalidation için opposite level yeterli mi, yoksa short-window swing extraction zorunlu mu?
- `wait_for_retest` sınıfı H16 içinde yeterli support üretecek mi?
- `actionable_break_candidate` için `1R` çok gevşek mi, yoksa ilk audit için doğru başlangıç mı?
