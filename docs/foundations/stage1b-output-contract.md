# Stage 1B Output Contract

**Status**: Draft v2
**Scope**: Stage 1B signal contract for downstream Stage 2 actionability work
**Primary selected Stage 1A checkpoint**: `ce_supcon_long_branch_ce_aux_v1_seed_41_e50/epoch_020.pt`
**Current Stage 1B candidates**:

```text
H8 fresh trigger: stage1b_h8_fresh_break_v3_z_fused_proximity
H16 dominant flow: stage1b_h16_dominant_v3_z_fused_proximity
```

## Purpose

Stage 1B answers a forward-structure question:

> Önümüzdeki horizon içinde mevcut yapının trade-relevant structural event'i ne?

Stage 1B does not produce a TradePlan. It does not decide entry, stop, target, leverage, sizing, or expected PnL.

The role of this contract is to define what Stage 1B can hand to Stage 2.

## Why v1 Was Not Enough

The original contract reduced Stage 1B to:

```text
break occurrence + direction
```

This was too narrow.

Example:

```text
current_state = bullish
fresh_break_target = none
future path = bullish_to_bearish_reversal
```

Fresh-break semantics correctly says `none`, because the anchor is already beyond the bullish break level. But TradePlan design still needs to know that the already-broken bullish structure is reversing bearish.

Therefore Stage 1B must distinguish:

```text
fresh break
continuation
reversal
no event
```

## Selected Horizons

Primary horizons:

| Horizon | Role |
|---|---|
| `H8` | near-term fresh trigger / immediate setup reading |
| `H16` | broader dominant forward flow / continuation-reversal reading |

H8 remains useful for near trigger timing. H16 is more useful for deciding whether the structural event is likely to continue or reverse.

## Input Contract

Stage 1B input is:

```text
Stage1BInput:
  latent:
    z_fused

  current_structure:
    current_label: intact | bullish | bearish
    current_close
    bull_level
    bear_level

  proximity_context:
    bull_distance_pct
    bear_distance_pct
    nearest_distance_pct
    nearest_side
    bull_close_count
    bear_close_count
    bull_wick_count
    bear_wick_count
    bull_final_excess
    bear_final_excess

  metadata:
    symbol
    timestamp
    horizon
```

Important:

```text
proximity_context future bilgi değildir.
```

It describes the current anchor window's relationship to structural levels. Future H8/H16 fields must not be used as input features.

## Core Output

The v2 Stage 1B output should be event-sequence based:

```text
Stage1BSignal:
  horizon: H8 | H16

  current_state:
    intact | bullish | bearish

  event_type:
    no_event
    fresh_break
    continuation
    reversal

  event_direction:
    none | bullish | bearish

  event_confidence:
    float

  expected_timing:
    immediate | soon | late | unknown

  maturity:
    trigger_near
    already_triggered
    extended
    invalidation_risk
    unknown

  supporting_context:
    bull_distance_pct
    bear_distance_pct
    nearest_side
    nearest_distance_pct
```

## Event Semantics

| `event_type` | Meaning |
|---|---|
| `no_event` | Horizon içinde trade-relevant structural event yok |
| `fresh_break` | Anchor pre-break tarafta; horizon içinde yeni close-confirmed break oluşuyor |
| `continuation` | Anchor zaten bullish/bearish broken; aynı yönde yapı korunuyor veya devam ediyor |
| `reversal` | Anchor zaten bullish/bearish broken; karşı yönde close-confirmed break oluşuyor |

`event_direction` is the structurally relevant forward direction:

| Case | `event_direction` |
|---|---|
| fresh bullish break | `bullish` |
| fresh bearish break | `bearish` |
| bullish continuation | `bullish` |
| bearish continuation | `bearish` |
| bullish-to-bearish reversal | `bearish` |
| bearish-to-bullish reversal | `bullish` |
| no event | `none` |

## Mapping From Current Label Fields

Current label artifacts already expose the required semantic fields:

```text
h{H}_fresh_break_direction
h{H}_break_semantic_label
h{H}_post_break_outcome
h{H}_dominant_forward_direction
```

Initial mapping:

| Field pattern | Proposed output |
|---|---|
| `h{H}_fresh_break_direction in bullish/bearish` | `event_type=fresh_break`, `event_direction=<direction>` |
| `h{H}_post_break_outcome = bullish_continuation` | `event_type=continuation`, `event_direction=bullish` |
| `h{H}_post_break_outcome = bearish_continuation` | `event_type=continuation`, `event_direction=bearish` |
| `h{H}_post_break_outcome = bullish_to_bearish_reversal` | `event_type=reversal`, `event_direction=bearish` |
| `h{H}_post_break_outcome = bearish_to_bullish_reversal` | `event_type=reversal`, `event_direction=bullish` |
| `h{H}_post_break_outcome = failed_bullish_break` | `event_type=no_event` or `event_type=reversal_risk` in a later version |
| `h{H}_post_break_outcome = failed_bearish_break` | `event_type=no_event` or `event_type=reversal_risk` in a later version |
| `h{H}_post_break_outcome = none` | `event_type=no_event`, `event_direction=none` |

`failed_*` outcomes are intentionally conservative in v2. They are structurally meaningful, but not yet clean enough to route directly into a TradePlan.

## Examples

### Intact To Fresh Bearish

```text
current_state: intact
h8_break_semantic_label: fresh_bearish_break
h8_post_break_outcome: bearish_continuation
```

Output:

```text
event_type: fresh_break
event_direction: bearish
expected_timing: soon
maturity: trigger_near
```

### Already Bullish To Bearish Reversal

```text
current_state: bullish
h8_break_semantic_label: already_broken_bullish
h8_post_break_outcome: bullish_to_bearish_reversal
h8_dominant_forward_direction: bearish
```

Output:

```text
event_type: reversal
event_direction: bearish
expected_timing: soon
maturity: already_triggered
```

This is not a fresh-break signal. It is a reversal signal.

## Stage 2 Bridge

Given:

```text
Stage1BSignal + Stage1BInput supporting context
```

Stage 2 should answer:

```text
no_trade
wait_for_break
wait_for_retest
actionable_break_candidate
actionable_reversal_candidate
actionable_continuation_candidate
```

Stage 2 is responsible for:

- entry trigger
- invalidation / stop
- target
- expected R
- fees / slippage adjustment
- no-trade filtering

## What Stage 1B Does Not Decide

Stage 1B does not decide:

- market entry now
- limit/stop entry level
- retest requirement
- stop-loss level
- target level
- expected risk/reward
- position size
- final no-trade vs actionable trade

These belong to Stage 2 and the final TradePlan contract.

## Current Implementation State

Current trained models are still partial implementations of this contract:

| Model | Covers | Gap |
|---|---|---|
| `stage1b_h8_fresh_break_v3_z_fused_proximity` | near-term fresh trigger quality | does not directly output reversal/continuation |
| `stage1b_h16_dominant_v3_z_fused_proximity` | broader dominant forward direction | does not explicitly classify event type |

Next implementation step:

```text
Train / audit a Stage 1B event-sequence head:
  event_type: no_event / fresh_break / continuation / reversal
  event_direction: none / bullish / bearish
```

This should be evaluated separately from the fresh-break H8 predictor.
