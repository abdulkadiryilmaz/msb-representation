# Worklog: 2026-05-13 — Branch-Isolated Pressure Aux Design

**Status**: Tested / rejected in v1  
**Related experiments**: `ce_supcon_long_aux_v1`, `ce_supcon_long_aux_v2`  

---

## Problem

`aux_v1` and `aux_v2` showed that pressure supervision is learnable, but the current auxiliary route can damage `z_long` geometry.

Current route:

```text
CE(pressure) -> z_fused -> z_short
                         -> z_long
```

Since `z_fused = concat(z_short, z_long)`, pressure CE sends gradient into both branches. `aux_v2` made the failure mode clear: pressure probe improved, but `z_long` and `z_long_proj` became symbol-heavy.

This suggests the issue is not only auxiliary weight. The issue is where the pressure gradient is allowed to flow.

---

## Design Decision

Add a configurable pressure auxiliary head input:

```text
pressure_head_input in {"z_fused", "z_short"}
```

Default remains `z_fused` for backward compatibility.

The next experiment will use:

```text
CE(confirmed_state) -> z_fused
SupCon              -> z_long_proj
CE(pressure)        -> z_short
```

Maturity auxiliary is disabled for the first branch-isolation ablation.

---

## Expected Behavior

If the diagnosis is correct:

- pressure probe should improve or remain above `ce_supcon_long_v1`
- `z_long` symbol agreement should stay close to `ce_supcon_long_v1`
- `z_long_proj` should not become symbol-heavy
- borderline behavior should not regress materially

If symbol shortcut still appears in `z_long`, then pressure supervision is not the only source of the shortcut and the interaction with CE/SupCon needs deeper review.

---

## Result

`ce_supcon_long_short_pressure_aux_v1` did not validate the hypothesis.

Observed:

- `z_short` pressure probe improved
- `z_fused` pressure probe stayed high
- `z_long` and `z_long_proj` became strongly symbol-heavy

This means direct pressure gradient into `z_long` was not the whole failure mechanism. A pressure-supervised `z_short` can still alter the fused classifier training dynamics enough that `z_long` loses the branch-aware SupCon geometry.

---

## Implementation Notes

- `Stage1AModel` now supports `pressure_head_input="z_short"`.
- `scripts/train_stage1a.py` exposes `--pressure-head-input`.
- Checkpoint metadata stores `pressure_head_input`.
- `load_checkpoint_bundle` defaults missing metadata to `z_fused` for old checkpoints.
