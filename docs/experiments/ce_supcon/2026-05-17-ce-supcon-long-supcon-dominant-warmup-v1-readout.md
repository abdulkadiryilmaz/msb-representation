# Experiment Readout: 2026-05-17 — CE + SupCon Long SupCon-Dominant Warmup V1

**Status**: Best balanced candidate so far  
**Family**: `CE + SupCon`  
**Baseline**: `ce_supcon_long_frontload_v1`, `ce_supcon_long_supcon_warmup_v1`  
**Runs**: `seed_41`, `seed_42`  

---

## Setup

This experiment tests a softer version of hard SupCon-only warmup.

Schedule:

```text
epoch 1-5:
  loss = 0.10 * CE + 0.10 * SupCon

epoch 6+:
  loss = 1.00 * CE + 0.05 * SupCon
```

Motivation:

- `frontload_v1` preserved `z_short` / `z_fused`, but `seed_41` stayed symbol-heavy in `z_long`
- `SupCon-only warmup v1` fixed `z_long`, but damaged `z_short` / `z_fused`
- this run tests whether a small nonzero CE signal during warmup keeps branch balance while still activating SupCon early

---

## Seed 41 Training Dynamics

SupCon plateau is broken early while CE remains active enough for classifier performance to stay meaningful.

| Epoch | CE weight | SupCon weight | train CE | train SupCon | val F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.10 | 0.10 | 0.4417 | 3.1707 | 0.8360 |
| 5 | 0.10 | 0.10 | 0.3365 | 2.9716 | 0.8713 |
| 10 | 1.00 | 0.05 | 0.2932 | 2.9494 | 0.8813 |
| 20 | 1.00 | 0.05 | 0.2496 | 2.9127 | 0.8859 |
| 30 | 1.00 | 0.05 | 0.2250 | 2.8914 | 0.8863 |
| 40 | 1.00 | 0.05 | 0.2072 | 2.8762 | 0.8870 |
| 50 | 1.00 | 0.05 | 0.1948 | 2.8650 | 0.8819 |

---

## Seed 41 Validation Sweep

| Epoch | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.8052 | 0.3353 | 0.8062 | 0.4409 | 0.8050 | 0.2808 | 0.8254 | 0.3335 | 0.4353 |
| 10 | 0.7992 | 0.3175 | 0.8132 | 0.2914 | 0.8095 | 0.2544 | 0.8395 | 0.3273 | 0.5427 |
| 20 | 0.7788 | 0.3383 | 0.8210 | 0.3204 | 0.8166 | 0.2852 | 0.8435 | 0.3359 | 0.6788 |
| 30 | 0.7772 | 0.3296 | 0.8211 | 0.3506 | 0.8193 | 0.2804 | 0.8444 | 0.3287 | 0.7854 |
| 40 | 0.7624 | 0.3031 | 0.8198 | 0.3759 | 0.8160 | 0.3023 | 0.8443 | 0.3271 | 0.9059 |
| 50 | 0.7551 | 0.3205 | 0.8204 | 0.3024 | 0.8192 | 0.2693 | 0.8396 | 0.3272 | 0.8714 |

Validation-selected checkpoint for seed 41: `epoch_010.pt`.

Reason:

- `z_long` symbol agreement is much lower than frontload seed 41
- `z_long_proj` is clean
- `z_short` remains usable
- `z_fused` symbol agreement stays near frontload levels
- effective long/short is strong but not excessive

---

## Seed 41 Test Readout

Selected checkpoint: `epoch_010.pt`

| Embedding | label agree | symbol agree |
|---|---:|---:|
| `z_short` | 0.7964 | 0.3034 |
| `z_long` | 0.8102 | 0.2826 |
| `z_fused` | 0.8361 | 0.3012 |
| `z_long_proj` | 0.8084 | 0.2542 |

Branch usage:

| Metric | Value |
|---|---:|
| `z_long / z_short` norm ratio | 0.6016 |
| classifier long/short weight ratio | 0.9085 |
| effective long/short ratio | 0.5465 |
| `z_long` centroid symbol/label ratio | 0.0196 |
| `z_long_proj` centroid symbol/label ratio | 0.0179 |

---

## Comparison: Seed 41

| Run | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `frontload_seed41` | 0.8284 | 0.2769 | 0.8068 | 0.4860 | 0.8002 | 0.3217 | 0.8413 | 0.3054 | 0.2863 |
| `hard_warmup_seed41` | 0.5174 | 0.7091 | 0.8380 | 0.2701 | 0.8382 | 0.2562 | 0.8438 | 0.5384 | 7.8831 |
| `dominant_warmup_seed41` | 0.7964 | 0.3034 | 0.8102 | 0.2826 | 0.8084 | 0.2542 | 0.8361 | 0.3012 | 0.5465 |

Interpretation:

- frontload preserved `z_short` / `z_fused`, but `z_long` stayed symbol-heavy
- hard warmup fixed `z_long`, but damaged `z_short` / `z_fused`
- dominant warmup is the first seed 41 run that keeps both sides reasonably balanced

---

## Concern

The best seed 41 checkpoint appears early (`epoch_010.pt`).

After epoch 10:

- `z_long` label agreement continues to improve slightly
- `z_short` label agreement degrades
- effective long/short rises
- `z_fused` remains acceptable but does not clearly improve

This suggests that the warmup creates a good early balance, but the post-warmup schedule may not preserve it indefinitely.

Seed 42 should check whether this early-optimum behavior is systematic.

---

## Seed 42 Training Dynamics

Seed 42 also breaks the early SupCon plateau while keeping CE active during warmup.

| Epoch | CE weight | SupCon weight | train CE | train SupCon | val F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.10 | 0.10 | 0.4433 | 3.2645 | 0.8503 |
| 5 | 0.10 | 0.10 | 0.3386 | 2.9760 | 0.8738 |
| 10 | 1.00 | 0.05 | 0.2941 | 2.9502 | 0.8841 |
| 20 | 1.00 | 0.05 | 0.2527 | 2.9205 | 0.8842 |
| 30 | 1.00 | 0.05 | 0.2271 | 2.8987 | 0.8881 |
| 40 | 1.00 | 0.05 | 0.2112 | 2.8841 | 0.8812 |
| 50 | 1.00 | 0.05 | 0.1956 | 2.8715 | 0.8873 |

---

## Seed 42 Validation Sweep

| Epoch | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.8162 | 0.2938 | 0.8036 | 0.4806 | 0.8024 | 0.2819 | 0.8269 | 0.3043 | 0.4565 |
| 10 | 0.7886 | 0.3141 | 0.8044 | 0.3815 | 0.8006 | 0.2797 | 0.8388 | 0.3198 | 0.5918 |
| 20 | 0.7741 | 0.3126 | 0.8124 | 0.3845 | 0.8125 | 0.2871 | 0.8393 | 0.3241 | 0.6638 |
| 30 | 0.7631 | 0.2991 | 0.8179 | 0.2884 | 0.8154 | 0.2620 | 0.8448 | 0.3142 | 0.7480 |
| 40 | 0.7629 | 0.2867 | 0.8193 | 0.3077 | 0.8179 | 0.2667 | 0.8425 | 0.2961 | 0.8213 |
| 50 | 0.7522 | 0.2976 | 0.8204 | 0.3287 | 0.8171 | 0.2677 | 0.8426 | 0.3087 | 0.7795 |

Validation-selected checkpoint for seed 42: `epoch_040.pt`.

Reason:

- `z_fused` symbol agreement is the lowest among later high-quality checkpoints
- `z_long` and `z_long_proj` remain clean
- effective long/short is stronger than frontload but not in hard-warmup collapse territory

---

## Seed 42 Test Readout

Selected checkpoint: `epoch_040.pt`

| Embedding | label agree | symbol agree |
|---|---:|---:|
| `z_short` | 0.7711 | 0.2771 |
| `z_long` | 0.8158 | 0.2787 |
| `z_fused` | 0.8422 | 0.2839 |
| `z_long_proj` | 0.8143 | 0.2548 |

Branch usage:

| Metric | Value |
|---|---:|
| `z_long / z_short` norm ratio | 0.9151 |
| classifier long/short weight ratio | 0.9384 |
| effective long/short ratio | 0.8587 |
| `z_long` centroid symbol/label ratio | 0.0236 |
| `z_long_proj` centroid symbol/label ratio | 0.0212 |

---

## Multi-Seed Comparison

| Run | Split | `z_short` label | `z_short` symbol | `z_long` label | `z_long` symbol | `z_long_proj` label | `z_long_proj` symbol | `z_fused` label | `z_fused` symbol | effective long/short |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dominant_seed41_e10` | val | 0.7992 | 0.3175 | 0.8132 | 0.2914 | 0.8095 | 0.2544 | 0.8395 | 0.3273 | 0.5427 |
| `dominant_seed41_e10` | test | 0.7964 | 0.3034 | 0.8102 | 0.2826 | 0.8084 | 0.2542 | 0.8361 | 0.3012 | 0.5465 |
| `dominant_seed42_e40` | val | 0.7629 | 0.2867 | 0.8193 | 0.3077 | 0.8179 | 0.2667 | 0.8425 | 0.2961 | 0.8213 |
| `dominant_seed42_e40` | test | 0.7711 | 0.2771 | 0.8158 | 0.2787 | 0.8143 | 0.2548 | 0.8422 | 0.2839 | 0.8587 |

Compared to frontload:

- seed 41 `z_long` symbol agreement is repaired without hard-warmup `z_short` collapse
- seed 42 remains strong and does not regress
- `z_fused` symbol agreement stays near frontload/reference range
- effective long/short is higher and healthier than frontload, but far below hard warmup

---

## Interim Decision

For seed 41 and seed 42:

```text
SupCon-dominant warmup v1 is the best balanced candidate so far.
```

Remaining caveat:

- seed 41 peaks early at `epoch_010.pt`
- seed 42 peaks later around `epoch_040.pt`
- this suggests fixed-budget checkpoint selection remains important

Next design direction should shift away from pure schedule tuning and test direct `z_long` activity constraints:

```text
fused CE
+ SupCon(z_long_proj)
+ auxiliary CE head on z_long
```

Rationale:

- schedule experiments showed the mechanism
- the desired invariant, active label-dominant `z_long`, should now be made explicit in the objective
