# Experiment Readout: 2026-04-17 — CE + SupCon Long Symbol V2

**Variant**: `ce_supcon_long_symbol_v2`  
**Status**: Rejected  
**Checkpoint**: `data/stage1a/binance/15m/checkpoints/ce_supcon_long_symbol_v2`

---

## Setup

Bu koşu şu mask mantığını test etti:

- `same-label + different-symbol` = positive
- `same-label + same-symbol` = neutral
- `different-label` = negative

Yapı:

- classifier head: `z_fused`
- contrastive target: `z_long_proj`
- batch sampling: symbol-balanced
- `supcon_weight = 0.05`
- `supcon_temperature = 0.1`

Amaç, `ce_supcon_long_v1`de görülen semantik kazancı koruyup symbol shortcut etkisini daha kontrollü baskılamaktı.

## Training Summary

- best val macro-F1: `0.8890`
- best val loss: `0.2713`
- early stopping: epoch `32`

Val classifier sonucu kabul edilemez ölçüde kötü değil, ama `ce_supcon_long_v1`i de geçemiyor:

- `ce_only`: `0.8901`
- `ce_supcon_long_v1`: `0.8900`
- `ce_supcon_long_symbol_v1`: `0.8906`
- `ce_supcon_long_symbol_v2`: `0.8890`

## Overall Readout

Ana sonuç şu:

- `symbol_v1`e göre biraz daha az zararlı
- ama `long_v1`in latent geometry kazancını koruyamıyor
- `z_long`, `z_long_proj` ve `z_fused` tarafında symbol clustering yeniden güçleniyor

Val overall:

- `z_long`
  - `long_v1`: label `0.8096`, symbol `0.3171`
  - `long_symbol_v2`: label `0.4661`, symbol `0.9233`
- `z_long_proj`
  - `long_v1`: label `0.8079`, symbol `0.2661`
  - `long_symbol_v2`: label `0.3650`, symbol `0.6677`
- `z_fused`
  - `long_v1`: label `0.8424`, symbol `0.2986`
  - `long_symbol_v2`: label `0.8446`, symbol `0.6111`

Test overall:

- `z_long`
  - `long_v1`: label `0.8080`, symbol `0.2987`
  - `long_symbol_v2`: label `0.4748`, symbol `0.9150`
- `z_long_proj`
  - `long_v1`: label `0.8067`, symbol `0.2621`
  - `long_symbol_v2`: label `0.3808`, symbol `0.6098`
- `z_fused`
  - `long_v1`: label `0.8425`, symbol `0.2784`
  - `long_symbol_v2`: label `0.8394`, symbol `0.6595`

Yani neutral same-symbol mask'i, same-symbol çiftleri doğrudan itmemiş olsa da latent uzayın cross-symbol hizalanmasını koruyamadı.

## Bucket Readout

Kritik bucket'larda desen yine olumsuz:

- `borderline_intact_break`
  - `z_long` ve `z_long_proj` tarafında label agreement düşüyor
  - symbol agreement çok sert biçimde yükseliyor
- `wick_sweep_up` / `wick_sweep_down`
  - `z_fused` label agreement bazı yerlerde korunuyor
  - ama neredeyse tüm komşuluk yapısı tekrar symbol üzerinden organize oluyor
- `high_vol_intact_wick_sweep`
  - bazı yerel top1 label artıları var
  - fakat bunlar yoğun symbol clustering pahasına geliyor

Bu yüzden kazanım, semantik hizalanma olarak değil, daha çok symbol-içi kümelenme olarak okunmalı.

## Interpretation

Bu koşu iki şeyi netleştiriyor:

1. `label_diff_symbol_neutral_same_symbol` mask'i, `label_diff_symbol` varyantına göre daha yumuşak ama yönü düzeltmiyor.
2. `ce_supcon_long_v1` hâlâ en iyi branch-aware SupCon varyantı.

Başka bir deyişle:

- symbol-aware fikri tamamen yanlış değil
- ama mevcut batch ve loss düzeninde cross-symbol positive sinyal yeterince baskın değil
- sonuçta model yine symbol shortcut tarafına kayıyor

## Decision

Bu varyant `rejected`.

Sebep:

- `ce_supcon_long_v1`e göre latent geometry belirgin biçimde daha kötü
- classifier tarafında bu kaybı telafi edecek bir artı yok
- `z_fused` tarafında symbol shortcut ciddi biçimde geri geliyor

Şu an en iyi SupCon adayı hâlâ:

- `ce_supcon_long_v1`

## Related

- `docs/experiments/ce_supcon/2026-04-17-ce-supcon-long-symbol-v2-plan.md`
- `docs/experiments/ce_supcon/2026-04-15-ce-supcon-long-v1-readout.md`
- `docs/experiments/ce_supcon/2026-04-16-ce-supcon-long-symbol-v1-readout.md`
