# CE + SupCon Experiments

Bu klasör, `CE + SupCon` deney planı ve sonuç okumalarını toplar.

## Timeline

1. `fused projection v1`
   - mixed
   - `z_long` semantik olarak iyileşti ama symbol bias da büyüdü

2. `long projection v1`
   - best current
   - `z_long` ve `z_fused` tarafında en iyi latent geometry dengesi burada görüldü

3. `long projection v2`
   - rejected
   - `supcon_weight=0.03` beklenen dengeyi getirmedi

4. `long symbol v1`
   - rejected
   - ilk symbol-aware positive tanımı geometry'yi bozdu

5. `long symbol v2`
   - in progress
   - cross-symbol positive + same-symbol neutral mask testi

Önerilen sıra:

1. `2026-04-14-ce-supcon-experiment-plan.md`
2. `2026-04-14-ce-supcon-v1-readout.md`
3. `2026-04-15-ce-supcon-long-v1-readout.md`
4. `2026-04-16-ce-supcon-long-v2-readout.md`
5. `2026-04-16-ce-supcon-long-symbol-v1-readout.md`
6. `2026-04-17-ce-supcon-long-symbol-v2-plan.md`

Kısa sonuç:

- fused-projection SupCon karışık sonuç verdi
- branch-aware `long v1` şu an en iyi varyant
- `long v2` ve ilk symbol-aware deneme reddedildi
- `long symbol v2` şu anda in-progress
