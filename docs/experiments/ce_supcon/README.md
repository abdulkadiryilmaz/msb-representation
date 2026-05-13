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
   - rejected
   - neutral same-symbol mask, `long_v1`in geometry kazancını koruyamadı

6. `long factor v1`
   - rejected
   - pressure probe iyileşti ama `z_long` / `z_fused` symbol shortcut geri döndü

7. `long aux v1`
   - mixed / promising ablation
   - pressure ve borderline behavior iyileşti, ama `z_long` / `z_long_proj` label agreement `long_v1`e göre geriledi

8. `long aux v2`
   - rejected
   - pressure-only auxiliary F1 ve pressure probe'u güçlendirdi, fakat `z_long` / `z_long_proj` yeniden symbol-heavy oldu

Önerilen sıra:

1. `2026-04-14-ce-supcon-experiment-plan.md`
2. `2026-04-14-ce-supcon-v1-readout.md`
3. `2026-04-15-ce-supcon-long-v1-readout.md`
4. `2026-04-16-ce-supcon-long-v2-readout.md`
5. `2026-04-16-ce-supcon-long-symbol-v1-readout.md`
6. `2026-04-17-ce-supcon-long-symbol-v2-plan.md`
7. `2026-04-17-ce-supcon-long-symbol-v2-readout.md`
8. `2026-05-13-ce-supcon-long-factor-v1-plan.md`
9. `2026-05-13-ce-supcon-long-factor-v1-readout.md`
10. `2026-05-13-ce-supcon-long-aux-v1-plan.md`
11. `2026-05-13-ce-supcon-long-aux-v1-readout.md`
12. `2026-05-13-ce-supcon-long-aux-v2-plan.md`
13. `2026-05-13-ce-supcon-long-aux-v2-readout.md`

Kısa sonuç:

- fused-projection SupCon karışık sonuç verdi
- branch-aware `long v1` şu an en iyi varyant
- `long v2` ve ilk symbol-aware deneme reddedildi
- `long symbol v2` reddedildi
- `long factor v1` reddedildi; pressure sinyali güçlendi ama symbol shortcut guardrail'i bozuldu
- `long aux v1` promising ablation; coarse SupCon pairing korununca symbol guardrail bozulmadı, fakat aux ağırlıkları `z_long` geometry için hâlâ fazla olabilir
- `long aux v2` reddedildi; pressure-only auxiliary symbol shortcut'ı geri getirdi
