# CE + SupCon Experiments

Bu klasör, `CE + SupCon` deney planı ve sonuç okumalarını toplar.

## Timeline

1. `fused projection v1`
   - mixed
   - `z_long` semantik olarak iyileşti ama symbol bias da büyüdü

2. `long projection v1`
   - previous best
   - `z_long` ve `z_fused` tarafında güçlü referans geometry verdi, fakat daha sonra seed-sensitive olduğu görüldü

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

9. `long short-pressure aux v1`
   - rejected
   - pressure head `z_short`a izole edildi; `z_short` pressure iyileşti ama `z_long` / `z_long_proj` geometry çöktü

10. `long v1 seed stability`
    - seed-sensitive
    - `seed_42` referans geometry'ye yakın kaldı, fakat `seed_41` ve `seed_43` `z_long` / `z_long_proj` tarafında symbol-heavy hale geldi

11. `long frontload v1`
    - useful / not fully seed-stable
    - frontloaded SupCon `seed_42`yi bozmadı ve `seed_43`ü toparladı; `seed_41` ise hâlâ symbol-heavy kaldı

12. `long SupCon-only warmup v1`
    - diagnostic
    - erken SupCon platosunu kırdı ve `z_long`u düzeltti; hard CE=0 warmup `z_short` / `z_fused` dengesini bozdu

13. `long SupCon-dominant warmup v1`
    - best balanced schedule candidate
    - seed 41 için `epoch_010.pt`, seed 42 için `epoch_040.pt` seçildi; iki seed'de de `z_long` temizlendi ve branch dengesi korundu

14. `long branch CE aux v1`
    - selected for downstream transfer
    - `seed_41/epoch_020.pt` Stage 1A aktarım adayı seçildi; seed 42 global metriklerde dengeli ama borderline-up asimetrisi sonraki model iyileştirme konusu

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
14. `2026-05-13-ce-supcon-long-short-pressure-aux-v1-plan.md`
15. `2026-05-13-ce-supcon-long-short-pressure-aux-v1-readout.md`
16. `2026-05-16-ce-supcon-long-v1-seed-stability-plan.md`
17. `2026-05-16-ce-supcon-long-v1-seed-stability-readout.md`
18. `2026-05-16-ce-supcon-long-frontload-v1-plan.md`
19. `2026-05-16-ce-supcon-long-frontload-v1-readout.md`
20. `2026-05-16-ce-supcon-long-supcon-warmup-v1-plan.md`
21. `2026-05-17-ce-supcon-long-supcon-warmup-v1-readout.md`
22. `2026-05-17-ce-supcon-long-supcon-dominant-warmup-v1-plan.md`
23. `2026-05-17-ce-supcon-long-supcon-dominant-warmup-v1-readout.md`
24. `2026-05-17-ce-supcon-long-branch-ce-aux-v1-plan.md`
25. `2026-05-17-ce-supcon-long-branch-ce-aux-v1-readout.md`

Kısa sonuç:

- fused-projection SupCon karışık sonuç verdi
- branch-aware `long v1` güçlü referans ama seed-sensitive
- `long v2` ve ilk symbol-aware deneme reddedildi
- `long symbol v2` reddedildi
- `long factor v1` reddedildi; pressure sinyali güçlendi ama symbol shortcut guardrail'i bozuldu
- `long aux v1` promising ablation; coarse SupCon pairing korununca symbol guardrail bozulmadı, fakat aux ağırlıkları `z_long` geometry için hâlâ fazla olabilir
- `long aux v2` reddedildi; pressure-only auxiliary symbol shortcut'ı geri getirdi
- `long short-pressure aux v1` reddedildi; direct pressure gradient izolasyonu tek başına `z_long` geometry'yi korumadı
- `long v1 seed stability` sonucunda `ce_supcon_long_v1` best observed checkpoint olarak kaldı, fakat setup seed-stable değil; branch usage stability çözülmeden Stage 1B varsayımlarına temel yapılmamalı
- `long frontload v1` useful ama tam seed-stable değil; erken SupCon aktivasyonu işe yaradı, ayrıca Stage 1A için fixed-budget + dense checkpoint + val geometry selection protokolü netleşti
- `long SupCon-only warmup v1` diagnostic success; SupCon activation problemini çözdü ama branch balance bozulduğu için final candidate değil
- `long SupCon-dominant warmup v1` şu ana kadarki en iyi dengeli schedule adayı; yine de seçilen epoch seed'e bağlı kaldığı için sonraki hat `z_long` auxiliary CE head gibi objective-level bir constraint olmalı
- `long branch CE aux v1` selected for downstream transfer; objective-level `z_long` CE head iyi bir yön ve `seed_41/epoch_020.pt` Stage 1A aktarım adayı olarak kullanılacak
