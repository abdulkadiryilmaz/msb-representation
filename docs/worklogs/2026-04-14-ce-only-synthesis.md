# Worklog: 2026-04-14 — CE-Only Synthesis: Expectation vs Current Readout

**Status**: Complete ✓  
**Goal**: `CE-only` baseline için başlangıç beklentisini, bugün itibarıyla kanıtlananları ve hâlâ açık kalan soruları tek notta toplamak

---

## Başlangıçta Beklentimiz Neydi

`CE-only` için başlangıç beklenti çıtası bilinçli olarak sınırlıydı.

Asıl soru şuydu:

- `bullish / bearish / intact` state-style supervision, Stage 1A için çalışan bir ilk latent düzen kurmaya yeter mi?

Bu aşamada beklediğimiz şeyler:

- model tamamen collapse etmesin
- `bullish` ve `bearish` kendi aralarında ayrışabilsin
- multi-symbol `core4` üstünde tek coin ezberiyle yürüyen kırılgan bir çözüm olmasın
- latent analysis için anlamlı bir referans çizgisi oluşsun

Beklemediğimiz şeyler:

- temsil kalitesinin nihai olarak çözülmüş olması
- tüm hard-case türlerinde net semantik ayrışma
- `z_long` branch'in ilk sürümde mükemmel davranması

Başka deyişle:

- `CE-only` bir son çözüm değil
- çalışan ve okunabilir bir baseline olmalıydı

## Bugün İtibarıyla Ne Kanıtlandı

### 1. Classifier baseline gerçekten çalışıyor

Val:

- accuracy yaklaşık `0.8885`
- macro-F1 güçlü

Test:

- accuracy yaklaşık `0.8879`

Bu çok önemli çünkü:

- sonuç sadece val içinde iyi görünmüyor
- OOS test'te de korunuyor

### 2. Ana hata modu yön karışıklığı değil

Bugünkü tablo açık biçimde şunu söylüyor:

- `bullish` ve `bearish` birbirine çok az karışıyor
- ana problem `intact <-> break` sınırı

Bu hem confusion yapısında hem bucket readout'ta tekrarlandı.

### 3. Latent geometry rastgele değil

Val tarafında:

- mean neighbor label agreement `0.8437`
- PCA ilk iki bileşende toplam varyansın yaklaşık `%93`'ünü taşıyor

Bu, latent uzayın organize olduğunu ve state bilgisi taşıdığını gösteriyor.

### 4. `z_short` ana semantik taşıyıcı

Hem val hem test'te:

- `z_short` label agreement yüksek
- `z_short` symbol agreement görece düşük

Bu nedenle ilk sürüm için şu yorum savunulabilir:

- local structure semantiğini esas olarak short branch öğrenmiş

### 5. `z_long` daha çok symbol/context taşıyor

Hem val hem test'te:

- `z_long` label agreement zayıf
- `z_long` symbol agreement çok yüksek

Bu, long branch'in ilk sürümde broader semantic context yerine daha çok coin/rejim shortcut'una yaslandığını düşündürüyor.

### 6. Güçlü break örnekleri temiz

Bucket readout'a göre:

- `true_break_up`
- `true_break_down`

bucket'ları hem val hem test'te çok temiz.

Bu da şu anlama geliyor:

- model gerçek teyitli structural shift state'lerini öğrenmiş

## Şu Anda En Güçlü Sonuç

Bugün en güçlü biçimde söyleyebildiğimiz şey şu:

- `CE-only` baseline yalnızca "çalıştı" değil
- aynı zamanda dürüst, stabil ve analize değer bir referans çizgisi oldu

Bu baseline:

- val ve test'te benzer davranıyor
- güçlü break state'lerini iyi ayırıyor
- multi-symbol setup'ta tamamen dağılmıyor
- sonraki objective denemeleri için ciddi bir benchmark veriyor

## Hâlâ Ne Bilmiyoruz

### 1. PCA / UMAP görsel cluster yorumu henüz manuel yapılmadı

Artefaktlar üretildi.

Ama henüz:

- scatter üstünde görsel cluster incelemesi
- sınıf / symbol / bucket katmanlı manuel yorum

yapılmadı.

Yani elimizde koordinatlar var, fakat görsel okuma henüz yüzeysel değil; daha çok sayısal/NN tabanlı.

### 2. Nearest-neighbor örneklerinin derin manuel okuması eksik

NN çıktıları hazır.

Ama henüz:

- örnek bazında pencereyi açıp fiyat davranışını karşılaştırma
- "bu iki komşu gerçekten aynı yapıyı mı taşıyor?" sorusunun manuel teyidi

yapılmadı.

Bu nedenle semantik komşuluk hakkında güçlü sayısal sinyalimiz var, ama tam nitel örnek incelemesi sınırlı.

### 3. `z_long` neden symbol-heavy?

Bunu gözlemledik.

Ama tam nedenini henüz ayırmadık:

- long window doğası mı
- feature ailesi mi
- objective mi
- coin identity shortcut mu

Bu hâlâ açık araştırma sorusu.

### 4. Downstream usefulness henüz kanıtlanmadı

`z_fused` usable görünüyor.

Ama henüz:

- probe
- transfer
- coin-held-out
- başka downstream görev

üstünden gerçek faydası ölçülmedi.

### 5. Hard-case semantiği iyileştirildi mi sorusuna hâlâ "kısmen" cevap verebiliyoruz

Bugün hard-case'leri daha iyi tanımlıyoruz.

Ama henüz:

- aynı bucket'larda ikinci bir objective ile iyileşme karşılaştırması

yapılmadı.

Yani problem bölgelerini biliyoruz; çözüm etkisini henüz görmedik.

## Şu Anda Doğru Sonuç Cümlesi

Bugünkü en dürüst sonuç cümlesi şu olur:

- `CE-only` Stage 1A için beklenenden güçlü, stabil ve açıklayıcı bir baseline verdi
- ana açık problem false-break / borderline intact-break sınırı
- short branch semantik olarak güçlü, long branch ise hâlâ symbol/context ağırlıklı

Bu, çok önemli bir fark yaratıyor:

- sonraki adım artık "baseline'ı kurtarmak" değil
- "nerede ve neden iyileştirmek istediğimizi" bildiğimiz bir noktadan ilerlemek

## Bundan Sonra

Eğer `CE-only` üstünde biraz daha kalınacaksa en değerli iki ek okuma:

- PCA / UMAP görsellerinin manuel cluster incelemesi
- nearest-neighbor örneklerinin fiyat-penceresi bazında manuel semantik kontrolü

Eğer objective tarafına geçilecekse:

- `SupCon` için gerekçe artık net
- hedef genel accuracy değil
- `z_long` ve `z_fused` tarafında semantik komşuluğu artırmak
- `borderline_intact_break` ve failed-break bucket'larında geometry'yi iyileştirmek

## Related

- `docs/worklogs/2026-04-13-ce-only-baseline-core4.md`
- `docs/worklogs/2026-04-14-ce-only-val-pca-nn-readout.md`
- `docs/worklogs/2026-04-14-ce-only-bucket-readout.md`
- `docs/worklogs/2026-04-14-ce-only-embedding-view-readout.md`
- `docs/worklogs/2026-04-14-ce-only-test-split-readout.md`

---

**Status**: Complete ✓
