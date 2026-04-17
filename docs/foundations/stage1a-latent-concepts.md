# Stage 1A Latent Concepts

Bu not, Stage 1A üzerine çalışırken tekrar tekrar dönen kavramları kısa ve kalıcı biçimde toplar.

Amaç:

- hangi hedefi optimize ettiğimizi unutmamak
- `label`, `latent`, `geometry`, `manifold`, `probe` gibi kavramları karıştırmamak
- Stage 1A ile sonraki fazların sınırını daha net hatırlamak

---

## Stage 1A'nin Çekirdek Hedefi

Stage 1A'in ana işi:

- mevcut pencerenin `current structural state`'ini semantik olarak taşıyan bir latent temsil öğrenmek

Pratik karşılığı:

- latent, pencerenin `intact / bullish / bearish` durumunu anlamlı biçimde organize edebilmelidir
- amaç doğrudan event timing veya tradable trigger üretmek değildir

Kısa ifade:

- Stage 1A, öncelikle bir `state representation` problemidir

---

## Mevcut Label Ne Taşıyor?

Mevcut `bullish / bearish / intact` label tasarımı:

- `event-style` değil
- `state-style` supervision verir

Bunun anlamı:

- label, "son MSB ne zaman oldu?" sorusunu cevaplamaz
- label, "bu pencerenin mevcut yapısal state'i nedir?" sorusunu cevaplar

Dolayısıyla aynı yapısal state:

- birkaç ardışık anchor boyunca tekrar edebilir

Bu davranış Stage 1A için bilinçli olarak kabul edilmiştir.

---

## Label Prediction, Latent Geometry, Manifold

Bu üç kavram aynı şey değildir.

### Label prediction

- modelin doğru etiketi tahmin etmesi
- örnek: `bullish / bearish / intact` sınıfını doğru söylemesi

Burada odak:

- loss
- accuracy / F1
- classifier head performansı

### Latent geometry

- örneklerin latent uzayda nasıl yerleştiği
- benzer yapısal örnekler yakın mı, farklı olanlar uzak mı sorusu

Burada odak:

- komşuluk yapısı
- sınıf ayrışması
- shortcut yerine semantik eksenlerde organizasyon

### Manifold

- latent geometry içindeki daha büyük, daha sürekli semantik yapı
- sadece "üç topak" değil, bu topakların iç yapısı ve aralarındaki geçişler

Pratik sezgi:

- `cluster` = benzer örnekler topaklanıyor mu
- `manifold` = bu topaklanma ve sınır yapısı gerçekten yapısal anlam taşıyor mu

---

## State Manifold Ne Demek?

Bu projede `state manifold` derken kastedilen:

- `bullish`, `bearish`, `intact` örneklerin latent uzayda semantik olarak organize olması

En kaba resim:

- `bullish` örnekler bir bölgede
- `bearish` örnekler bir bölgede
- `intact` örnekler bir bölgede

Ama bundan fazlası da beklenir:

- `clean intact`, `high-vol intact`, `borderline_intact_break` gibi alt yapılar aynı manifold içinde farklı alt alanlar oluşturabilir
- `close_confirmed_break` ve `true_break` gibi daha güçlü örnekler kendi bölgelerinde yoğunlaşabilir
- sınır örnekleri merkez örneklerden farklı yerde durabilir

Yani iyi bir state manifold:

- sadece sınıf ayrımı yapmaz
- sınıf içi varyasyonu ve sınır yapısını da taşır

---

## Borderline Neden Kritik?

`borderline_intact_break` gibi örnekler araştırma açısından kritik, çünkü:

- kolay örneklerde çoğu model iyi görünebilir
- asıl soru sınır bölgelerinde latent'in ne yaptığıdır

İyi bir latent'te borderline örnekler:

- `clean intact` merkezine gömülmemeli
- doğrudan `true break` bölgesine de çökmeden
- sınır bölgesinde anlamlı konumlanmalıdır

Bu yüzden borderline davranışı, manifold kalitesinin pratik testlerinden biridir.

Kısa yorum kuralı:

- kolay örnekler iyi, borderline kötü -> manifold var ama yeterince rafine değil
- her yer dağınık -> manifold çok zayıf veya yok

---

## Directional Pressure Nerede Anlamlı?

`Yapısal baskı` fikri en çok `intact` örneklerde anlamlıdır.

Sebep:

- `bullish` label zaten upside state'in gerçekleştiğini söyler
- `bearish` label zaten downside state'in gerçekleştiğini söyler
- yönlü baskı ise henüz break olmamış ama bir yöne eğilim taşıyan pencerelerde aranır

Bu nedenle mevcut Stage 1A label tasarımı:

- state'i doğrudan taşır
- ama `intact` içindeki `up-pressure / down-pressure / neutral` ayrımını doğrudan kodlamaz

Bu ayrım, Stage 1A'de probe edilmesi veya sonraki fazlarda açık hedefe dönüştürülmesi gereken semantiklerden biridir.

---

## Probe Neden Kullanılır?

Training history bize şunu söyler:

- ana model kendi optimize ettiği görevde iyi mi?

Ama şu soruyu cevaplamaz:

- latent, optimize edilmemiş başka bir bilgiyi de taşıyor mu?

`Probe` burada devreye girer.

Probe mantığı:

1. ana modeli eğit
2. latent'leri dondur
3. latent üstünde küçük bir ayrı model eğit
4. ilgilenilen bilginin latent'ten okunup okunamadığına bak

Bu yüzden probe:

- yeni ana model kurmak değildir
- mevcut representation'ın içinde hangi bilginin erişilebilir olduğunu test etmektir

---

## Linear Probe Neden Önemli?

`Linear probe`, ilgili bilginin latent'te ne kadar temiz ve kolay okunabilir olduğunu test eder.

Eğer linear probe iyi çalışıyorsa:

- bilgi latent'te düzenli biçimde yerleşmiş olabilir
- ayrım lineer olarak okunabilir durumdadır

Eğer linear probe zayıf ama küçük MLP iyi ise:

- bilgi tamamen kayıp olmayabilir
- ama latent'te daha dağınık veya dolanık biçimde taşınıyor olabilir

Kısa okuma:

- `linear iyi` -> temiz geometri sinyali
- `linear zayıf, MLP iyi` -> bilgi var ama dağınık
- `ikisi de zayıf` -> bilgi çok zayıf veya pratikte okunamıyor

---

## Pressure Probe Nasıl Yorumlanmalı?

`intact` örnekler içinde `up-pressure / down-pressure / neutral` gibi analiz etiketleri tanımlanıp latent üstünde probe edilebilir.

Bu probe'un amacı:

- yeni ana hedef eklemek değil
- mevcut Stage 1A latent'inin bu ince semantiği zaten taşıyıp taşımadığını görmek

Sonuçları yorumlama çerçevesi:

- `linear iyi` -> pressure bilgisi latent'te temiz şekilde var
- `MLP iyi, linear zayıf` -> bilgi var ama iyi organize değil
- `ikisi de zayıf` -> pressure bilgisi mevcut representation'da zayıf

Bu son durum tek başına otomatik olarak "Stage 1A başarısız" demek değildir.

Karar bağlamı:

- ana state ayrımı iyiyse, pressure konusu Stage 1B'ye bırakılabilir
- ana zorluk zaten `intact <-> break` sınırındaysa, bu eksik bilgi Stage 1A refinement gerekçesi olabilir

---

## Stage 1A ve Sonraki Fazlar Ayrımı

Pratik ayrım şu şekilde hatırlanabilir:

- `Stage 1A`:
  - current structural state
  - state manifold
  - semantik latent organization

- `Stage 1B+`:
  - transition / event freshness
  - future structure behavior
  - tradability / continuation / failure

Başka deyişle:

- Stage 1A daha çok `neredeyiz?`
- sonraki fazlar daha çok `nereye gidiyoruz?` ve `bu hareket ne kadar değerli?`

---

## Kısa Hatırlatma

Bu not için en kısa özet:

- mevcut label `state-style`'dır
- Stage 1A'nin ana işi `state manifold` kurmaktır
- borderline örnekler manifold kalitesinin en kritik testlerinden biridir
- `directional pressure` özellikle `intact` içinde aranır
- probe, latent'in optimize edilmemiş bir semantiği taşıyıp taşımadığını anlamak için kullanılır
- linear probe iyi ise, ilgili bilgi latent'te daha temiz yerleşmiş demektir
