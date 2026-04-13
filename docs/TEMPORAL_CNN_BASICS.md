# Temporal CNN Basics

Bu not, Stage 1A mimari tartışmasında geçen temel kavramları kısa ve tekrar bakılabilir şekilde açıklar:

- `1D CNN`
- `Temporal CNN / TCN`
- `receptive field`
- `dilation`
- `residual block`

## 1D CNN vs Temporal CNN

`1D CNN`, teknik operasyon adıdır:

- veri bir eksen boyunca akar
- convolution filtresi bu eksen üzerinde kayar

Zaman serisinde bu eksen genellikle `time` olur.

`Temporal CNN` veya `TCN` ise, zaman serisi için daha bilinçli tasarlanmış 1D CNN ailesidir.

Tipik özellikleri:

- zaman ekseni üstünde çalışır
- receptive field'i kontrollü büyütür
- çoğu zaman residual block kullanır
- çoğu zaman dilation kullanır

Kısaca:

- her TCN bir tür 1D CNN'dir
- ama her 1D CNN, TCN tarzında tasarlanmış değildir

## Receptive Field

`Receptive field`, modelin bir çıkış üretirken geçmişte kaç zaman adımını "görebildiği" demektir.

Zaman serisi açısından soru şu olur:

- bir latent özelliği üretirken model kaç barlık geçmişten etkileniyor?

Örnek:

- kernel size = 3 olan tek bir conv katmanı
- her çıktı noktası yalnızca 3 komşu barı görür

Bu durumda receptive field küçüktür.

Birden çok katman üst üste gelince receptive field büyür:

- ilk katman küçük lokal pattern'leri görür
- sonraki katmanlar daha geniş zaman aralığını kapsar

Bu neden önemli:

- kısa receptive field -> wick, body, küçük swing gibi lokal yapı iyi yakalanır
- büyük receptive field -> daha geniş formasyon, birkaç saatlik veya birkaç günlük yapı görülebilir

MSB gibi problemde ikisi de gerekir:

- lokal kırılım davranışı
- biraz daha geniş yapı bağlamı

## Dilation

`Dilation`, convolution filtresinin ardışık barlara değil, aralıklı barlara bakmasıdır.

Normal conv:

- kernel 3 ise örneğin `[t-2, t-1, t]` gibi komşu barlara bakar

Dilated conv:

- dilation = 2 ise `[t-4, t-2, t]`
- dilation = 4 ise `[t-8, t-4, t]`

Yani filtre "seyrek örnekleme" ile daha geniş geçmişe bakar.

Faydası:

- çok fazla katman eklemeden receptive field büyür
- uzun menzilli örüntüler daha verimli görülebilir

Risk:

- fazla agresif dilation mikro detay kaybına yol açabilir
- kısa vadeli yapı için fazla seyrek bakmak zararlı olabilir

Bu yüzden:

- short branch'te dilation daha temkinli
- long branch'te daha rahat kullanılabilir

## Residual Block

Normal blok:

- `x -> bazı katmanlar -> F(x)`

Residual block:

- giriş doğrudan korunur
- blok çıkışı işlenmiş temsil ile toplanır

Formül:

- `output = x + F(x)`

Buradaki fikir:

- model sıfırdan her şeyi yeniden yazmak yerine
- mevcut temsile hangi düzeltmeyi / ek bilgiyi yapacağını öğrenir

Faydaları:

- derin ağları daha stabil eğitirsin
- gradient akışı iyileşir
- erken katmanlardaki faydalı bilgi kaybolmaz
- model gerekirse çok küçük değişiklik yapabilir

Zaman serisinde neden yararlı:

- ilk katmanlar lokal candle yapısını yakalar
- üst katmanlar swing / displacement / bağlam örüntüsü ekler
- residual bağlantı sayesinde alt seviye bilgi tamamen silinmez

## Bu Kavramlar Stage 1A'e Nasıl Bağlanır

Stage 1A için temporal conv ailesi düşünülürken amaç:

- short branch'te lokal yapıyı iyi görmek
- long branch'te daha geniş bağlamı verimli görmek

Bu yüzden:

- receptive field bilinçli büyümeli
- dilation dikkatli kullanılmalı
- residual block ile derinlik stabil tutulmalı

Kısa sezgisel çeviri:

- `receptive field` = model kaç barlık geçmişe bakıyor
- `dilation` = bu geçmişe ne kadar seyrek aralıklarla bakıyor
- `residual block` = mevcut temsili koruyup üstüne düzeltme ekliyor
