# Multi-Scale MSB Interpretation

Bu not, Stage 1A worklog'undan ayrı olarak yapılan teknik tartışmayı saklar. Amaç, `MSB` kavramının ölçeğe bağımlı doğasını ve kısa/uzun pencereyle aynı olayın neden farklı okunabildiğini belgelemektir.

## Ana Fikir

MSB tek ve ölçekten bağımsız bir nesne değildir. Aynı fiyat hareketi:

- kısa pencerede `bullish MSB`
- daha geniş pencerede `yalnızca toparlanma`
- daha da geniş bağlamda `relief rally`

olarak görünebilir.

Bu bir çelişki değil, yapı ölçeğinin değişmesidir.

## Üç Katmanlı Okuma

### 1. Local MSB

Soru:

- son saatler / son 12 saat içinde en yakın anlamlı swing kırıldı mı?

Bu katmanda görülen şey:

- wick vs close davranışı
- kısa vadeli displacement
- lokal swing high / low ihlali

Bu, kısa pencerenin taşıdığı yapıdır.

### 2. Intermediate MSB

Soru:

- son 2-3 günlük yapıda gerçekten anlamlı yön değişimi oldu mu?

Bu katmanda görülen şey:

- birkaç günlük swing yapısı
- lokal kırılımın daha büyük yapı içindeki yeri
- hareketin recovery leg mi, gerçek structure shift mi olduğu

Bu, uzun pencerenin taşıdığı yapıdır.

### 3. Macro Context

Soru:

- bu hareket daha büyük resimde neyin içinde oluyor?

Bu katmanda görülen şey:

- trend / range zemini
- relief rally mi continuation mı
- mevcut lokal kırılımın daha büyük rejime göre anlamı

Macro context, olayı isimlendirmekten çok anlamlandırır.

## NEAR Örneği

`2026-04-11` tarihindeki yukarı hareket, kısa ve uzun pencerede farklı tonlarda okunabilir:

- `local`: bullish structure shift / breakout leg gibi görünebilir
- `intermediate`: daha geniş yapıda henüz tam trend dönüşü olmayabilir
- `macro`: daha büyük düşüş veya dalgalı rejim içinde güçlü bir toparlanma da olabilir

Dolayısıyla şu cümle teknik olarak tutarlıdır:

- "Lokal bullish MSB var, ama intermediate düzeyde henüz güçlü reversal teyidi yok."

## Neden Önemli

Bu not şu kafa karışıklığını çözer:

- "Aynı hareket bazı pencerelerde MSB, bazılarında değil. Bu bir sorun mu?"

Cevap:

- hayır
- çünkü MSB her zaman seçilen yapı ölçeğine göre tanımlanır

Bu yüzden tek pencereyle çalışan sert label mantığı, gözün çok ölçekli olarak gördüğü yapıyı her zaman tam yansıtmayabilir.

## Projeye Etkisi

Bu tartışma şu tasarım kararını destekler:

- short branch local structure'ı taşımalı
- long branch broader context'i taşımalı
- downstream karar, bu iki ölçeğin birleşiminden gelmeli

Başka deyişle:

- short tek başına "setup şimdi neye benziyor?" sorusunu cevaplar
- long ise "bu setup hangi hikayenin içinde oluşuyor?" sorusunu cevaplar

Bu yaklaşım, `aynı anı daha geniş bağlamda temsil etmek` ilkesinin doğrudan motivasyonudur.
