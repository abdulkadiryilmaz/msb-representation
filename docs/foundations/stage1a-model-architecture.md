# Stage 1A Model Architecture Reference

**Created**: 2026-05-17  
**Scope**: Stage 1A model bileşenleri, loss hesabı ve gradient akışı

---

## Genel Çerçeve

Stage 1A bir sınıflandırıcı değil, bir temsil öğrenen sistemdir. `bullish / bearish / intact` label'ları doğrudan sınıflandırma amacıyla değil, latent uzayın geometrisini şekillendirmek için kullanılır. Asıl çıktı `z_fused` vektörüdür — downstream Stage 1B / Stage 2 görevleri bunu tüketir.

---

## Input

İki pencere, ikisi de aynı anchor bar'a sağ hizalı:

```
Short window: 48 bar × 15m = 12h  → [batch, channels, 48]
Long window:  288 bar × 15m = 72h → [batch, channels, 288]
```

---

## Encoder'lar

İki ayrı `TemporalConvEncoder` — yapısal olarak aynı, ağırlıkları bağımsız:

```
Conv1d stem → 4 residual blok (dilation 1, 2, 4, 8) → AdaptiveAvgPool1d → Linear
```

Pool, zaman boyutunu tek vektöre sıkıştırır. Pencere uzunluğundan bağımsız olarak her encoder tek bir vektör üretir:

```
short_encoder([batch, channels, 48])  → z_short: [batch, 64]
long_encoder([batch, channels, 288])  → z_long:  [batch, 32]
```

Concat ile birleştirme:

```
z_fused = cat([z_short, z_long]) → [batch, 96]
```

---

## Head'ler ve Ürettikleri

Forward pass sonunda model bir dict döndürür. Temel olanlar her zaman üretilir, diğerleri config'e göre eklenir:

| Key | Input | Shape | Kullanım |
|---|---|---|---|
| `z_short` | short encoder çıktısı | [batch, 64] | diagnostic |
| `z_long` | long encoder çıktısı | [batch, 32] | diagnostic + aux head |
| `z_fused` | concat | [batch, 96] | downstream temsil |
| `z_proj` | projection_head(z_fused) | [batch, 64] | (eski SupCon key'i) |
| `logits` | classifier(z_fused) | [batch, 3] | fused CE loss |
| `z_long_proj` | long_projection_head(z_long) | [batch, 64] | SupCon loss |
| `long_aux_logits` | long_aux_classifier(z_long) | [batch, 3] | aux CE loss |

Conditional head'ler `model.py`'de şu şekilde eklenir:

```python
outputs = {"z_short": z_short, "z_long": z_long, "z_fused": z_fused, ...}

if self.long_projection_head is not None:
    outputs["z_long_proj"] = self.long_projection_head(z_long)

if self.long_aux_classifier is not None:
    outputs["long_aux_logits"] = self.long_aux_classifier(z_long)
```

---

## Loss Hesabı

Mevcut aktif konfigürasyon (`branch_ce_aux_v1`):

```python
loss = 1.00 * ce_loss
     + 0.05 * supcon_loss
     + 0.10 * long_aux_loss
```

**ce_loss**: `CrossEntropy(logits, y)` — z_fused üzerinden classifier çıktısı ile gerçek label karşılaştırılır.

**supcon_loss**: `z_long_proj` üzerinden hesaplanır. Aynı label'daki örnekler yaklaştırılır, farklı label'dakiler uzaklaştırılır.

**long_aux_loss**: `CrossEntropy(long_aux_logits, y)` — yalnızca z_long üzerinden classifier çıktısı ile gerçek label karşılaştırılır.

Her loss tek bir batch ortalaması scalar'ıdır.

---

## Gradient Akışı

`loss.backward()` hesap grafiğini geriye doğru gezerek her parametrenin gradient'ını hesaplar. Gradient yalnızca o değerin hesaplanmasında kullanılan parametrelere akar.

```
ce_loss        → classifier → z_fused → short_encoder + long_encoder
supcon_loss    → long_projection_head → z_long → long_encoder
long_aux_loss  → long_aux_classifier → z_long → long_encoder
```

Her parametre için:

| Bileşen | ce_loss | supcon_loss | long_aux_loss |
|---|:---:|:---:|:---:|
| `short_encoder` | ✓ | — | — |
| `long_encoder` | ✓ | ✓ | ✓ |
| `classifier` | ✓ | — | — |
| `long_projection_head` | — | ✓ | — |
| `long_aux_classifier` | — | — | ✓ |

`long_encoder` üç kaynaktan gradient alıyor. `optimizer.step()` bunları toplu uygular:

```
∂loss/∂w = 1.00 * ∂ce_loss/∂w
          + 0.05 * ∂supcon_loss/∂w
          + 0.10 * ∂long_aux_loss/∂w
```

`short_encoder` yalnızca `ce_loss`'tan gradient alır — diğer loss terimleri ona ulaşmaz.

---

## Tasarım Kararı: long_aux_classifier neden z_long'dan çalışıyor

`long_aux_classifier`'ı z_fused üzerine bağlamak mümkündü. Ancak bu durumda `long_aux_loss` hem `short_encoder`'a hem `long_encoder`'a gradient taşırdı. Amaç yalnızca `long_encoder`'ı doğrudan zorlamak — z_long'un label-dominant bilgi taşımasını garantilemek — olduğundan, head yalnızca `z_long` üzerinden bağlandı.
