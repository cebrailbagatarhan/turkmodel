# 6.08B / 400-Step H100 PoC Sonucu

> Bu dosya sınırlı bir eğitim koşusunu raporlar. Model kalitesi, üretime hazır olma veya 350M/8B presetlerinin eğitildiği anlamına gelmez.

Kanonik koşu özeti: [`../results/training_info.json`](../results/training_info.json).

| Metrik | Değer | Kanıt türü |
| --- | ---: | --- |
| Parametre | 6,082,670,592 | Public koşu metadata'sı |
| Mimari | Llama dense causal LM | Public config/script |
| GPU | 1× NVIDIA H100 80GB HBM3 | Public koşu metadata'sı |
| Step | 400 | Public koşu metadata'sı |
| Süre | 66.018 dakika | Public koşu metadata'sı |
| GPU-saat | ≈1.100 | Türetilmiş |
| İşlenen token | 13,107,200 | Public metadata/config hesabı |
| Ortalama throughput | ≈3,309 token/s | Türetilmiş |
| Final training loss | 4.78736 | Public koşu metadata'sı |
| Training perplexity | ≈119.98 | `exp(4.78736)`; türetilmiş |
| Eğitim tokenı / parametre | ≈0.00215 | `13.1M / 6.08B`; türetilmiş |
| Eval loss / perplexity | Kaydedilmedi | Public sonuçta yok |
| Sistem RAM / peak VRAM | Kaydedilmedi | Public sonuçta yok |

## Yorum

400 step ve yaklaşık 0.00215 token/parametre, 6.08B model için son derece kısa bir smoke/PoC bütçesidir. Loss'un düşmesi eğitim döngüsünün çalıştığını gösterir; dil kalitesini veya genellemeyi kanıtlamaz.

## Artifact Durumu

Bu GitHub deposunda model ağırlıkları, checkpoint, checksum veya tamamlanmış model kartı yoktur. Daha önceki doğrulanmamış Hugging Face indirme komutu bu dosyadan çıkarılmıştır. İndirme talimatı ancak public artifact gerçekten erişilebilir olduğunda ve checksum ile doğrulandığında eklenmelidir.

## Yayınlanması Gereken Ek Kanıt

- Tam `trainer_state.json` veya sanitize edilmiş training log
- Final eval loss/perplexity ve bağımsız benchmark
- Peak allocated/reserved VRAM ve sistem RAM
- Seed, commit SHA, dataset manifesti, split ve dedup özeti
- Public checkpoint/model kartı ve checksum
