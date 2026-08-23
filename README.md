# Turkmodel — Türkçe/İngilizce LLM Eğitim Deneyleri

Bu depo, sıfırdan Llama-tarzı dense Transformer eğitimi için kod, Colab akışları ve sınırlı bir H100 koşusunun özetini içerir. Dosya ve klasör adlarındaki `350M` / `8B` ifadeleri tarihsel hedeflerdir; kaydedilmiş sonuç dosyasındaki gerçek koşu **6,082,670,592 parametreli** model içindir.

> **Kapsam:** Buradaki “trained” etiketi yalnızca bir eğitim koşusu özeti bulunduğunu söyler. Model kalitesi, üretime hazır olma veya checkpoint'in bu GitHub deposunda mevcut olduğu anlamına gelmez.

## Durum Etiketleri

- **trained:** Commit edilmiş bir koşu özeti/final training loss var.
- **experimental:** Çalıştırılabilir kod veya notebook var; tamamlanmış koşu kanıtı yok.
- **planned:** Tasarım, ölçeklendirme hedefi veya tahmin var; koşu yok.

## Model ve Deney Envanteri

| Çalışma | Durum | Gerçekte ne var? | Kanıt / eksik |
| --- | --- | --- | --- |
| **TR-EN 6.08B, 400-step H100 koşusu** | **trained — sınırlı, metadata-only** | 32 katman, hidden 4096, 32 attention / 8 KV head, vocab 50,257, context 2,048 | [`results/training_info.json`](results/training_info.json) ve [`TRAINING_RESULTS.md`](llm-8b-tr-en/TRAINING_RESULTS.md) var; ağırlık/checkpoint bu GitHub deposunda yok |
| **`LLM_Egitimi_350M.ipynb`** | **experimental** | Dosya adına rağmen hücreler yaklaşık 6.08B model oluşturacak şekilde ayarlı | Notebook'ta commit edilmiş execution output yok |
| **350M pipeline** | **experimental** | `train_from_scratch.py --model-size 350m` yapılandırması | Ayrı koşu sonucu/checkpoint yok |
| **1B pipeline** | **experimental** | `train_from_scratch.py --model-size 1b` yapılandırması | Ayrı koşu sonucu/checkpoint yok |
| **8B pipeline / ölçeklendirme** | **planned** | Kod yapılandırması ve zaman/token hedefleri | Eğitim sonucu yok; klasör adı mevcut 6B config ile aynı değil |
| **Qwen3.5 / büyük MoE brifingi** | **planned** | [`Sifirdan_LLM_Egitimi_Copilot.md`](Sifirdan_LLM_Egitimi_Copilot.md) içinde tasarım notları | Uygulama veya koşu sonucu değil |

## Kaydedilmiş 6.08B Koşusu

Kaynak: [`results/training_info.json`](results/training_info.json). Türetilen değerler aşağıda ayrıca işaretlenmiştir.

| Metrik | Değer | Not |
| --- | ---: | --- |
| Parametre | 6,082,670,592 | Koşu metadata'sı |
| GPU | 1× NVIDIA H100 80GB HBM3 | Koşu metadata'sı |
| Training step | 400 | Kısa PoC bütçesi |
| İşlenen token | 13,107,200 | Config hesabı: `400 × 4 batch × 4 grad accumulation × 2,048` |
| Duvar saati | 66.018 dakika | Koşu metadata'sı |
| GPU-saat | 1.100 | 1 GPU × duvar saati; türetilmiş |
| Final training loss | 4.78736 | Son loglanan training loss; held-out eval değildir |
| Training perplexity | ≈119.98 | `exp(4.78736)`; türetilmiş, held-out değildir |
| Ortalama throughput | ≈3,309 token/s | Config tokenı / duvar saati; türetilmiş |
| Sistem RAM | Kaydedilmedi | Sonuç dosyasında yok |
| Peak VRAM | Kaydedilmedi | H100 kapasitesi peak kullanım değildir |
| Eval loss / eval perplexity | Kaydedilmedi | Kalite karşılaştırması yapılamaz |

Bu koşu yaklaşık **2.15 token/parametre** görmüştür. Bu nedenle model boyutuna göre çok kısa bir smoke/PoC koşusudur; kullanılabilir dil modeli kalitesi iddiası için yeterli değildir.

## Mimari

Kaydedilmiş koşu ve [`model_config.json`](llm-8b-tr-en/configs/model_config.json):

| Alan | Değer |
| --- | ---: |
| Model türü | Llama-style dense causal LM |
| Vocab | 50,257 |
| Hidden size | 4,096 |
| Intermediate size | 11,008 |
| Katman | 32 |
| Attention head | 32 |
| KV head | 8 (GQA) |
| Context | 2,048 |
| Precision | BF16 |
| Attention backend | FlashAttention 2 olarak yapılandırılmış |
| Optimizer | 8-bit AdamW olarak yapılandırılmış |

## Repo Yapısı

```text
LLM_Egitimi_350M.ipynb          # Tarihsel isimli; mevcut hücreler ~6.08B config kullanıyor
Sifirdan_LLM_Egitimi_Copilot.md # Plan/tasarım brifingi
results/training_info.json      # Kaydedilmiş 400-step koşu özeti
llm-8b-tr-en/
├── configs/                    # Mevcut ~6B model ve kısa PoC training config'i
├── data/                       # İndirme, temizleme, dedup ve karışım scriptleri
├── tokenizer/                  # Özel tokenizer eğitim scripti
├── training/                   # Genel trainer ve tek-dosya Colab akışı
├── evaluation/                 # Perplexity ve küçük Türkçe prompt testleri
└── TRAINING_RESULTS.md         # Kısa koşu özeti
```

## Çalıştırma

```bash
git clone https://github.com/cebrailbagatarhan/turkmodel.git
cd turkmodel/llm-8b-tr-en
pip install -r requirements.txt
```

Genel eğitim scripti:

```bash
# Bu presetler için depoda ayrı sonuç bulunmuyor.
python training/train_from_scratch.py --model-size 350m --no-wandb
python training/train_from_scratch.py --model-size 1b --no-wandb
python training/train_from_scratch.py --model-size 8b --no-wandb
```

Kaydedilmiş 6.08B kısa koşusuna karşılık gelen tek-dosya Colab akışı:

```bash
python training/colab_setup.py
```

Script; H100/A100, veri erişimi ve harici servis kimlik bilgileri bekler. API tokenlarını kaynak koda yazmayın; ortam değişkenleri veya Colab Secrets kullanın.

## Sonuçları “trained” Olarak Yayınlama Kontrol Listesi

Yeni bir koşuyu tabloya eklemeden önce:

1. Notebook/script çıktısını ve tam config'i commit edin.
2. Seed, commit SHA, veri kaynakları, lisansları, split ve dedup bilgisini kaydedin.
3. Gerçek optimizer step ve padding hariç gerçek token sayısını raporlayın.
4. Wall time, GPU-saat, tokens/s, sistem RAM ve peak VRAM ölçün.
5. Train loss ile held-out eval loss/perplexity'yi ayırın.
6. Türkçe benchmark sonuçlarını baseline ile karşılaştırın.
7. İndirilebilir checkpoint/model kartı bağlantısını doğrulayın.
8. Kesintili/başarısız koşuları `experimental` olarak bırakın.

## Bilinen Tutarsızlıklar

- `LLM_Egitimi_350M.ipynb` adı, hücrelerde oluşturulan yaklaşık 6.08B modelle eşleşmiyor.
- `llm-8b-tr-en/` klasör adı, mevcut `model_config.json` içindeki yaklaşık 6.08B modelle eşleşmiyor.
- Nested README'deki 350M ve “beklenen” throughput/token tabloları kaydedilmiş 6.08B koşusunun sonucu değildir.
- Notebook'ta execution output bulunmadığı için `training_info.json` değerleri notebook çıktısıyla bağımsız olarak doğrulanamıyor.
- Bu GitHub deposunda model ağırlıkları/checkpoint yok.
- Kök dizinde `LICENSE` dosyası yok; yeniden kullanım koşulları belirtilmemiş.

## Lisans

Bu depoda şu anda bir `LICENSE` dosyası bulunmuyor. Yeniden kullanım/dağıtım için açık bir lisans eklenmelidir.

## Maintainer

[@cebrailbagatarhan](https://github.com/cebrailbagatarhan)
