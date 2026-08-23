# TR-EN LLM Eğitim Pipeline'ı

> **Durum: experimental pipeline + sınırlı bir trained PoC.** Bu klasördeki `8B` adı tarihsel hedeftir. Commit edilmiş tek koşu özeti **6,082,670,592 parametreli**, 400-step bir H100 denemesine aittir; 350M veya 8B tamamlanmış koşu sonucu değildir.

Kök dizindeki [ana README](../README.md), model/deney envanteri ve “trained / experimental / planned” ayrımı için kanonik kaynaktır.

## Kanıtlanan Koşu

Herkese açık kaynaklar: [`../results/training_info.json`](../results/training_info.json), [`TRAINING_RESULTS.md`](TRAINING_RESULTS.md) ve [`training/colab_setup.py`](training/colab_setup.py).

| Metrik | Değer | Not |
| --- | ---: | --- |
| Parametre | 6,082,670,592 | `LlamaForCausalLM` config'i |
| GPU | 1× NVIDIA H100 80GB HBM3 | Koşu özeti |
| Step | 400 | Kısa PoC bütçesi |
| İşlenen token | 13,107,200 | `400 × 4 batch × 4 accumulation × 2,048` |
| Süre | 66.018 dakika | Koşu özeti |
| Final training loss | 4.78736 | Held-out eval değildir |
| Training perplexity | ≈119.98 | `exp(4.78736)`; türetilmiş |
| Ortalama throughput | ≈3,309 token/s | Token / duvar saati; türetilmiş |
| Eğitim tokenı / parametre | ≈0.00215 | Çok kısa bir smoke/PoC koşusu |
| Eval, RAM, peak VRAM | Public sonuçta yok | Kalite/kapasite iddiası yapılamaz |

Bu metrikler modelin eğitiminin başladığını ve 400 optimizer step tamamlandığını gösterir; kullanılabilir dil modeli kalitesini kanıtlamaz. Model ağırlıkları/checkpoint bu GitHub deposunda yer almıyor.

## Bu Klasördeki İki Ayrı Akış

| Akış | Gerçekte ne yapar? | Durum |
| --- | --- | --- |
| `training/colab_setup.py` | GPT-2 tokenizer ve yaklaşık 6.08B Llama config'iyle 400-step quick run | Bir sınırlı koşu özeti var |
| `training/train_from_scratch.py` | `350m`, `1b`, `8b` presetlerini sunan genel pipeline | Bu presetler için ayrı public koşu sonucu yok |

350M presetinin varlığı, 350M modelin eğitildiği anlamına gelmez. Aynı şekilde klasör adı 8B olsa da public sonuç 6.08B config içindir.

## Hızlı Başlangıç

```bash
pip install -r requirements.txt
```

Genel pipeline:

```bash
# Aşağıdaki presetler için bu depoda ayrı tamamlanmış sonuç yoktur.
python training/train_from_scratch.py --model-size 350m --no-wandb
python training/train_from_scratch.py --model-size 1b --no-wandb
python training/train_from_scratch.py --model-size 8b --no-wandb
```

Kaydedilmiş kısa koşuya karşılık gelen Colab akışı:

```bash
python training/colab_setup.py
```

Script H100/A100, harici veri erişimi ve servis kimlik bilgileri bekler. GitHub, Hugging Face, Kaggle veya WandB tokenlarını kaynak koda yazmayın; ortam değişkenleri ya da Colab Secrets kullanın.

## Veri Akışı

Manuel akış:

```bash
python data/download_datasets.py --mode quick
python data/clean_turkish.py --input ./data/raw_dataset --output ./data/cleaned_dataset
python data/dedup.py --input ./data/cleaned_dataset --output ./data/deduped_dataset
python data/create_data_mix.py --input ./data/deduped_dataset --output ./data/mixed_dataset
python tokenizer/train_tokenizer.py --vocab-size 32000
python data/tokenize_dataset.py \
  --input ./data/mixed_dataset \
  --tokenizer ./tokenizer \
  --output ./data/tokenized_dataset
```

`data/tokenize_dataset.py`, belgeleri önce `seed=42` ile train/validation olarak ayırır; her split'i daha sonra bağımsız tokenize edip paketler. Çıktıdaki `dataset_manifest.json`, tokenizer SHA-256 kimliği ve split ayarlarını kaydeder. Eğitim scripti bu kimliği doğrular ve uyuşmayan tokenizer ile başlamaz. Varsayılan belge karışımı %80 Türkçe, %15 İngilizce ve %5 koddur. `colab_setup.py` quick akışı ise GPT-2 tokenizer kullanır ve ayrı bir özel-tokenizer benchmarkı değildir.

## Mimari — Kaydedilmiş 6.08B Config

| Alan | Değer |
| --- | ---: |
| Vocab | 50,257 |
| Hidden | 4,096 |
| Intermediate | 11,008 |
| Katman | 32 |
| Attention / KV head | 32 / 8 |
| Context | 2,048 |
| Precision | BF16 |
| Attention | FlashAttention 2 kurulmuşsa onu, aksi halde SDPA |
| Optimizer | 8-bit AdamW olarak yapılandırılmış |

## Bilinen Sınırlar

- Eski “500M–1B token”, “50–100K token/s” ve benzeri değerler ölçülmüş sonuç değildi; bu README'den çıkarıldı.
- Public sonuçta eval loss/perplexity, sistem RAM ve peak VRAM yok.
- Checkpoint/model kartı public olarak doğrulanamıyor.
- Quick run'daki veri karışımı ve tokenizer, genel pipeline'ın tüm seçeneklerini temsil etmiyor.
- Kök dizinde açık bir `LICENSE` dosyası yok.
