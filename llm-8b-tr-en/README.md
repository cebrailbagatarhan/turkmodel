# 🧠 Sıfırdan LLM Eğitimi — Türkçe + İngilizce (2 Saat H100)

> **Model:** 350M parametre Dense Transformer (Llama mimarisi)  
> **Platform:** Google Colab — NVIDIA H100 80GB  
> **Süre:** ~2 saat  
> **Diller:** Türkçe (%40) + İngilizce (%55) + Kod (%5)

---

## ⚡ Hızlı Başlangıç (Colab)

**En kolay yol:** `training/colab_setup.py` dosyasını Colab'a yükleyin ve çalıştırın.  
Bu script her şeyi otomatik yapar: kurulum → veri indirme → tokenizasyon → eğitim → değerlendirme.

```
1. Google Colab'ı açın → Runtime > Change Runtime Type > H100
2. training/colab_setup.py içeriğini kopyalayın
3. Çalıştırın ve ~2 saat bekleyin
4. Model Google Drive'a kaydedilir ✓
```

---

## 📁 Proje Yapısı

```
llm-8b-tr-en/
├── README.md                          ← Bu dosya
├── requirements.txt                   ← Python bağımlılıkları
│
├── configs/
│   ├── model_config.json              ← 350M model parametreleri
│   └── training_config.yaml           ← Eğitim hyperparametreleri
│
├── data/
│   ├── download_datasets.py           ← HuggingFace'den veri çekme
│   ├── clean_turkish.py               ← Türkçe metin temizleme
│   ├── dedup.py                       ← MinHash deduplikasyon
│   ├── tokenize_dataset.py            ← BPE tokenizasyon
│   └── create_data_mix.py             ← Dil karışım oranları
│
├── tokenizer/
│   └── train_tokenizer.py             ← Özel BPE tokenizer eğitimi
│
├── training/
│   ├── train_from_scratch.py          ← Ana eğitim scripti
│   └── colab_setup.py                 ← Colab tek-dosya çözümü
│
└── evaluation/
    ├── turkish_bench.py               ← Türkçe kalite testi
    └── perplexity_eval.py             ← Perplexity ölçümü
```

---

## 🔧 Adım Adım Kullanım (Yerel / Manuel)

### 1. Kütüphane Kurulumu
```bash
pip install -r requirements.txt
```

### 2. Veri İndirme
```bash
# 2 saat H100 için (küçük örneklem)
python data/download_datasets.py --mode quick

# Tam eğitim için (tüm veri)
python data/download_datasets.py --mode full
```

### 3. Veri Temizleme
```bash
python data/clean_turkish.py --input ./data/raw_dataset --output ./data/cleaned_dataset
```

### 4. Deduplikasyon
```bash
python data/dedup.py --input ./data/cleaned_dataset --output ./data/deduped_dataset
```

### 5. Tokenizasyon
```bash
# Opsiyonel: Özel tokenizer eğit
python tokenizer/train_tokenizer.py --vocab-size 32000

# Veriyi tokenize et
python data/tokenize_dataset.py --input ./data/deduped_dataset --output ./data/tokenized_dataset
```

### 6. Eğitim
```bash
# 350M model (2 saat H100)
python training/train_from_scratch.py --model-size 350m --no-wandb

# 1B model (daha fazla GPU süresi gerekir)
python training/train_from_scratch.py --model-size 1b

# 8B model (60-80 gün, tam eğitim)
python training/train_from_scratch.py --model-size 8b
```

### 7. Değerlendirme
```bash
# Türkçe kalite testi
python evaluation/turkish_bench.py --model ./checkpoints/llm-350m-tr-en/final_model

# Perplexity ölçümü
python evaluation/perplexity_eval.py --model ./checkpoints/llm-350m-tr-en/final_model
```

---

## 📊 2 Saat H100 ile Beklenen Sonuçlar

| Metrik | Beklenen Değer |
|--------|---------------|
| **Model boyutu** | ~350M parametre |
| **İşlenen token** | ~500M - 1B |
| **Final loss** | ~3.5 - 5.0 |
| **Perplexity** | ~30 - 100 |
| **Türkçe kalitesi** | Temel kelime/cümle oluşturma |
| **Throughput** | ~50-100K token/sn |

> ⚠️ **Not:** 2 saatlik eğitim bir proof-of-concept'tir. Gerçek kullanılabilir bir model için
> 8B parametre ve 100B+ token ile haftalarca eğitim gerekir.

---

## 🏗️ Model Mimarisi

```
350M Dense Transformer (Llama tabanlı):
├── Katman sayısı:       24
├── Hidden boyutu:       1024
├── FFN boyutu:          4096
├── Attention heads:     16
├── KV heads:            4 (GQA)
├── Max context:         2048 token
├── Aktivasyon:          SwiGLU
├── Normalizasyon:       RMSNorm
├── Pozisyon encoding:   RoPE
└── Precision:           BF16
```

---

## 📈 Ölçeklendirme Yol Haritası

| Aşama | Model | GPU | Süre | Token |
|-------|-------|-----|------|-------|
| **1. PoC** | 350M | 1x H100, 2 saat | 2 saat | ~1B |
| **2. Prototip** | 1B | 1x H100, 1-2 hafta | 1-2 hafta | ~10B |
| **3. Üretim** | 8B | 1x H100, 2-3 ay | 2-3 ay | ~100B |
| **4. İleri** | 8B MoE | 4-8x H100, 3-6 ay | 3-6 ay | ~500B |

---

## 📝 Notlar

- **Tokenizer:** 2 saatlik sürümde GPT-2 tokenizer kullanılır (hız için). Tam projede özel Türkçe tokenizer eğitin.
- **Veri:** Quick modda Wikipedia-TR ve FineWeb-Edu örneklemleri kullanılır.
- **Checkpoint:** Google Drive'a kaydedilir, Colab oturumu kapansa bile korunur.
- **WandB:** Eğitim takibi için `--no-wandb` flag'ini kaldırın ve API key girin.

---

*Oluşturulma: Mart 2026 | Platform: H100 80GB*
