# 🧠 Sıfırdan Modern LLM Eğitimi — Copilot Teknik Brifing
> **Hedef:** Qwen3.5-397B-A17B mimarisine benzer, sıfırdan eğitilmiş, Türkçe + İngilizce destekli bir LLM.  
> **Platform:** Google Colab Pro+ — NVIDIA H100 80GB  
> **Yöntem:** Pre-training from scratch (ağırlık transferi yok)

---

## 1. 🎯 Model Boyutu Seçimi — Hangisi Uygun?

| Model Boyutu | Katman | Kafa | Hidden Dim | H100 Sayısı | Eğitim Süresi* | Öneri |
|---|---|---|---|---|---|---|
| **7B (Dense)** | 32 | 32 | 4096 | 1x H100 | ~3-5 hafta | ✅ **En gerçekçi başlangıç** |
| **8B (Dense)** | 32 | 32 | 4096 | 1x H100 | ~4-6 hafta | ✅ Llama-3 mimarisi |
| **9B (Dense)** | 36 | 36 | 4096 | 1x H100 | ~5-7 hafta | ✅ Gemma-2 benzeri |
| **13B (Dense)** | 40 | 40 | 5120 | 2x H100 | ~8-12 hafta | ⚠️ Colab'da zorlanır |
| **40B (MoE)** | 64 | 64 | 8192 | 4-8x H100 | ~20+ hafta | ❌ Colab için değil |
| **397B MoE** | 94 | 64 | 7168 | 32+ H100 | Aylar | ❌ Data center gerekir |

### 🏆 ÖNERİLEN: **8B Dense Model (Llama-3 / Qwen-2.5 tarzı)**
```
Neden 8B?
→ Tek H100 80GB'a tam olarak sığar (BF16: ~16GB, optimizer dahil ~60GB)
→ Türkçe için yeterli kapasite (GPT-3.5 seviyesine ulaşılabilir)
→ 100B token ile makul sürede eğitilebilir
→ Toplulukta en çok test edilmiş boyut
→ Colab H100 ile günde ~1-2B token işlenebilir
```

---

## 2. 🏗️ Model Mimarisi (8B Sıfırdan)

```python
# model_config.py — 8B Dense Transformer

MODEL_CONFIG = {
    # --- Temel Boyutlar ---
    "vocab_size": 128_256,        # Türkçe tokenlar eklenince 130K+ önerilir
    "hidden_size": 4096,          # Gizli katman boyutu
    "intermediate_size": 14336,   # FFN iç boyutu (hidden * 3.5)
    "num_hidden_layers": 32,      # Transformer katman sayısı
    
    # --- Attention Başlıkları ---
    "num_attention_heads": 32,    # Sorgu başlıkları
    "num_key_value_heads": 8,     # GQA: K/V başlıkları (bellek tasarrufu)
    "head_dim": 128,              # Her başlık boyutu
    
    # --- Pozisyon & Bağlam ---
    "max_position_embeddings": 8192,   # Context window (32K'ya uzatılabilir)
    "rope_theta": 500000.0,            # RoPE frekans tabanı
    "rope_scaling": {
        "type": "llama3",
        "factor": 8.0,
    },
    
    # --- Aktivasyon & Normalizasyon ---
    "hidden_act": "silu",         # SwiGLU aktivasyonu
    "rms_norm_eps": 1e-05,        # RMSNorm epsilon
    
    # --- Diğer ---
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "model_type": "llama",        # Llama-3 mimarisi kullan (en stabil)
}

# Parametre sayısı tahmini:
# Embedding: 128256 * 4096 = ~525M
# Her katman: ~134M (32 katman = ~4.3B)
# LM Head: ~525M
# TOPLAM: ~7.5-8.5B parametre ✓
```

---

## 3. 🔤 Tokenizer — Türkçe Desteği

```python
# tokenizer_train.py — SentencePiece BPE ile özel tokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# Türkçe + İngilizce tokenizer eğit
trainer = BpeTrainer(
    vocab_size=130_000,           # Türkçe için +2K ek token
    min_frequency=2,
    special_tokens=[
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|pad|>",
        "<|im_start|>",           # Chat template için
        "<|im_end|>",
        "<|unk|>",
    ],
    # Türkçe özel karakterler korunur
    initial_alphabet=ByteLevel.alphabet(),
)

# Eğitim verisi: Türkçe + İngilizce karma dosyalar
tokenizer_model = Tokenizer(BPE(unk_token="<|unk|>"))
tokenizer_model.pre_tokenizer = ByteLevel()

# Corpus dosyaları
corpus_files = [
    "./data/turkish_corpus.txt",   # ~10GB Türkçe metin
    "./data/english_corpus.txt",   # ~10GB İngilizce metin
]
tokenizer_model.train(files=corpus_files, trainer=trainer)
tokenizer_model.save("./tokenizer/tokenizer.json")

print(f"Türkçe test: {tokenizer_model.encode('merhaba dünya').tokens}")
# Beklenen: ['merhaba', 'Ġdünya'] — tek parça (iyi tokenization)
```

---

## 4. 📦 Veri Seti Hazırlığı

### 4.1 Türkçe Kaynaklar (~40-50B token hedefi)

```python
TURKISH_SOURCES = {
    # Web Crawl
    "oscar_tr": {
        "path": "oscar-corpus/OSCAR-2301",
        "lang": "tr",
        "size_gb": 45,
        "tokens_est": "15B",
    },
    # Ansiklopedi
    "wikipedia_tr": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.tr",
        "size_gb": 1.2,
        "tokens_est": "350M",
    },
    # Haber
    "milliyet_news": {
        "path": "Overfit-GM/turkish-news-category-dataset",
        "tokens_est": "500M",
    },
    # Kitaplar & Akademik
    "dergipark": {
        "source": "https://dergipark.org.tr",
        "method": "scraping",
        "tokens_est": "2B",
    },
    # Sosyal Medya & Diyalog
    "turkish_tweets": {
        "path": "malhajar/turkish-twitter",
        "tokens_est": "1B",
    },
    # Talimat Verisi (SFT için)
    "alpaca_tr": {
        "path": "Trendyol/trendyol-llm-turkish-alpaca-clean",
        "tokens_est": "50M",
    },
}
```

### 4.2 İngilizce Kaynaklar (~50-60B token hedefi)

```python
ENGLISH_SOURCES = {
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "score_threshold": 3,          # Yüksek kalite filtresi
        "tokens_est": "300B (sample al)",
    },
    "dolma": {
        "path": "allenai/dolma",
        "subsets": ["c4", "stack", "wiki"],
        "tokens_est": "50B (sample al)",
    },
    "openhermes": {
        "path": "teknium/OpenHermes-2.5",
        "tokens_est": "1B",
    },
}

# Dil Karışım Oranı
DATA_MIX = {
    "turkish": 0.40,    # %40 Türkçe
    "english": 0.55,    # %55 İngilizce
    "code": 0.05,       # %5 Kod (Python, genel)
}
```

### 4.3 Veri Temizleme Pipeline

```python
# data_cleaning.py

import re
from langdetect import detect

def quality_filter(text: str, lang: str) -> bool:
    """Kalite filtresi"""
    # Minimum uzunluk
    if len(text.split()) < 50:
        return False
    # Maksimum tekrar oranı
    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.3:          # %30'dan az benzersiz kelime → spam
        return False
    # Dil kontrolü
    try:
        detected = detect(text[:500])
        if detected != lang:
            return False
    except:
        return False
    # Türkçe karakter oranı (TR için)
    if lang == "tr":
        tr_chars = set("çğıöşüÇĞİÖŞÜ")
        tr_ratio = sum(c in tr_chars for c in text) / len(text)
        if tr_ratio < 0.005:        # Çok az Türkçe karakter → geç
            return False
    return True

def dedup_documents(dataset):
    """MinHash LSH ile yakın kopya kaldırma"""
    from datasketch import MinHash, MinHashLSH
    lsh = MinHashLSH(threshold=0.85, num_perm=128)
    unique_docs = []
    for i, doc in enumerate(dataset):
        m = MinHash(num_perm=128)
        for word in doc["text"].split():
            m.update(word.encode("utf8"))
        if not lsh.query(m):
            lsh.insert(f"doc_{i}", m)
            unique_docs.append(doc)
    return unique_docs
```

---

## 5. ⚙️ Eğitim Konfigürasyonu (Sıfırdan / From Scratch)

### 5.1 Hyperparameter Tablosu

| Parametre | 7-8B Değeri | Açıklama |
|---|---|---|
| `learning_rate` | 3e-4 | Peak LR (Cosine decay) |
| `min_lr` | 3e-5 | LR'nin düşeceği minimum |
| `warmup_steps` | 2000 | LR warm-up adımı |
| `batch_size` | 2M token | Global batch (grad accum ile) |
| `seq_length` | 4096 | Eğitim sequence uzunluğu |
| `weight_decay` | 0.1 | L2 regularization |
| `grad_clip` | 1.0 | Gradient clipping |
| `optimizer` | AdamW | β1=0.9, β2=0.95, ε=1e-8 |
| `dtype` | BF16 | H100 native format |
| `total_tokens` | 100B+ | Eğitim token hedefi |

### 5.2 Tam Eğitim Scripti

```python
# train_from_scratch.py

import torch
import torch.distributed as dist
from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_from_disk
import wandb

# ============================================================
# MODEL YAPILANDIRMASI
# ============================================================
config = LlamaConfig(
    vocab_size=130_000,
    hidden_size=4096,
    intermediate_size=14336,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,          # Grouped Query Attention
    max_position_embeddings=8192,
    rope_theta=500000.0,
    hidden_act="silu",
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Sıfırdan model oluştur (random weights)
model = LlamaForCausalLM(config)
model = model.to(dtype=torch.bfloat16)

# Parametre sayısını göster
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Toplam Parametre: {total_params/1e9:.2f}B")

# ============================================================
# EĞİTİM ARGÜMANLARI
# ============================================================
training_args = TrainingArguments(
    output_dir="./checkpoints/llm-8b-tr-en",
    
    # Token & Batch
    max_steps=50_000,                    # 100B token / (2M batch) = 50K adım
    per_device_train_batch_size=4,       # H100'de 4 sequence
    gradient_accumulation_steps=128,     # Efektif batch: 4*128=512 seq * 4096 tok = ~2M token
    
    # Learning Rate
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=2000,
    weight_decay=0.1,
    max_grad_norm=1.0,
    
    # Precision & Memory
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,         # VRAM tasarrufu
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Loglama & Kaydetme
    logging_steps=10,
    save_steps=1000,
    save_total_limit=3,
    report_to="wandb",
    run_name="llm-8b-tr-en-scratch",
    
    # Veri
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # Optimizasyon
    optim="adamw_torch_fused",           # H100 için fused AdamW
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
)

# ============================================================
# VERİ & EĞİTİCİ
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
dataset = load_from_disk("./data/tokenized_dataset")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,                           # Causal LM (GPT-tarzı)
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ============================================================
# EĞİTİMİ BAŞLAT
# ============================================================
print("🚀 Eğitim başlıyor...")
wandb.init(project="llm-tr-en-scratch", name="8b-run-001")
trainer.train()
trainer.save_model("./final_model")
print("✅ Eğitim tamamlandı!")
```

---

## 6. 💾 Google Colab H100 Kurulumu

```python
# ============================================================
# COLAB HÜCRE 1 — Sistem Kontrolü
# ============================================================
import subprocess, torch

gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(gpu_info.stdout)

assert "H100" in gpu_info.stdout, "❌ H100 bulunamadı! Runtime > Change runtime type > H100"
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")
print(f"✅ CUDA: {torch.version.cuda}")
print(f"✅ BF16 Destekli: {torch.cuda.is_bf16_supported()}")

# ============================================================
# COLAB HÜCRE 2 — Kütüphane Kurulumu
# ============================================================
!pip install -q \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "datasets>=3.0.0" \
    "tokenizers>=0.19.0" \
    "trl>=0.11.0" \
    "wandb" \
    "datasketch" \
    "langdetect" \
    "sentencepiece" \
    "flash-attn==2.6.3" --no-build-isolation

# ============================================================
# COLAB HÜCRE 3 — Google Drive Bağlantısı (checkpoint kaydetmek için)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
CHECKPOINT_DIR = "/content/drive/MyDrive/LLM_Training/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"✅ Checkpoint dizini: {CHECKPOINT_DIR}")

# ============================================================
# COLAB HÜCRE 4 — Weights & Biases (eğitim takibi)
# ============================================================
import wandb
wandb.login(key="WANDB_API_KEY_BURAYA")
```

---

## 7. 📉 Beklenen Eğitim Metrikleri

```
Eğitim Adımı    | Loss    | Perplexity | Açıklama
----------------|---------|------------|---------------------------
Step 100        | ~10.5   | ~36000     | Başlangıç (random weights)
Step 1,000      | ~5.8    | ~330       | Temel kelime öğreniyor
Step 5,000      | ~3.9    | ~49        | Cümle yapısı oluşuyor
Step 15,000     | ~3.1    | ~22        | Dilbilgisi yerleşiyor
Step 30,000     | ~2.6    | ~13        | Türkçe fluency başlıyor
Step 50,000     | ~2.3    | ~10        | Temel LLM kalitesi ✓
```

> **Sağlıklı eğitim sinyalleri:**
> - Loss düzenli düşmeli (ani artışlar → LR çok yüksek)
> - Gradient norm ~0.5-2.0 arasında (>10 → exploding gradient)
> - GPU kullanımı >%90 olmalı

---

## 8. 🗂️ Proje Dosya Yapısı

```
llm-8b-tr-en/
│
├── 📄 README.md                    ← Bu dosya
├── 📄 requirements.txt
│
├── 📁 configs/
│   ├── model_config.json           ← 8B mimari parametreleri
│   ├── tokenizer_config.json
│   └── training_config.yaml
│
├── 📁 data/
│   ├── download_datasets.py        ← HuggingFace'den veri çekme
│   ├── clean_turkish.py            ← TR metin temizleme
│   ├── dedup.py                    ← MinHash deduplification
│   ├── tokenize_dataset.py         ← BPE tokenization
│   └── create_data_mix.py          ← %40 TR / %55 EN / %5 kod karıştırma
│
├── 📁 tokenizer/
│   ├── train_tokenizer.py          ← SentencePiece BPE eğitimi
│   └── tokenizer.json
│
├── 📁 model/
│   ├── modeling_llama.py           ← Model mimarisi (Llama-3 tabanlı)
│   └── configuration_llama.py
│
├── 📁 training/
│   ├── train_from_scratch.py       ← Ana eğitim scripti
│   ├── eval.py                     ← Değerlendirme
│   └── colab_setup.ipynb           ← Colab notebook
│
└── 📁 evaluation/
    ├── turkish_bench.py            ← Türkçe benchmark
    └── perplexity_eval.py
```

---

## 9. ⚡ H100 Verim Optimizasyonları

```python
# H100'e özel optimizasyonlar

import torch
torch.backends.cuda.matmul.allow_tf32 = True    # TF32 aritmetik
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Flash Attention 2 (H100'de 3x hız artışı)
model = LlamaForCausalLM.from_pretrained(
    config,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

# Torch.compile (Triton JIT — %20-30 ekstra hız)
model = torch.compile(model, mode="reduce-overhead")

# Beklenen throughput (H100 80GB, 8B model, seq_len=4096):
# BF16 + Flash Attention 2 + torch.compile:
# → ~15,000-20,000 token/saniye
# → ~1.3-1.7B token/gün
# → 100B token için ~60-80 gün (tek H100)
```

---

## 10. 🇹🇷 Türkçe Kalite Değerlendirmesi

```python
# turkish_eval.py — Eğitim sırasında Türkçe kalite ölçümü

TURKISH_EVAL_PROMPTS = [
    # Dilbilgisi kontrolü
    "Türkiye'nin başkenti",                    # → "Ankara'dır"
    "İstanbul Boğazı hangi iki denizi",        # → "Karadeniz ve Marmara'yı"
    
    # Çekim ekleri
    "Çocuklar parka",                          # → "gittiler"
    "Bu kitabı okumak",                        # → "istiyorum"
    
    # Soru-cevap
    "Atatürk ne zaman doğdu?",                 # → "1881"
    "Python'da liste nasıl oluşturulur?",
    
    # Yaratıcı yazarlık
    "Bir sonbahar günü hakkında kısa şiir:",
]

def eval_turkish_quality(model, tokenizer, step):
    model.eval()
    results = {}
    for prompt in TURKISH_EVAL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results[prompt] = response
        print(f"[Step {step}] {prompt}\n→ {response}\n")
    return results
```

---

## 11. 💡 Özet — Copilot'a Söylenecekler

```
1. MİMARİ: Llama-3 8B mimarisini uygula
   - 32 katman, 32 attention head, 8 KV head (GQA)
   - hidden_size=4096, intermediate_size=14336
   - RoPE pozisyon encoding, SwiGLU aktivasyon
   - RMSNorm, Flash Attention 2

2. TOKENİZER: Yeni BPE tokenizer eğit
   - vocab_size=130_000 (Türkçe karakterler dahil)
   - SentencePiece veya HuggingFace tokenizers

3. VERİ: 100B token hedefi
   - %40 Türkçe (OSCAR-TR, Wikipedia-TR, haber, kitap)
   - %55 İngilizce (FineWeb-Edu, Dolma)
   - %5 Kod (Python, genel)

4. EĞİTİM: H100'de from scratch
   - BF16 + Flash Attention 2 + torch.compile
   - Cosine LR schedule, peak=3e-4
   - Gradient checkpointing açık
   - 2M token global batch size

5. ORTAM: Google Colab H100 80GB
   - transformers + accelerate + trl + flash-attn
   - Checkpointler Google Drive'a kaydet
   - WandB ile takip et
```

---

*Hazırlayan: Claude Sonnet | Versiyon: 1.0 | Tarih: Mart 2026*
