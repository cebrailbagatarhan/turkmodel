"""
colab_setup.py — Google Colab H100 Tam Kurulum & Eğitim Script'i
==================================================================
Bu dosyayı doğrudan Colab'da çalıştırabilir veya hücrelere bölerek
notebook olarak kullanabilirsiniz.

⏰ HEDEF: 2 saat H100 ile ~6B parametre model eğitimi
📊 ÇIKTI: Sıfırdan eğitilmiş Türkçe+İngilizce LLM

Colab'da kullanım:
    1. Runtime > Change runtime type > H100 GPU seçin
    2. Bu dosyayı Colab'a yükleyin veya içeriği kopyalayın
    3. Her bölümü sırasıyla çalıştırın
"""

# ============================================================
# BÖLÜM 1: SİSTEM KONTROLÜ
# ============================================================
print("=" * 60)
print("🔍 BÖLÜM 1: Sistem Kontrolü")
print("=" * 60)

import subprocess
import sys

# GPU kontrolü
gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(gpu_info.stdout)

import torch
assert torch.cuda.is_available(), "❌ CUDA bulunamadı!"

gpu_name = torch.cuda.get_device_name(0)
gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"✅ GPU: {gpu_name}")
print(f"✅ VRAM: {gpu_mem_gb:.0f} GB")
print(f"✅ CUDA: {torch.version.cuda}")
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ BF16: {torch.cuda.is_bf16_supported()}")

if "H100" not in gpu_name and "A100" not in gpu_name:
    print("⚠️  H100/A100 değil! Eğitim yavaş olabilir.")
    print("   Runtime > Change runtime type > GPU: H100 seçin")


# ============================================================
# BÖLÜM 2: KÜTÜPHANE KURULUMU
# ============================================================
print("\n" + "=" * 60)
print("📦 BÖLÜM 2: Kütüphane Kurulumu")
print("=" * 60)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.46.0",
    "accelerate>=0.34.0",
    "datasets>=3.0.0",
    "tokenizers>=0.19.0",
    "trl>=0.11.0",
    "wandb",
    "huggingface_hub>=0.24.0",
    "kagglehub",
    "datasketch",
    "langdetect",
    "sentencepiece",
    "bitsandbytes>=0.43.0",
], check=True)

# Flash Attention (ayrı kur, build gerekebilir)
try:
    import flash_attn
    print(f"✅ Flash Attention {flash_attn.__version__} zaten kurulu")
except ImportError:
    print("📦 Flash Attention kuruluyor (bu birkaç dakika sürebilir)...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q",
        "flash-attn>=2.6.3", "--no-build-isolation",
    ])

print("✅ Tüm kütüphaneler kuruldu!")

import os

def get_secret(name: str):
    """Ortam değişkeni veya Colab Secrets içinden bir değer oku."""
    value = os.getenv(name)
    if value:
        return value

    try:
        from google.colab import userdata
        return userdata.get(name)
    except Exception:
        return None


# Kaggle API ayarları — değerleri kaynak koda yazmayın.
kaggle_username = get_secret("KAGGLE_USERNAME")
kaggle_key = get_secret("KAGGLE_KEY")
if kaggle_username and kaggle_key:
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    print("✅ Kaggle API ayarlandı")
else:
    print("⚠️  KAGGLE_USERNAME/KAGGLE_KEY bulunamadı; Kaggle adımları başarısız olabilir.")


# ============================================================
# BÖLÜM 3: GOOGLE DRIVE BAĞLANTISI
# ============================================================
print("\n" + "=" * 60)
print("💾 BÖLÜM 3: Google Drive Bağlantısı")
print("=" * 60)

try:
    from google.colab import drive
    drive.mount("/content/drive")
    SAVE_DIR = "/content/drive/MyDrive/LLM_Training"
    IN_COLAB = True
except ImportError:
    SAVE_DIR = "./output"
    IN_COLAB = False
    print("⚠️  Colab ortamında değilsiniz, yerel dizin kullanılıyor")

CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
MODEL_DIR = os.path.join(SAVE_DIR, "final_model")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"✅ Checkpoint dizini: {CHECKPOINT_DIR}")


# ============================================================
# BÖLÜM 4: VERİ İNDİRME (Quick Mode)
# ============================================================
print("\n" + "=" * 60)
print("📥 BÖLÜM 4: Veri İndirme")
print("=" * 60)

from datasets import load_dataset, Dataset, concatenate_datasets
import random

random.seed(42)

NUM_SAMPLES = 200_000  # 2 saat için yeterli

all_texts = []

# --- Türkçe Wikipedia ---
print("📥 Türkçe Wikipedia indiriliyor...")
try:
    wiki_tr = load_dataset(
        "wikimedia/wikipedia", "20231101.tr",
        split=f"train[:{NUM_SAMPLES}]",
        trust_remote_code=True,
    )
    for item in wiki_tr:
        if len(item["text"]) > 200:
            all_texts.append({"text": item["text"][:5000], "lang": "tr"})
    print(f"  ✅ {sum(1 for t in all_texts if t['lang']=='tr'):,} Türkçe belge")
except Exception as e:
    print(f"  ⚠️ Wikipedia-TR: {e}")

# --- İngilizce FineWeb-Edu ---
print("📥 İngilizce veri indiriliyor...")
try:
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT",
        split=f"train[:{int(NUM_SAMPLES * 1.5)}]",
        trust_remote_code=True,
    )
    for item in fineweb:
        if len(item["text"]) > 200:
            all_texts.append({"text": item["text"][:5000], "lang": "en"})
    print(f"  ✅ {sum(1 for t in all_texts if t['lang']=='en'):,} İngilizce belge")
except Exception as e:
    print(f"  ⚠️ FineWeb: {e}")
    # Alternatif
    try:
        wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"train[:{NUM_SAMPLES}]")
        for item in wikitext:
            if len(item["text"]) > 200:
                all_texts.append({"text": item["text"][:5000], "lang": "en"})
        print(f"  ✅ WikiText alternatif kullanıldı")
    except:
        pass

print(f"\n📊 Toplam: {len(all_texts):,} belge")
tr_count = sum(1 for t in all_texts if t["lang"] == "tr")
en_count = sum(1 for t in all_texts if t["lang"] == "en")
print(f"  TR: {tr_count:,} | EN: {en_count:,}")

# Karıştır
random.shuffle(all_texts)
raw_dataset = Dataset.from_list(all_texts)


# ============================================================
# BÖLÜM 5: TOKENİZASYON
# ============================================================
print("\n" + "=" * 60)
print("🔤 BÖLÜM 5: Tokenizasyon")
print("=" * 60)

from transformers import AutoTokenizer
from itertools import chain

# Mevcut küçük tokenizer kullan (zaman kazanmak için)
# Gerçek projede özel tokenizer eğitin
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

SEQ_LENGTH = 2048
VOCAB_SIZE = tokenizer.vocab_size

print(f"  Tokenizer: GPT-2 ({VOCAB_SIZE} vocab)")
print(f"  Sequence uzunluğu: {SEQ_LENGTH}")

# Tokenize et
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=False, padding=False,
                     return_attention_mask=False)

tokenized = raw_dataset.map(tokenize_fn, batched=True, batch_size=1000,
                             num_proc=2, remove_columns=raw_dataset.column_names)

# Grupla
def group_texts(examples):
    concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // SEQ_LENGTH) * SEQ_LENGTH
    result = {
        k: [concatenated[k][i:i + SEQ_LENGTH] for i in range(0, total_length, SEQ_LENGTH)]
        for k in concatenated.keys()
    }
    result["labels"] = result["input_ids"].copy()
    return result

grouped = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=2)

# Train/Val böl
split = grouped.train_test_split(test_size=0.05, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

total_tokens = len(train_dataset) * SEQ_LENGTH
print(f"\n✅ Train: {len(train_dataset):,} sequence ({total_tokens:,} token = {total_tokens/1e6:.1f}M)")
print(f"✅ Val:   {len(eval_dataset):,} sequence")


# ============================================================
# BÖLÜM 6: MODEL OLUŞTURMA (SIFIRDAN)
# ============================================================
print("\n" + "=" * 60)
print("🏗️ BÖLÜM 6: Model Oluşturma (Sıfırdan)")
print("=" * 60)

from transformers import LlamaForCausalLM, LlamaConfig

# H100 optimizasyonları
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Flash Attention kontrolü
try:
    import flash_attn
    attn_impl = "flash_attention_2"
except ImportError:
    attn_impl = "sdpa"

# ~6B model konfigürasyonu (Llama-2-7B tabanlı, GQA)
config = LlamaConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=SEQ_LENGTH,
    rope_theta=500000.0,
    hidden_act="silu",
    rms_norm_eps=1e-5,
    tie_word_embeddings=False,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_impl,
)

model = LlamaForCausalLM(config)
model = model.to(dtype=torch.bfloat16)

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model oluşturuldu: {total_params/1e6:.1f}M parametre")
print(f"  Hidden: {config.hidden_size}, Layers: {config.num_hidden_layers}")
print(f"  Heads: {config.num_attention_heads}, KV Heads: {config.num_key_value_heads}")
print(f"  Attention: {attn_impl}")

# VRAM tahmini
param_mem = total_params * 2 / 1e9  # BF16 = 2 byte
optim_mem = total_params * 2 / 1e9  # 8-bit Adam states
print(f"  VRAM tahmini: ~{param_mem + optim_mem:.1f} GB (model + optimizer)")


# ============================================================
# BÖLÜM 7: EĞİTİM
# ============================================================
print("\n" + "=" * 60)
print("🚀 BÖLÜM 7: EĞİTİM (2 Saat H100)")
print("=" * 60)

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import time
import math

# Eğitim süresi tahmini
BATCH_SIZE = 4
GRAD_ACCUM = 4
effective_batch_tokens = BATCH_SIZE * GRAD_ACCUM * SEQ_LENGTH
MAX_STEPS = 400  # 300-400 iterasyon yeterli, fazlası overfit yapar

print(f"  Efektif batch: {effective_batch_tokens:,} token")
print(f"  Max steps: {MAX_STEPS:,}")
print(f"  Toplam token: ~{MAX_STEPS * effective_batch_tokens / 1e6:.0f}M")

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_steps=40,
    weight_decay=0.1,
    max_grad_norm=1.0,
    
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    logging_steps=10,
    logging_first_step=True,
    save_steps=100,
    save_total_limit=5,
    eval_strategy="steps",
    eval_steps=100,
    
    report_to="none",  # WandB istenirse "wandb" yapın
    run_name="llm-6b-tr-en-2h",
    
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    remove_unused_columns=False,
    
    optim="adamw_bnb_8bit",
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-8,
    
    seed=42,
    include_tokens_per_second=True,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
)

# Başlat!
start_time = time.time()
print(f"\n🚀 Eğitim başlıyor... (Hedef: {MAX_STEPS} step)")
print("   Loss düşüşünü takip edin!")
print("-" * 60)

trainer.train()

elapsed = time.time() - start_time
print(f"\n✅ Eğitim tamamlandı! Süre: {elapsed/60:.1f} dakika ({elapsed/3600:.1f} saat)")


# ============================================================
# BÖLÜM 8: MODEL KAYDETME
# ============================================================
print("\n" + "=" * 60)
print("💾 BÖLÜM 8: Model Kaydetme")
print("=" * 60)

trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
print(f"✅ Model kaydedildi: {MODEL_DIR}")

# Konfigürasyon kaydet
import json
# Son train loss'u bul (log_history'de eval entry'ler de olabilir)
final_loss = "N/A"
for log_entry in reversed(trainer.state.log_history):
    if "loss" in log_entry:
        final_loss = log_entry["loss"]
        break

training_info = {
    "model_size": f"{total_params/1e6:.1f}M",
    "total_params": total_params,
    "training_steps": MAX_STEPS,
    "training_time_minutes": elapsed / 60,
    "total_tokens_processed": MAX_STEPS * effective_batch_tokens,
    "gpu": gpu_name,
    "final_loss": final_loss,
}
with open(os.path.join(MODEL_DIR, "training_info.json"), "w") as f:
    json.dump(training_info, f, indent=2)


# ============================================================
# BÖLÜM 9: HIZLI DEĞERLENDİRME
# ============================================================
print("\n" + "=" * 60)
print("📝 BÖLÜM 9: Hızlı Değerlendirme")
print("=" * 60)

model.eval()

test_prompts = [
    "Türkiye'nin başkenti",
    "Istanbul is a city that",
    "Python programming language is",
    "Yapay zeka teknolojisi",
    "The transformer architecture",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🔹 {prompt}")
    print(f"   → {response}")

# ============================================================
# BÖLÜM 10: HUGGINGFACE HUB'A YÜKLEME
# ============================================================
print("\n" + "=" * 60)
print("🚀 BÖLÜM 10: HuggingFace Hub'a Yükleme")
print("=" * 60)

# HuggingFace Hub'a yükle (model dosyaları büyük — GitHub 100MB limiti var)
# Bu sayede modeli istediğiniz yerden indirebilirsiniz
HF_REPO_ID = "cebrailbagatarhan/turkmodel-6b"  # ← Kendi HF kullanıcı adınızla değiştirin

try:
    from huggingface_hub import HfApi, login
    
    # HuggingFace token ile otomatik giriş
    HF_TOKEN = get_secret("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN bulunamadı; ortam değişkeni veya Colab Secrets kullanın.")

    print("📝 HuggingFace Hub'a giriş yapılıyor...")
    login(token=HF_TOKEN)
    
    api = HfApi()
    
    # Repo oluştur (yoksa)
    try:
        api.create_repo(repo_id=HF_REPO_ID, exist_ok=True, repo_type="model")
        print(f"  ✅ HF repo hazır: https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"  ⚠️ Repo oluşturulamadı: {e}")
    
    # Model dosyalarını yükle
    print("📤 Model yükleniyor (bu birkaç dakika sürebilir)...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message=f"350M TR-EN LLM - {MAX_STEPS} steps - {elapsed/60:.0f}min on {gpu_name}",
    )
    print(f"  ✅ Model yüklendi: https://huggingface.co/{HF_REPO_ID}")
    print(f"  📥 İndirmek için: git clone https://huggingface.co/{HF_REPO_ID}")
    
except Exception as e:
    print(f"  ⚠️ HuggingFace Hub yükleme atlanıyor: {e}")
    print("  → Manuel yükleme: huggingface-cli upload cebrailbagatarhan/turkmodel-6b ./final_model")


# ============================================================
# BÖLÜM 11: GİTHUB'A PUSH (Script + Sonuçlar)
# ============================================================
print("\n" + "=" * 60)
print("📦 BÖLÜM 11: GitHub'a Push")
print("=" * 60)

GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = "https://github.com/cebrailbagatarhan/turkmodel.git"

try:
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN bulunamadı; GitHub push atlanıyor.")

    # Tokenı clone URL'sine veya .git/config dosyasına yazmadan Git'e ilet.
    import base64
    basic_auth = base64.b64encode(
        f"x-access-token:{GITHUB_TOKEN}".encode("utf-8")
    ).decode("ascii")
    git_env = os.environ.copy()
    git_env.update({
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraHeader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic_auth}",
    })

    # Git kurulumu
    subprocess.run(["git", "config", "--global", "user.email", "cebrailbagatarhan@users.noreply.github.com"], check=False)
    subprocess.run(["git", "config", "--global", "user.name", "cebrailbagatarhan"], check=False)
    
    # Repo'yu klonla (token ile auth)
    REPO_DIR = "/content/turkmodel_repo"
    if os.path.exists(REPO_DIR):
        import shutil
        shutil.rmtree(REPO_DIR)
    
    subprocess.run(["git", "clone", GITHUB_REPO, REPO_DIR], check=True, env=git_env)
    
    # training_info.json ve eval sonuçlarını kopyala (küçük dosyalar)
    import shutil
    results_dir = os.path.join(REPO_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Training info
    info_src = os.path.join(MODEL_DIR, "training_info.json")
    if os.path.exists(info_src):
        shutil.copy2(info_src, os.path.join(results_dir, "training_info.json"))
    
    # Eval sonuçları (varsa)
    eval_src = os.path.join(MODEL_DIR, "turkish_eval_results.json")
    if os.path.exists(eval_src):
        shutil.copy2(eval_src, os.path.join(results_dir, "turkish_eval_results.json"))
    
    # README güncelle: model indirme linkini ekle
    readme_path = os.path.join(REPO_DIR, "llm-8b-tr-en", "TRAINING_RESULTS.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(f"""# Eğitim Sonuçları

## Model Bilgileri
- **Boyut:** {total_params/1e6:.1f}M parametre
- **Mimari:** Llama (Dense Transformer)
- **Eğitim:** {MAX_STEPS} step, {elapsed/60:.1f} dakika
- **GPU:** {gpu_name}
- **Final Loss:** {final_loss}

## Modeli İndir
```bash
# HuggingFace'den indir (önerilen)
pip install huggingface_hub
huggingface-cli download {HF_REPO_ID} --local-dir ./model

# veya Python ile
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{HF_REPO_ID}")
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_ID}")
```

## Google Drive
Model ayrıca Google Drive'da da kaydedildi:
`{MODEL_DIR}`
""")
    
    # Git push
    subprocess.run(["git", "-C", REPO_DIR, "add", "-A"], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "commit", "-m",
                     f"Egitim sonuclari - {total_params/1e6:.0f}M model - {MAX_STEPS} steps"],
                    check=True)
    subprocess.run(["git", "-C", REPO_DIR, "push"], check=True, env=git_env)
    
    print(f"  ✅ GitHub'a push edildi: {GITHUB_REPO}")
    print(f"  📄 Sonuçlar: results/training_info.json")

except Exception as e:
    print(f"  ⚠️ GitHub push atlanıyor: {e}")
    print(f"  → Model Google Drive'da mevcut: {MODEL_DIR}")


print("\n" + "=" * 60)
print("🎉 TAMAMLANDI!")
print("=" * 60)
print(f"""
📊 Özet:
  Model:     {total_params/1e6:.1f}M parametre (Llama mimarisi)
  Eğitim:    {MAX_STEPS} step, {elapsed/60:.1f} dakika
  GPU:       {gpu_name}
  Kaydedildi: {MODEL_DIR}

📥 Modeli İndirme Yolları:
  1. HuggingFace: https://huggingface.co/{HF_REPO_ID}
  2. GitHub sonuçları: {GITHUB_REPO}
  3. Google Drive: {MODEL_DIR}

⏭️ Sonraki Adımlar:
  1. SFT (Supervised Fine-Tuning) yapabilirsiniz
  2. Daha fazla veri ile eğitime devam edebilirsiniz
  3. Daha büyük model (1B, 8B) için aynı pipeline'ı kullanın
""")
