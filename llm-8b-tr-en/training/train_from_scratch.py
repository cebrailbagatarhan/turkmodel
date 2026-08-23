"""
train_from_scratch.py — Sıfırdan LLM Eğitimi (2 Saat H100 Versiyonu)
======================================================================
350M parametre Dense Transformer — Türkçe + İngilizce

Bu script hem 2 saat H100 (350M) hem de tam eğitim (8B) için kullanılabilir.
--model-size parametresi ile seçim yapılır.

Kullanım:
    # 2 saat H100 (350M model)
    python training/train_from_scratch.py --model-size 350m

    # Tam eğitim (8B model) — çoklu GPU / uzun süre
    python training/train_from_scratch.py --model-size 8b
"""

import os
import sys
import json
import hashlib
import argparse
import time
from datetime import timedelta

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_from_disk

# ============================================================
# MODEL KONFİGÜRASYONLARI
# ============================================================

def tokenizer_fingerprint(tokenizer) -> str:
    """Tokenizer vocabulary ve ayarlarının SHA-256 kimliğini üret."""
    digest = hashlib.sha256()
    digest.update(tokenizer.backend_tokenizer.to_str().encode("utf-8"))
    digest.update(
        json.dumps(
            tokenizer.special_tokens_map,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


MODEL_CONFIGS = {
    "350m": {
        "vocab_size": 32_000,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "rope_theta": 500000.0,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        # Eğitim parametreleri
        "max_steps": 5000,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 8,
        "learning_rate": 6e-4,
        "warmup_steps": 500,
        "seq_length": 2048,
    },
    "1b": {
        "vocab_size": 32_000,
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 2048,
        "rope_theta": 500000.0,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "max_steps": 3000,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 16,
        "learning_rate": 5e-4,
        "warmup_steps": 500,
        "seq_length": 2048,
    },
    "8b": {
        "vocab_size": 130_000,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "max_steps": 50_000,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 128,
        "learning_rate": 3e-4,
        "warmup_steps": 2000,
        "seq_length": 4096,
    },
}


# ============================================================
# ÖZEL CALLBACK — Eğitim İzleme
# ============================================================

class TrainingMonitorCallback(TrainerCallback):
    """Eğitim süresince metrikleri izle ve yazdır"""
    
    def __init__(self):
        self.start_time = None
        self.step_times = []
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("🚀 EĞİTİM BAŞLADI")
        print("=" * 60)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            elapsed = time.time() - self.start_time
            step = state.global_step
            total_steps = state.max_steps
            
            # Tahmini kalan süre
            if step > 0:
                per_step = elapsed / step
                remaining = per_step * (total_steps - step)
                eta = timedelta(seconds=int(remaining))
            else:
                eta = "?"
            
            loss = logs.get("loss", 0)
            lr = logs.get("learning_rate", 0)
            grad_norm = logs.get("grad_norm", 0)
            
            # Perplexity
            import math
            ppl = math.exp(min(loss, 20))  # overflow koruması
            
            print(f"  Step {step:>6}/{total_steps} | "
                  f"Loss: {loss:.4f} | PPL: {ppl:.1f} | "
                  f"LR: {lr:.2e} | GradNorm: {grad_norm:.2f} | "
                  f"ETA: {eta}")
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(f"✅ EĞİTİM TAMAMLANDI — Toplam süre: {timedelta(seconds=int(total_time))}")
        print("=" * 60)


# ============================================================
# ANA EĞİTİM FONKSİYONU
# ============================================================

def train_model(
    model_size: str = "350m",
    dataset_path: str = "./data/tokenized_dataset",
    tokenizer_path: str = "./tokenizer",
    output_dir: str = None,
    resume_from_checkpoint: str = None,
    use_wandb: bool = True,
    wandb_project: str = "llm-tr-en-scratch",
):
    """
    Sıfırdan LLM eğitimi.
    """
    
    preset_config = MODEL_CONFIGS.get(model_size)
    if preset_config is None:
        print(f"❌ Bilinmeyen model boyutu: {model_size}")
        print(f"   Seçenekler: {list(MODEL_CONFIGS.keys())}")
        return
    config_dict = dict(preset_config)
    
    if output_dir is None:
        output_dir = f"./checkpoints/llm-{model_size}-tr-en"
    
    print(f"🏗️  Model Boyutu: {model_size.upper()}")
    print(f"📂 Dataset: {dataset_path}")
    print(f"💾 Checkpoint: {output_dir}")
    
    # ============================================================
    # H100 OPTİMİZASYONLARI
    # ============================================================
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
        print(f"✅ BF16: {torch.cuda.is_bf16_supported()}")
    else:
        print("⚠️  CUDA bulunamadı! CPU ile eğitim çok yavaş olacak.")
    
    # ============================================================
    # TOKENIZER VE DATASET MANIFEST DOĞRULAMA
    # ============================================================
    print("\n🔤 Tokenizer yükleniyor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Özel tokenizer yüklenemedi. Tokenized dataset üzerinde GPT-2 "
            "fallback kullanmak token ID anlamlarını bozar."
        ) from exc
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer pad_token veya eos_token tanımlamalıdır")
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer_sha256 = tokenizer_fingerprint(tokenizer)
    manifest_path = os.path.join(dataset_path, "dataset_manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError(
            "dataset_manifest.json bulunamadı. Dataset'i güncel "
            "data/tokenize_dataset.py ile yeniden üretin."
        )
    
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        dataset_manifest = json.load(manifest_file)
    
    expected_tokenizer_sha = dataset_manifest.get("tokenizer_sha256")
    if expected_tokenizer_sha != tokenizer_sha256:
        raise ValueError(
            "Tokenizer/dataset kimliği uyuşmuyor: "
            f"dataset={expected_tokenizer_sha}, tokenizer={tokenizer_sha256}"
        )
    
    expected_seq_length = config_dict["seq_length"]
    actual_seq_length = dataset_manifest.get("seq_length")
    if actual_seq_length != expected_seq_length:
        raise ValueError(
            "Dataset sequence uzunluğu model preset'i ile uyuşmuyor: "
            f"dataset={actual_seq_length}, model={expected_seq_length}"
        )
    
    # Model vocabulary boyutu tokenizer'dan gelir; sonradan resize edilmez.
    configured_vocab_size = config_dict["vocab_size"]
    config_dict["vocab_size"] = len(tokenizer)
    if configured_vocab_size != len(tokenizer):
        print(
            f"  ℹ️ Preset vocab_size {configured_vocab_size:,} → "
            f"{len(tokenizer):,} olarak tokenizer ile eşitlendi"
        )
    print(f"  ✅ Tokenizer: {len(tokenizer):,} token")
    print(f"  ✅ SHA-256: {tokenizer_sha256}")
    
    # ============================================================
    # MODEL OLUŞTUR (sıfırdan — random ağırlıklar)
    # ============================================================
    print("\n📦 Model oluşturuluyor (sıfırdan)...")
    
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        print("  ✅ Flash Attention 2 aktif")
    except ImportError:
        attn_impl = "sdpa"
        print("  ⚠️  Flash Attention yok, SDPA kullanılıyor")
    
    llama_config = LlamaConfig(
        vocab_size=config_dict["vocab_size"],
        hidden_size=config_dict["hidden_size"],
        intermediate_size=config_dict["intermediate_size"],
        num_hidden_layers=config_dict["num_hidden_layers"],
        num_attention_heads=config_dict["num_attention_heads"],
        num_key_value_heads=config_dict["num_key_value_heads"],
        max_position_embeddings=config_dict["max_position_embeddings"],
        rope_theta=config_dict["rope_theta"],
        hidden_act=config_dict["hidden_act"],
        rms_norm_eps=config_dict["rms_norm_eps"],
        tie_word_embeddings=config_dict["tie_word_embeddings"],
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    
    model = LlamaForCausalLM(llama_config)
    model = model.to(dtype=torch.bfloat16)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Toplam parametre:    {total_params / 1e6:.1f}M ({total_params / 1e9:.2f}B)")
    print(f"  Eğitilebilir param:  {trainable_params / 1e6:.1f}M")
    
    # ============================================================
    # VERİ SETİ
    # ============================================================
    print("\n📂 Veri seti yükleniyor...")
    
    try:
        train_dataset = load_from_disk(os.path.join(dataset_path, "train"))
        print(f"  ✅ Train: {len(train_dataset):,} sequence")
    except Exception:
        print("  ⚠️  Tokenized dataset bulunamadı!")
        print("  → Önce veri pipeline'ını çalıştırın:")
        print("     python data/download_datasets.py --mode quick")
        print("     python data/clean_turkish.py")
        print("     python data/dedup.py")
        print("     python data/tokenize_dataset.py")
        return
    
    try:
        eval_dataset = load_from_disk(os.path.join(dataset_path, "validation"))
        print(f"  ✅ Validation: {len(eval_dataset):,} sequence")
    except Exception:
        eval_dataset = None
        print("  ⚠️  Validation set bulunamadı, eval atlanıyor")
    
    # Eğitim lineage manifestini checkpoint dizinine kaydet.
    os.makedirs(output_dir, exist_ok=True)
    training_manifest = {
        "format_version": 1,
        "model_size": model_size,
        "model_config": config_dict,
        "tokenizer_path": os.path.abspath(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_manifest": dataset_manifest,
        "source_commit": os.environ.get("GIT_COMMIT"),
    }
    with open(
        os.path.join(output_dir, "training_manifest.json"),
        "w",
        encoding="utf-8",
    ) as training_manifest_file:
        json.dump(
            training_manifest,
            training_manifest_file,
            indent=2,
            ensure_ascii=False,
        )
    
    # ============================================================
    # EĞİTİM ARGÜMANLARI
    # ============================================================
    
    seq_len = config_dict["seq_length"]
    batch_size = config_dict["per_device_train_batch_size"]
    grad_accum = config_dict["gradient_accumulation_steps"]
    effective_batch_tokens = batch_size * grad_accum * seq_len
    total_tokens = config_dict["max_steps"] * effective_batch_tokens
    
    print(f"\n⚙️  Eğitim Konfigürasyonu:")
    print(f"  Max steps:           {config_dict['max_steps']:,}")
    print(f"  Batch size:          {batch_size} (per device)")
    print(f"  Grad accumulation:   {grad_accum}")
    print(f"  Efektif batch:       {effective_batch_tokens:,} token")
    print(f"  Toplam token hedefi: {total_tokens:,} ({total_tokens/1e9:.2f}B)")
    print(f"  Learning rate:       {config_dict['learning_rate']}")
    print(f"  Warmup steps:        {config_dict['warmup_steps']}")
    
    # WandB
    report_to = "wandb" if use_wandb else "none"
    run_name = f"llm-{model_size}-tr-en-scratch"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        
        # Steps
        max_steps=config_dict["max_steps"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        
        # LR Schedule
        learning_rate=config_dict["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=config_dict["warmup_steps"],
        weight_decay=0.1,
        max_grad_norm=1.0,
        
        # Precision & Memory
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        save_steps=500,
        save_total_limit=5,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        
        # WandB
        report_to=report_to,
        run_name=run_name,
        
        # DataLoader
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        
        # Optimizer
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        
        # Diğer
        seed=42,
        data_seed=42,
        include_tokens_per_second=True,
    )
    
    # ============================================================
    # DATA COLLATOR
    # ============================================================
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        pad_to_multiple_of=8,
    )
    
    # ============================================================
    # TRAINER
    # ============================================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[TrainingMonitorCallback()],
    )
    
    # ============================================================
    # EĞİTİM BAŞLAT
    # ============================================================
    
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                name=run_name,
                config={
                    "model_size": model_size,
                    "total_params": total_params,
                    "total_tokens_target": total_tokens,
                    **config_dict,
                },
            )
        except Exception as e:
            print(f"⚠️ WandB başlatılamadı: {e}")
    
    print("\n" + "=" * 60)
    print("🚀 SIFIRDAN EĞİTİM BAŞLIYOR")
    print("=" * 60 + "\n")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Modeli kaydet
    final_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    print(f"\n✅ Final model kaydedildi: {final_path}")
    
    # torch.compile versiyonunu da test et
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        print("✅ torch.compile başarılı (inference için kullanılabilir)")
    except Exception:
        print("⚠️  torch.compile atlanıyor")
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Sıfırdan LLM Eğitimi")
    parser.add_argument("--model-size", choices=["350m", "1b", "8b"], default="350m",
                        help="Model boyutu (2h H100 için '350m' önerilir)")
    parser.add_argument("--dataset", type=str, default="./data/tokenized_dataset",
                        help="Tokenized dataset yolu")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer",
                        help="Tokenizer yolu")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Checkpoint kayıt dizini")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint'tan devam et")
    parser.add_argument("--no-wandb", action="store_true",
                        help="WandB takibini kapat")
    parser.add_argument("--wandb-project", type=str, default="llm-tr-en-scratch",
                        help="WandB proje adı")
    args = parser.parse_args()
    
    train_model(
        model_size=args.model_size,
        dataset_path=args.dataset,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
        use_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
