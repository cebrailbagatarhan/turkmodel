"""
tokenize_dataset.py — Veri Setini Tokenize Et
================================================
Temizlenmiş veriyi tokenizer ile tokenize edip eğitime hazır hale getirir.

Kullanım:
    python data/tokenize_dataset.py \
        --input ./data/deduped_dataset \
        --tokenizer ./tokenizer \
        --output ./data/tokenized_dataset \
        --seq-length 2048
"""

import os
import argparse
from itertools import chain

from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm


def tokenize_and_group(
    input_path: str,
    tokenizer_path: str,
    output_path: str,
    seq_length: int = 2048,
    num_proc: int = 4,
):
    """
    Veri setini tokenize et ve seq_length uzunluğunda gruplara böl.
    """
    print(f"📂 Veri seti: {input_path}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    print(f"📏 Sequence uzunluğu: {seq_length}")
    
    # Tokenizer yükle
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        # Eğer özel tokenizer yoksa, mevcut bir tokenizer kullan
        print("⚠️  Özel tokenizer bulunamadı, GPT-2 tokenizer kullanılıyor (geçici çözüm)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Veri seti yükle
    dataset = load_from_disk(input_path)
    print(f"  Toplam belge: {len(dataset):,}")
    
    # Tokenizasyon fonksiyonu
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=False,       # Truncation yapmıyoruz, gruplamada keseceğiz
            padding=False,
            return_attention_mask=False,
        )
    
    # Tokenize et
    print("🔤 Tokenizasyon başlıyor...")
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenize ediliyor",
    )
    
    # Gruplama fonksiyonu — sequence'ları birleştirip sabir uzunlukta kes
    def group_texts(examples):
        # Tüm token'ları birleştir
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        
        # seq_length'in katlarına kes
        total_length = (total_length // seq_length) * seq_length
        
        # Gruplara böl
        result = {
            k: [concatenated[k][i:i + seq_length] for i in range(0, total_length, seq_length)]
            for k in concatenated.keys()
        }
        
        # Labels = input_ids (causal LM)
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    # Grupla
    print("📦 Sequence'lar gruplanıyor...")
    grouped = tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc="Gruplanıyor",
    )
    
    # Train/Validation böl (%95 / %5)
    split = grouped.train_test_split(test_size=0.05, seed=42)
    split_dataset = {
        "train": split["train"],
        "validation": split["test"],
    }
    
    # Kaydet
    os.makedirs(output_path, exist_ok=True)
    for split_name, ds in split_dataset.items():
        save_path = os.path.join(output_path, split_name)
        ds.save_to_disk(save_path)
    
    # İstatistikler
    train_tokens = len(split_dataset["train"]) * seq_length
    val_tokens = len(split_dataset["validation"]) * seq_length
    
    print(f"\n📊 Tokenizasyon Sonuçları:")
    print(f"  Train sequence sayısı:   {len(split_dataset['train']):,}")
    print(f"  Val sequence sayısı:     {len(split_dataset['validation']):,}")
    print(f"  Train token (tahmini):   {train_tokens:,} ({train_tokens/1e6:.1f}M)")
    print(f"  Val token (tahmini):     {val_tokens:,} ({val_tokens/1e6:.1f}M)")
    print(f"  Sequence uzunluğu:       {seq_length}")
    print(f"  Kaydedildi:              {output_path}")
    
    return split_dataset


def main():
    parser = argparse.ArgumentParser(description="Veri Seti Tokenizasyonu")
    parser.add_argument("--input", type=str, default="./data/deduped_dataset")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer")
    parser.add_argument("--output", type=str, default="./data/tokenized_dataset")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()
    
    tokenize_and_group(
        args.input, args.tokenizer, args.output,
        args.seq_length, args.num_proc,
    )


if __name__ == "__main__":
    main()
