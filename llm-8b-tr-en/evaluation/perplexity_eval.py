"""
perplexity_eval.py — Perplexity Değerlendirmesi
=================================================
Eğitilmiş modelin perplexity skorunu ölç.

Kullanım:
    python evaluation/perplexity_eval.py \
        --model ./checkpoints/llm-350m-tr-en/final_model \
        --dataset ./data/tokenized_dataset/validation
"""

import os
import argparse
import math

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_from_disk
from tqdm import tqdm


def calculate_perplexity(
    model_path: str,
    dataset_path: str,
    batch_size: int = 8,
    max_samples: int = 1000,
):
    """
    Perplexity hesapla.
    Düşük perplexity = iyi model (dili iyi öğrenmiş).
    """
    
    print(f"📦 Model: {model_path}")
    print(f"📂 Dataset: {dataset_path}")
    
    # Model yükle
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset yükle
    dataset = load_from_disk(dataset_path)
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    print(f"  Değerlendirme örnekleri: {len(dataset):,}")
    
    # DataLoader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
    )
    
    # Perplexity hesapla
    total_loss = 0.0
    total_tokens = 0
    
    print("\n📊 Perplexity hesaplanıyor...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            # Token sayısı (padding hariç)
            num_tokens = (batch["labels"] != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Ortalama loss ve perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(min(avg_loss, 100))  # overflow koruması
    
    print(f"\n{'=' * 50}")
    print(f"📊 DEĞERLENDİRME SONUÇLARI")
    print(f"{'=' * 50}")
    print(f"  Ortalama Loss:  {avg_loss:.4f}")
    print(f"  Perplexity:     {perplexity:.2f}")
    print(f"  Toplam Token:   {total_tokens:,}")
    print(f"{'=' * 50}")
    
    # Referans değerler
    print(f"\n📈 Referans Perplexity Değerleri (350M model):")
    print(f"  Başlangıç (random):  ~30,000+")
    print(f"  1K step sonrası:     ~300-500")
    print(f"  3K step sonrası:     ~50-100")
    print(f"  5K step sonrası:     ~20-50 (hedef)")
    print(f"  İyi model:           <20")
    
    return {"loss": avg_loss, "perplexity": perplexity, "total_tokens": total_tokens}


def main():
    parser = argparse.ArgumentParser(description="Perplexity Değerlendirmesi")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="./data/tokenized_dataset/validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()
    
    calculate_perplexity(args.model, args.dataset, args.batch_size, args.max_samples)


if __name__ == "__main__":
    main()
