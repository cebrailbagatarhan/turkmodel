"""
tokenize_dataset.py — Veri Setini Tokenize Et
================================================
Belgeleri train/validation olarak ayırdıktan sonra tokenize edip paketler.
Bu sıra, aynı belgenin parçalarının iki split'e sızmasını engeller.

Kullanım:
    python data/tokenize_dataset.py \
        --input ./data/mixed_dataset \
        --tokenizer ./tokenizer \
        --output ./data/tokenized_dataset \
        --seq-length 2048
"""

import argparse
import hashlib
import json
import os
from itertools import chain

from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer


def tokenizer_fingerprint(tokenizer) -> str:
    """Tokenizer vocabulary ve ayarları için kararlı SHA-256 üret."""
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


def _tokenize_split(dataset, tokenizer, seq_length: int, num_proc: int, name: str):
    """Tek bir belge split'ini tokenize edip sabit uzunlukta paketle."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer eos_token_id tanımlamalıdır")

    def tokenize_function(examples):
        encoded = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        # Belgeler arası geçişi açıkça sınırla.
        encoded["input_ids"] = [
            token_ids + [eos_token_id]
            for token_ids in encoded["input_ids"]
        ]
        return encoded

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"{name} tokenize ediliyor",
    )

    def group_texts(examples):
        concatenated = {
            key: list(chain.from_iterable(examples[key]))
            for key in examples
        }
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // seq_length) * seq_length

        result = {
            key: [
                concatenated[key][start:start + seq_length]
                for start in range(0, total_length, seq_length)
            ]
            for key in concatenated
        }
        result["labels"] = [
            list(token_ids)
            for token_ids in result["input_ids"]
        ]
        return result

    return tokenized.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        desc=f"{name} paketleniyor",
    )


def tokenize_and_group(
    input_path: str,
    tokenizer_path: str,
    output_path: str,
    seq_length: int = 2048,
    num_proc: int = 4,
    validation_ratio: float = 0.05,
    seed: int = 42,
):
    """Veriyi belge seviyesinde ayır, tokenize et ve paketle."""
    if seq_length <= 1:
        raise ValueError("seq_length 1'den büyük olmalıdır")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio 0 ile 1 arasında olmalıdır")

    print(f"📂 Veri seti: {input_path}")
    print(f"🔤 Tokenizer: {tokenizer_path}")
    print(f"📏 Sequence uzunluğu: {seq_length}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "Tokenizer yüklenemedi. Dataset daha önce hangi tokenizer ile "
            "üretildiyse aynı yerel tokenizer dizinini verin; otomatik GPT-2 "
            "fallback güvenli değildir."
        ) from exc

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer pad_token veya eos_token tanımlamalıdır")
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(input_path)
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError("DatasetDict girdisi 'train' split'i içermelidir")
        dataset = dataset["train"]

    if len(dataset) < 2:
        raise ValueError("Train/validation split için en az iki belge gereklidir")
    if "text" not in dataset.column_names:
        raise ValueError("Dataset 'text' sütunu içermelidir")

    print(f"  Toplam belge: {len(dataset):,}")

    validation_documents = max(1, round(len(dataset) * validation_ratio))
    if validation_documents >= len(dataset):
        validation_documents = 1

    # Kritik sıra: önce belge seviyesinde split, sonra tokenize/pack.
    document_split = dataset.train_test_split(
        test_size=validation_documents,
        seed=seed,
        shuffle=True,
    )
    train_documents = document_split["train"]
    validation_documents_ds = document_split["test"]

    print(
        "  Belge split'i: "
        f"{len(train_documents):,} train / "
        f"{len(validation_documents_ds):,} validation"
    )

    split_dataset = {
        "train": _tokenize_split(
            train_documents,
            tokenizer,
            seq_length,
            num_proc,
            "Train",
        ),
        "validation": _tokenize_split(
            validation_documents_ds,
            tokenizer,
            seq_length,
            num_proc,
            "Validation",
        ),
    }

    os.makedirs(output_path, exist_ok=True)
    for split_name, split_data in split_dataset.items():
        split_data.save_to_disk(os.path.join(output_path, split_name))

    fingerprint = tokenizer_fingerprint(tokenizer)
    manifest = {
        "format_version": 1,
        "source_path": os.path.abspath(input_path),
        "source_dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "tokenizer_path": os.path.abspath(tokenizer_path),
        "tokenizer_sha256": fingerprint,
        "tokenizer_vocab_size": len(tokenizer),
        "seq_length": seq_length,
        "split_seed": seed,
        "validation_ratio": validation_ratio,
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents_ds),
        "train_sequences": len(split_dataset["train"]),
        "validation_sequences": len(split_dataset["validation"]),
    }
    with open(
        os.path.join(output_path, "dataset_manifest.json"),
        "w",
        encoding="utf-8",
    ) as manifest_file:
        json.dump(manifest, manifest_file, indent=2, ensure_ascii=False)

    train_tokens = len(split_dataset["train"]) * seq_length
    validation_tokens = len(split_dataset["validation"]) * seq_length
    print("\n📊 Tokenizasyon Sonuçları:")
    print(f"  Train sequence:       {len(split_dataset['train']):,}")
    print(f"  Validation sequence:  {len(split_dataset['validation']):,}")
    print(f"  Train token:          {train_tokens:,}")
    print(f"  Validation token:     {validation_tokens:,}")
    print(f"  Tokenizer SHA-256:    {fingerprint}")
    print(f"  Kaydedildi:           {output_path}")

    return split_dataset


def main():
    parser = argparse.ArgumentParser(description="Veri Seti Tokenizasyonu")
    parser.add_argument("--input", default="./data/mixed_dataset")
    parser.add_argument("--tokenizer", default="./tokenizer")
    parser.add_argument("--output", default="./data/tokenized_dataset")
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenize_and_group(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output_path=args.output,
        seq_length=args.seq_length,
        num_proc=args.num_proc,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
