"""
train_tokenizer.py — Türkçe + İngilizce BPE Tokenizer Eğitimi
==============================================================
2 saat H100 senaryosu için 32K vocab ile küçük tokenizer.
Tam 8B model için vocab_size=130_000 yapılabilir.

Kullanım:
    python tokenizer/train_tokenizer.py

Çıktı:
    ./tokenizer/tokenizer.json
    ./tokenizer/tokenizer_config.json
"""

import os
import json
import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

# ============================================================
# KONFİGÜRASYON
# ============================================================
VOCAB_SIZE = 32_000          # 2h versiyon: 32K | 8B versiyon: 130_000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = [
    "<|begin_of_text|>",     # BOS
    "<|end_of_text|>",       # EOS
    "<|pad|>",               # Padding
    "<|im_start|>",          # Chat mesaj başlangıcı
    "<|im_end|>",            # Chat mesaj sonu
    "<|unk|>",               # Bilinmeyen token
]


def create_sample_corpus(output_dir: str):
    """
    Eğer corpus dosyaları yoksa, küçük bir örnek oluştur.
    Gerçek eğitimde HuggingFace'den indirilen veriler kullanılır.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    tr_file = os.path.join(output_dir, "turkish_sample.txt")
    en_file = os.path.join(output_dir, "english_sample.txt")
    
    if not os.path.exists(tr_file):
        turkish_texts = [
            "Türkiye Cumhuriyeti, Avrupa ile Asya kıtalarının kesiştiği noktada yer alan bir ülkedir.",
            "İstanbul, hem Avrupa hem de Asya kıtasında toprakları olan tek şehirdir.",
            "Ankara, Türkiye'nin başkenti ve ikinci en büyük şehridir.",
            "Türk dili, Altay dil ailesi'ne ait Oğuz grubundan bir dildir.",
            "Yapay zeka ve makine öğrenmesi günümüzün en önemli teknoloji alanlarıdır.",
            "Doğal dil işleme, bilgisayarların insan dilini anlamasını sağlayan bir yapay zeka dalıdır.",
            "Büyük dil modelleri milyarlarca parametre ile eğitilir ve çeşitli görevlerde kullanılır.",
            "Python programlama dili, yapay zeka araştırmalarında en çok tercih edilen dillerden biridir.",
            "Transformer mimarisi, doğal dil işlemede devrim yaratan bir sinir ağı modelidir.",
            "Dikkat mekanizması, modelin girdi dizisindeki farklı konumlara odaklanmasını sağlar.",
        ]
        with open(tr_file, "w", encoding="utf-8") as f:
            f.write("\n".join(turkish_texts * 100))  # Tekrarla
    
    if not os.path.exists(en_file):
        english_texts = [
            "Artificial intelligence is transforming every industry around the world.",
            "Large language models are trained on billions of tokens from diverse sources.",
            "The transformer architecture uses self-attention to process sequential data.",
            "Natural language processing enables computers to understand human language.",
            "Machine learning algorithms can learn patterns from data without explicit programming.",
            "Deep learning models use multiple layers of neural networks for feature extraction.",
            "Tokenization is the process of breaking text into smaller units called tokens.",
            "Pre-training a language model from scratch requires significant computational resources.",
            "The attention mechanism allows models to focus on relevant parts of the input.",
            "Fine-tuning adapts a pre-trained model to specific downstream tasks.",
        ]
        with open(en_file, "w", encoding="utf-8") as f:
            f.write("\n".join(english_texts * 100))
    
    return [tr_file, en_file]


def train_tokenizer(corpus_files: list, output_dir: str, vocab_size: int = VOCAB_SIZE):
    """BPE tokenizer eğit"""
    
    print(f"🔤 Tokenizer eğitimi başlıyor...")
    print(f"   Vocab boyutu: {vocab_size:,}")
    print(f"   Corpus dosyaları: {corpus_files}")
    
    # BPE modeli oluştur
    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    
    # Pre-tokenizer: ByteLevel (UTF-8 uyumlu)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor (BOS/EOS ekleme)
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<|begin_of_text|> $A <|end_of_text|>",
        pair="<|begin_of_text|> $A <|end_of_text|> <|begin_of_text|> $B <|end_of_text|>",
        special_tokens=[
            ("<|begin_of_text|>", 0),
            ("<|end_of_text|>", 1),
        ],
    )
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    
    # Eğit
    tokenizer.train(files=corpus_files, trainer=trainer)
    
    # Kaydet
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    
    # Config dosyası oluştur 
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "model_max_length": 2048,
        "vocab_size": vocab_size,
    }
    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Tokenizer kaydedildi: {tokenizer_path}")
    print(f"✅ Config kaydedildi: {config_path}")
    
    # Test
    test_sentences = [
        "Merhaba dünya! Türkçe tokenizer testi.",
        "Hello world! English tokenizer test.",
        "İstanbul'un tarihi çok zengindir.",
        "def hello(): print('merhaba')",
    ]
    
    print("\n📝 Tokenizer Test Sonuçları:")
    print("-" * 60)
    for sent in test_sentences:
        encoded = tokenizer.encode(sent)
        print(f"Girdi: {sent}")
        print(f"Token: {encoded.tokens}")
        print(f"ID:    {encoded.ids}")
        print(f"Sayı:  {len(encoded.tokens)} token")
        print("-" * 60)
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="BPE Tokenizer Eğitimi")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"Vocab boyutu (varsayılan: {VOCAB_SIZE})")
    parser.add_argument("--corpus-dir", type=str, default="./data/corpus",
                        help="Corpus dosyalarının dizini")
    parser.add_argument("--output-dir", type=str, default="./tokenizer",
                        help="Çıktı dizini")
    parser.add_argument("--corpus-files", nargs="+", default=None,
                        help="Belirli corpus dosyaları (verilmezse sample oluşturulur)")
    args = parser.parse_args()
    
    if args.corpus_files:
        corpus_files = args.corpus_files
    else:
        print("⚠️  Corpus dosyası belirtilmedi, örnek corpus oluşturuluyor...")
        corpus_files = create_sample_corpus(args.corpus_dir)
    
    train_tokenizer(corpus_files, args.output_dir, args.vocab_size)


if __name__ == "__main__":
    main()
