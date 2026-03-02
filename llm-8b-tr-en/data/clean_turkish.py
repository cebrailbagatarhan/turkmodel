"""
clean_turkish.py — Türkçe Metin Temizleme Pipeline
====================================================
Kalite filtresi, dil kontrolü, deduplikasyon.

Kullanım:
    python data/clean_turkish.py --input ./data/raw_dataset --output ./data/cleaned_dataset
"""

import re
import os
import argparse
from typing import Optional

from datasets import load_from_disk, Dataset
from tqdm import tqdm


# ============================================================
# TEMİZLEME FONKSİYONLARI
# ============================================================

# Türkçe özel karakterler
TR_CHARS = set("çğıöşüÇĞİÖŞÜ")

# Kaldırılacak pattern'lar
REMOVE_PATTERNS = [
    r"https?://\S+",                    # URL'ler
    r"www\.\S+",                        # www linkleri
    r"\S+@\S+\.\S+",                   # E-posta adresleri
    r"<[^>]+>",                         # HTML tag'leri
    r"\{\{[^}]+\}\}",                  # Wiki şablonları
    r"\[\[Kategori:[^\]]+\]\]",        # Wiki kategorileri
    r"\[\[Dosya:[^\]]+\]\]",           # Wiki dosya referansları
    r"={2,}[^=]+={2,}",               # Wiki başlıkları (==Başlık==)
    r"\|[^|]*\|",                       # Tablo hücreleri
    r"thumb\|",                         # Wiki thumbnail
]

COMPILED_PATTERNS = [re.compile(p) for p in REMOVE_PATTERNS]


def clean_text(text: str) -> str:
    """Temel metin temizleme"""
    # Pattern'ları kaldır
    for pattern in COMPILED_PATTERNS:
        text = pattern.sub(" ", text)
    
    # Fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Başlangıç/bitiş boşlukları
    text = text.strip()
    
    return text


def quality_filter(text: str, lang: str = "tr", min_words: int = 30) -> bool:
    """
    Kalite filtresi — düşük kaliteli belgeleri at.
    
    Returns:
        True: Belge kaliteli, kullan
        False: Belge düşük kaliteli, atla
    """
    if not text or not text.strip():
        return False
    
    words = text.split()
    
    # Minimum kelime sayısı
    if len(words) < min_words:
        return False
    
    # Maksimum kelime sayısı (çok uzun belgeler sorunlu olabilir)
    if len(words) > 100_000:
        return False
    
    # Benzersizlik oranı (spam filtresi)
    unique_words = set(w.lower() for w in words)
    unique_ratio = len(unique_words) / len(words)
    if unique_ratio < 0.2:  # %20'den az benzersiz kelime → muhtemelen spam
        return False
    
    # Ortalama kelime uzunluğu kontrolü
    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 2 or avg_word_len > 25:
        return False
    
    # Alfanümerik karakter oranı
    alpha_chars = sum(c.isalpha() for c in text)
    if len(text) > 0 and alpha_chars / len(text) < 0.5:
        return False
    
    # Türkçe spesifik kontroller
    if lang == "tr":
        tr_char_count = sum(c in TR_CHARS for c in text)
        # Türkçe metinde en az birkaç Türkçe özel karakter olmalı
        if len(text) > 200 and tr_char_count < 2:
            return False
    
    return True


def detect_language_simple(text: str) -> Optional[str]:
    """
    Basit dil tespiti (langdetect olmadan).
    Türkçe özel karakterlere bakarak hızlı kontrol.
    """
    sample = text[:1000]
    
    tr_indicators = ["bir", "ve", "bu", "için", "ile", "olan", "gibi", "daha", "çok", "sonra"]
    en_indicators = ["the", "and", "is", "for", "that", "with", "this", "from", "have", "are"]
    
    words_lower = sample.lower().split()
    word_set = set(words_lower[:200])
    
    tr_score = sum(1 for w in tr_indicators if w in word_set)
    en_score = sum(1 for w in en_indicators if w in word_set)
    
    # Türkçe karakter varlığı
    tr_chars_present = any(c in TR_CHARS for c in sample)
    if tr_chars_present:
        tr_score += 3
    
    if tr_score > en_score:
        return "tr"
    elif en_score > tr_score:
        return "en"
    else:
        return None


def process_dataset(input_path: str, output_path: str, min_words: int = 30):
    """
    Veri setini temizle ve filtrele.
    """
    print(f"📂 Veri seti yükleniyor: {input_path}")
    dataset = load_from_disk(input_path)
    
    original_count = len(dataset)
    print(f"  Toplam belge: {original_count:,}")
    
    cleaned_items = []
    stats = {
        "total": original_count,
        "cleaned": 0,
        "removed_quality": 0,
        "removed_short": 0,
        "removed_lang": 0,
    }
    
    print("🧹 Temizleme başlıyor...")
    for item in tqdm(dataset, desc="Temizleniyor"):
        text = item.get("text", "")
        lang = item.get("lang", "tr")
        source = item.get("source", "unknown")
        
        # Metni temizle
        cleaned = clean_text(text)
        
        # Kalite filtresi
        if not quality_filter(cleaned, lang=lang, min_words=min_words):
            stats["removed_quality"] += 1
            continue
        
        cleaned_items.append({
            "text": cleaned,
            "lang": lang,
            "source": source,
        })
        stats["cleaned"] += 1
    
    # Temizlenmiş veri setini kaydet
    if cleaned_items:
        cleaned_dataset = Dataset.from_list(cleaned_items)
        cleaned_dataset.save_to_disk(output_path)
        
        print(f"\n📊 Temizleme Sonuçları:")
        print(f"  Orijinal: {stats['total']:,}")
        print(f"  Temiz:    {stats['cleaned']:,} ({stats['cleaned']/stats['total']*100:.1f}%)")
        print(f"  Atılan:   {stats['removed_quality']:,}")
        print(f"  Kaydedildi: {output_path}")
    else:
        print("❌ Temiz belge bulunamadı!")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Türkçe Metin Temizleme")
    parser.add_argument("--input", type=str, default="./data/raw_dataset",
                        help="Girdi veri seti dizini")
    parser.add_argument("--output", type=str, default="./data/cleaned_dataset",
                        help="Çıktı dizini")
    parser.add_argument("--min-words", type=int, default=30,
                        help="Minimum kelime sayısı")
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.min_words)


if __name__ == "__main__":
    main()
