"""
dedup.py — MinHash LSH ile Yakın Kopya Kaldırma
=================================================
Benzer belgeleri tespit edip kaldırır (threshold=0.85).

Kullanım:
    python data/dedup.py --input ./data/cleaned_dataset --output ./data/deduped_dataset
"""

import os
import argparse
from datasets import load_from_disk, Dataset
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Belge için MinHash oluştur"""
    m = MinHash(num_perm=num_perm)
    # 3-gram shingle kullan (kelime bazlı)
    words = text.lower().split()
    for i in range(len(words) - 2):
        shingle = " ".join(words[i:i+3])
        m.update(shingle.encode("utf-8"))
    return m


def deduplicate_dataset(
    input_path: str,
    output_path: str,
    threshold: float = 0.85,
    num_perm: int = 128,
):
    """
    MinHash LSH ile yakın kopya belgeler kaldır.
    
    Args:
        threshold: Benzerlik eşiği (0.85 = %85 benzer → kopya sayılır)
        num_perm: MinHash permütasyon sayısı (128 iyi denge)
    """
    print(f"📂 Veri seti yükleniyor: {input_path}")
    dataset = load_from_disk(input_path)
    
    original_count = len(dataset)
    print(f"  Toplam belge: {original_count:,}")
    print(f"  Benzerlik eşiği: {threshold}")
    
    # LSH index oluştur
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    unique_indices = []
    duplicate_count = 0
    empty_count = 0
    seen_short_texts = set()
    
    print("🔍 Deduplifikasyon başlıyor...")
    for i, item in enumerate(tqdm(dataset, desc="Kontrol ediliyor")):
        text = item.get("text", "")
        normalized_text = " ".join(text.lower().split())
        words = normalized_text.split()
        
        if not normalized_text:
            empty_count += 1
            continue
        
        # MinHash kısa metinlerde anlamlı 3-gram üretmez. Kısa ve benzersiz
        # örnekleri koru; yalnızca tam kopyalarını kaldır.
        if len(words) < 10:
            if normalized_text in seen_short_texts:
                duplicate_count += 1
            else:
                seen_short_texts.add(normalized_text)
                unique_indices.append(i)
            continue
        
        mh = create_minhash(normalized_text, num_perm)
        
        # Benzer belge var mı kontrol et
        result = lsh.query(mh)
        
        if not result:
            # Benzeri yok → benzersiz
            try:
                lsh.insert(f"doc_{i}", mh)
                unique_indices.append(i)
            except ValueError:
                # Aynı key zaten varsa atla
                pass
        else:
            duplicate_count += 1
    
    # Benzersiz belgeleri seç
    unique_dataset = dataset.select(unique_indices)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    unique_dataset.save_to_disk(output_path)
    
    removed_count = original_count - len(unique_indices)
    removed_pct = removed_count / original_count * 100 if original_count > 0 else 0
    print(f"\n📊 Deduplifikasyon Sonuçları:")
    print(f"  Orijinal:  {original_count:,}")
    print(f"  Benzersiz: {len(unique_indices):,}")
    print(f"  Yakın/tam kopya: {duplicate_count:,}")
    print(f"  Boş belge:       {empty_count:,}")
    print(f"  Kaldırılan:      {removed_count:,} ({removed_pct:.1f}%)")
    print(f"  Kaydedildi: {output_path}")
    
    return unique_dataset


def main():
    parser = argparse.ArgumentParser(description="MinHash Deduplifikasyon")
    parser.add_argument("--input", type=str, default="./data/cleaned_dataset")
    parser.add_argument("--output", type=str, default="./data/deduped_dataset")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--num-perm", type=int, default=128)
    args = parser.parse_args()
    
    deduplicate_dataset(args.input, args.output, args.threshold, args.num_perm)


if __name__ == "__main__":
    main()
