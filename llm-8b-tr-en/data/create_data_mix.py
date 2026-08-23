"""
create_data_mix.py — Dil Karışım Oranlarını Uygula
=====================================================
%80 Türkçe / %15 İngilizce / %5 Kod oranında veri karıştır.

Kullanım:
    python data/create_data_mix.py --input ./data/deduped_dataset --output ./data/mixed_dataset
"""

import os
import argparse
import random
from collections import defaultdict

from datasets import load_from_disk


# Hedef karışım oranları
TARGET_MIX = {
    "tr": 0.80,     # %80 Türkçe
    "en": 0.15,     # %15 İngilizce
    "code": 0.05,   # %5 Kod
}


def validate_mix_ratios(mix_ratios: dict) -> None:
    """Karışım oranlarının pozitif ve toplamda 1 olduğunu doğrula."""
    if not mix_ratios:
        raise ValueError("En az bir karışım oranı verilmelidir")
    if any(ratio <= 0 for ratio in mix_ratios.values()):
        raise ValueError("Tüm karışım oranları pozitif olmalıdır")
    total = sum(mix_ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Karışım oranları toplamı 1 olmalıdır; alınan: {total}")


def create_data_mix(
    input_path: str,
    output_path: str,
    mix_ratios: dict = None,
    seed: int = 42,
):
    """
    Dil karışım oranlarına göre veri setini dengele.
    """
    if mix_ratios is None:
        mix_ratios = TARGET_MIX
    validate_mix_ratios(mix_ratios)
    
    random.seed(seed)
    
    print(f"📂 Veri seti yükleniyor: {input_path}")
    dataset = load_from_disk(input_path)
    
    # Dil bazında ayır
    lang_groups = defaultdict(list)
    for i, item in enumerate(dataset):
        lang = item.get("lang", "unknown")
        lang_groups[lang].append(i)
    
    print(f"\n📊 Mevcut Dağılım:")
    total = len(dataset)
    for lang, indices in sorted(lang_groups.items()):
        pct = len(indices) / total * 100
        print(f"  {lang}: {len(indices):,} ({pct:.1f}%)")
    
    # Hedef oranları hesapla
    # En küçük grubu temel al
    target_sizes = {}
    min_ratio_group = None
    min_ratio_available = float("inf")
    
    for lang, ratio in mix_ratios.items():
        available = len(lang_groups.get(lang, []))
        if available > 0:
            available_at_ratio = available / ratio
            if available_at_ratio < min_ratio_available:
                min_ratio_available = available_at_ratio
                min_ratio_group = lang
    
    if min_ratio_group is None:
        print("❌ Hedef dillerde veri bulunamadı!")
        return
    
    # Her dilin hedef boyutunu hesapla
    for lang, ratio in mix_ratios.items():
        available = len(lang_groups.get(lang, []))
        target = int(min_ratio_available * ratio)
        target_sizes[lang] = min(target, available)
    
    print(f"\n🎯 Hedef Karışım:")
    mixed_indices = []
    for lang, target in sorted(target_sizes.items()):
        available = len(lang_groups.get(lang, []))
        if available > 0 and target > 0:
            # Rastgele örnekle
            indices = lang_groups[lang]
            if target <= available:
                sampled = random.sample(indices, target)
            else:
                sampled = indices  # Yeterli yoksa hepsini al
            mixed_indices.extend(sampled)
            new_pct = len(sampled) / max(1, sum(target_sizes.values())) * 100
            print(f"  {lang}: {len(sampled):,} / {available:,} ({new_pct:.1f}%)")
    
    # Karıştır
    random.shuffle(mixed_indices)
    
    # Yeni veri seti oluştur
    mixed_dataset = dataset.select(mixed_indices)
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    mixed_dataset.save_to_disk(output_path)
    
    print(f"\n✅ Karışık veri seti kaydedildi:")
    print(f"  Toplam: {len(mixed_dataset):,} belge")
    print(f"  Yol: {output_path}")
    
    return mixed_dataset


def main():
    parser = argparse.ArgumentParser(description="Dil Karışım Oranları")
    parser.add_argument("--input", type=str, default="./data/deduped_dataset")
    parser.add_argument("--output", type=str, default="./data/mixed_dataset")
    parser.add_argument("--tr-ratio", type=float, default=0.80)
    parser.add_argument("--en-ratio", type=float, default=0.15)
    parser.add_argument("--code-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    mix_ratios = {
        "tr": args.tr_ratio,
        "en": args.en_ratio,
        "code": args.code_ratio,
    }
    
    create_data_mix(args.input, args.output, mix_ratios, args.seed)


if __name__ == "__main__":
    main()
