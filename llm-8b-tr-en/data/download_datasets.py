"""
download_datasets.py — HuggingFace'den Türkçe + İngilizce veri çekme
=====================================================================
2 saat H100 senaryosu: Küçük örneklemler indirip hızlıca başla.
Tam eğitim için tüm verisetlerini indir.

Kullanım:
    python data/download_datasets.py --mode quick    # 2 saat için
    python data/download_datasets.py --mode full     # Tam eğitim için
"""

import os
import argparse
from datasets import load_dataset, concatenate_datasets, Dataset
from tqdm import tqdm

# ============================================================
# VERİ KAYNAKLARI
# ============================================================

TURKISH_SOURCES = {
    "wikipedia_tr": {
        "path": "wikimedia/wikipedia",
        "name": "20231101.tr",
        "split": "train",
        "text_field": "text",
        "description": "Türkçe Wikipedia (~350M token)",
    },
    "oscar_tr": {
        "path": "oscar-corpus/OSCAR-2301",
        "name": "tr",
        "split": "train",
        "text_field": "text",
        "trust_remote_code": True,
        "description": "Türkçe web crawl (~15B token)",
    },
}

ENGLISH_SOURCES = {
    "fineweb_edu_sample": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "description": "Yüksek kaliteli İngilizce web (~10B token sample)",
    },
}

CODE_SOURCES = {
    "python_code": {
        "path": "bigcode/starcoderdata",
        "data_dir": "python",
        "split": "train",
        "text_field": "content",
        "description": "Python kodu",
    },
}


def download_quick_mode(output_dir: str, num_samples: int = 500_000):
    """
    Hızlı mod: Her kaynaktan küçük örneklem al.
    2 saat H100 için yeterli veri (~500M-1B token).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_texts = []
    
    # --- Türkçe Wikipedia (kolay erişim) ---
    print("📥 Türkçe Wikipedia indiriliyor...")
    try:
        wiki_tr = load_dataset(
            "wikimedia/wikipedia",
            "20231101.tr",
            split=f"train[:{num_samples}]",
            trust_remote_code=True,
        )
        tr_texts = [{"text": item["text"], "lang": "tr", "source": "wikipedia_tr"} 
                     for item in wiki_tr if len(item["text"]) > 100]
        all_texts.extend(tr_texts)
        print(f"  ✅ {len(tr_texts):,} Türkçe belge alındı")
    except Exception as e:
        print(f"  ⚠️ Wikipedia-TR atlanıyor: {e}")
    
    # --- Türkçe OSCAR (web crawl) ---
    print("📥 Türkçe OSCAR indiriliyor...")
    try:
        oscar_tr = load_dataset(
            "oscar-corpus/OSCAR-2301",
            "tr",
            split=f"train[:{num_samples}]",
            trust_remote_code=True,
        )
        oscar_texts = [{"text": item["text"], "lang": "tr", "source": "oscar_tr"} 
                       for item in oscar_tr if len(item["text"]) > 100]
        all_texts.extend(oscar_texts)
        print(f"  ✅ {len(oscar_texts):,} Türkçe web belgesi alındı")
    except Exception as e:
        print(f"  ⚠️ OSCAR-TR atlanıyor: {e}")
        # Alternatif: CC-100
        print("📥 Alternatif: CC-100 Türkçe deneniyor...")
        try:
            cc100_tr = load_dataset("cc100", "tr", split=f"train[:{num_samples}]")
            cc_texts = [{"text": item["text"], "lang": "tr", "source": "cc100_tr"} 
                        for item in cc100_tr if len(item["text"]) > 100]
            all_texts.extend(cc_texts)
            print(f"  ✅ {len(cc_texts):,} Türkçe CC-100 belgesi alındı")
        except Exception as e2:
            print(f"  ⚠️ CC-100 de atlanıyor: {e2}")
    
    # --- İngilizce FineWeb-Edu ---
    print("📥 İngilizce FineWeb-Edu indiriliyor...")
    try:
        fineweb = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT",
            split=f"train[:{int(num_samples * 1.5)}]",  # İng biraz daha fazla
            trust_remote_code=True,
        )
        en_texts = [{"text": item["text"], "lang": "en", "source": "fineweb_edu"} 
                    for item in fineweb if len(item["text"]) > 100]
        all_texts.extend(en_texts)
        print(f"  ✅ {len(en_texts):,} İngilizce belge alındı")
    except Exception as e:
        print(f"  ⚠️ FineWeb-Edu atlanıyor: {e}")
        # Alternatif: C4
        print("📥 Alternatif: C4 deneniyor...")
        try:
            c4 = load_dataset("allenai/c4", "en", split=f"train[:{int(num_samples * 1.5)}]",
                              trust_remote_code=True)
            c4_texts = [{"text": item["text"], "lang": "en", "source": "c4"} 
                        for item in c4 if len(item["text"]) > 100]
            all_texts.extend(c4_texts)
            print(f"  ✅ {len(c4_texts):,} İngilizce C4 belgesi alındı")
        except Exception as e2:
            print(f"  ⚠️ C4 de atlanıyor: {e2}")
    
    # --- Python Kodu ---
    print("📥 Python kodu indiriliyor...")
    try:
        code = load_dataset(
            "bigcode/starcoderdata",
            data_dir="python",
            split=f"train[:{int(num_samples * 0.1)}]",
            trust_remote_code=True,
        )
        code_texts = [{"text": item["content"], "lang": "code", "source": "starcoder_python"} 
                      for item in code if len(item["content"]) > 50]
        all_texts.extend(code_texts)
        print(f"  ✅ {len(code_texts):,} Python kodu belgesi alındı")
    except Exception as e:
        print(f"  ⚠️ StarCoder atlanıyor: {e}")
    
    # Dataset oluştur ve kaydet
    if all_texts:
        dataset = Dataset.from_list(all_texts)
        dataset_path = os.path.join(output_dir, "raw_dataset")
        dataset.save_to_disk(dataset_path)
        
        # İstatistikler
        lang_counts = {}
        for item in all_texts:
            lang = item["lang"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        print(f"\n📊 Veri Seti Özeti:")
        print(f"  Toplam belge: {len(all_texts):,}")
        for lang, count in sorted(lang_counts.items()):
            pct = count / len(all_texts) * 100
            print(f"  {lang}: {count:,} ({pct:.1f}%)")
        print(f"  Kaydedildi: {dataset_path}")
        
        return dataset_path
    else:
        print("❌ Hiç veri indirilemedi!")
        return None


def download_full_mode(output_dir: str):
    """
    Tam mod: Tüm veri kaynaklarını indir.
    100B token hedefi — büyük disk alanı gerektirir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 TAM MOD — Tüm veri kaynakları indiriliyor...")
    print("⚠️  Bu işlem saatler sürebilir ve yüzlerce GB disk gerektirir!")
    
    # Türkçe kaynaklar
    for name, config in TURKISH_SOURCES.items():
        print(f"\n📥 {name} indiriliyor...")
        try:
            kwargs = {"path": config["path"], "split": config["split"]}
            if "name" in config:
                kwargs["name"] = config["name"]
            if config.get("trust_remote_code"):
                kwargs["trust_remote_code"] = True
            
            ds = load_dataset(**kwargs)
            save_path = os.path.join(output_dir, f"raw_{name}")
            ds.save_to_disk(save_path)
            print(f"  ✅ {name} kaydedildi: {save_path}")
        except Exception as e:
            print(f"  ⚠️ {name} atlanıyor: {e}")
    
    # İngilizce kaynaklar
    for name, config in ENGLISH_SOURCES.items():
        print(f"\n📥 {name} indiriliyor...")
        try:
            kwargs = {"path": config["path"], "split": config["split"]}
            if "name" in config:
                kwargs["name"] = config["name"]
            ds = load_dataset(**kwargs, trust_remote_code=True)
            save_path = os.path.join(output_dir, f"raw_{name}")
            ds.save_to_disk(save_path)
            print(f"  ✅ {name} kaydedildi: {save_path}")
        except Exception as e:
            print(f"  ⚠️ {name} atlanıyor: {e}")
    
    # Kod kaynakları
    for name, config in CODE_SOURCES.items():
        print(f"\n📥 {name} indiriliyor...")
        try:
            kwargs = {"path": config["path"], "split": config["split"]}
            if "name" in config:
                kwargs["name"] = config["name"]
            if "data_dir" in config:
                kwargs["data_dir"] = config["data_dir"]
            if config.get("trust_remote_code"):
                kwargs["trust_remote_code"] = True
            ds = load_dataset(**kwargs, trust_remote_code=True)
            save_path = os.path.join(output_dir, f"raw_{name}")
            ds.save_to_disk(save_path)
            print(f"  ✅ {name} kaydedildi: {save_path}")
        except Exception as e:
            print(f"  ⚠️ {name} atlanıyor: {e}")
    
    print("\n✅ Tam indirme tamamlandı!")


def main():
    parser = argparse.ArgumentParser(description="Veri Seti İndirme")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick: 2 saat H100 için | full: tam eğitim")
    parser.add_argument("--output-dir", type=str, default="./data",
                        help="Çıktı dizini")
    parser.add_argument("--num-samples", type=int, default=500_000,
                        help="Quick modda kaynak başına max örnek sayısı")
    args = parser.parse_args()
    
    if args.mode == "quick":
        download_quick_mode(args.output_dir, args.num_samples)
    else:
        download_full_mode(args.output_dir)


if __name__ == "__main__":
    main()
