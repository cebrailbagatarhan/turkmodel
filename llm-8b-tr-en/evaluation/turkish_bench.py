"""
turkish_bench.py — Türkçe Kalite Değerlendirmesi
==================================================
Eğitim sırasında ve sonrasında Türkçe çıktı kalitesini ölç.

Kullanım:
    python evaluation/turkish_bench.py --model ./checkpoints/llm-350m-tr-en/final_model
"""

import os
import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# TÜRKÇE DEĞERLENDİRME PROMPT'LARI
# ============================================================

TURKISH_EVAL_PROMPTS = {
    "dilbilgisi": [
        {"prompt": "Türkiye'nin başkenti", "expected_keywords": ["Ankara"]},
        {"prompt": "İstanbul Boğazı hangi iki denizi birbirine bağlar?", 
         "expected_keywords": ["Karadeniz", "Marmara"]},
        {"prompt": "Dünya'nın en büyük kıtası", "expected_keywords": ["Asya"]},
    ],
    "cekim_ekleri": [
        {"prompt": "Çocuklar parka", "expected_keywords": ["gittiler", "gitti", "gidiyor"]},
        {"prompt": "Bu kitabı okumak", "expected_keywords": ["istiyorum", "isterim", "güzel"]},
        {"prompt": "Yarın hava nasıl olacak", "expected_keywords": ["güneşli", "yağmurlu", "sıcak", "soğuk"]},
    ],
    "soru_cevap": [
        {"prompt": "Atatürk ne zaman doğdu?", "expected_keywords": ["1881"]},
        {"prompt": "Python'da bir liste nasıl oluşturulur?", 
         "expected_keywords": ["list", "[]", "append"]},
        {"prompt": "Yapay zeka nedir?", 
         "expected_keywords": ["bilgisayar", "makine", "öğrenme", "zeka"]},
    ],
    "yaratici_yazarlik": [
        {"prompt": "Bir sonbahar günü hakkında kısa bir paragraf yaz:", "expected_keywords": []},
        {"prompt": "İstanbul hakkında üç cümle:", "expected_keywords": ["İstanbul"]},
        {"prompt": "Merhaba! Benim adım", "expected_keywords": []},
    ],
    "devam_ettirme": [
        {"prompt": "Türk mutfağının en bilinen yemeklerinden biri olan", "expected_keywords": []},
        {"prompt": "Bilgisayar biliminin temel konularından biri", "expected_keywords": []},
        {"prompt": "Günlük hayatta teknoloji kullanımı", "expected_keywords": []},
    ],
}


def evaluate_model(
    model_path: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "auto",
):
    """
    Modeli Türkçe prompt'larla değerlendir.
    """
    
    print(f"📦 Model yükleniyor: {model_path}")
    
    # Model yükle
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device if torch.cuda.is_available() else "cpu",
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    total_score = 0
    total_tests = 0
    
    print("\n" + "=" * 70)
    print("📝 TÜRKÇE KALİTE DEĞERLENDİRMESİ")
    print("=" * 70)
    
    for category, prompts in TURKISH_EVAL_PROMPTS.items():
        print(f"\n📂 Kategori: {category.upper()}")
        print("-" * 50)
        
        category_results = []
        
        for item in prompts:
            prompt = item["prompt"]
            expected = item["expected_keywords"]
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt):].strip()
            
            # Basit skor hesapla
            score = 0
            if expected:
                matches = sum(1 for kw in expected if kw.lower() in generated.lower())
                score = matches / len(expected) * 100
            else:
                # Anahtar kelime yoksa, uzunluk ve Türkçe karakter kontrolü
                if len(generated.split()) >= 5:
                    score += 50
                tr_chars = sum(c in "çğıöşüÇĞİÖŞÜ" for c in generated)
                if tr_chars > 0:
                    score += 50
            
            total_score += score
            total_tests += 1
            
            status = "✅" if score >= 50 else "⚠️" if score > 0 else "❌"
            
            print(f"\n  {status} Prompt: {prompt}")
            print(f"     Çıktı:  {generated[:200]}{'...' if len(generated) > 200 else ''}")
            if expected:
                print(f"     Beklenen: {expected}")
            print(f"     Skor:   {score:.0f}%")
            
            category_results.append({
                "prompt": prompt,
                "generated": generated,
                "expected": expected,
                "score": score,
            })
        
        results[category] = category_results
    
    # Genel skor
    avg_score = total_score / total_tests if total_tests > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"📊 GENEL SKOR: {avg_score:.1f}% ({total_tests} test)")
    print("=" * 70)
    
    if avg_score >= 70:
        print("🏆 İyi kalite — Model Türkçe'yi öğrenmeye başlamış!")
    elif avg_score >= 40:
        print("📈 Orta kalite — Daha fazla eğitim gerekiyor")
    else:
        print("📉 Düşük kalite — Model henüz başlangıç aşamasında")
    
    # Sonuçları dosyaya kaydet
    output_file = os.path.join(model_path, "turkish_eval_results.json")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "average_score": avg_score,
                "total_tests": total_tests,
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Sonuçlar kaydedildi: {output_file}")
    except Exception:
        pass
    
    return results, avg_score


def main():
    parser = argparse.ArgumentParser(description="Türkçe Kalite Değerlendirmesi")
    parser.add_argument("--model", type=str, required=True,
                        help="Model yolu")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maksimum üretilecek token")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    
    evaluate_model(
        args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
