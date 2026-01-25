"""
Skin Image Quality Enrichment using Google Gemini API

Analyzes dermoscopic images for quality artifacts:
- blurry, hairy, bubble, ruler, vignette, gel_border
"""
import google.generativeai as genai
import pandas as pd
import os
import sys
import time
import json
from tqdm import tqdm

# Import configuration
from .config import (
    GOOGLE_API_KEY,
    MODEL_NAME,
    IMAGE_ARTIFACTS,
    METADATA_CSV,
    METADATA_ENRICHED_CSV,
    REQUEST_DELAY,
    RETRY_ATTEMPTS,
    RETRY_DELAY_BASE,
    DEFAULT_DAILY_LIMIT
)


def setup_model():
    """Initialize Gemini model with API key"""
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(MODEL_NAME)


def analyze_image_with_retry(model, img_path, retries=RETRY_ATTEMPTS):
    """
    Analyze image with LLM, retry on failure with exponential backoff.
    
    Args:
        model: Generative AI model instance
        img_path: Path to image file
        retries: Number of retry attempts
        
    Returns:
        dict: Analysis results or None if all retries failed
    """
    
    prompt = f"""
    Sen uzman bir dermatologsun. Bu dermoskopik görüntüyü teknik kalite açısından analiz et.
    Sadece aşağıdaki görsel kusurlar var mı yok mu (1 veya 0) karar ver.
    
    Listemiz:
    - blurry: Görüntü bulanık mı?
    - hairy: Görüntüde dikkat dağıtıcı kıl/tüy var mı?
    - bubble: Hava kabarcıkları var mı?
    - ruler: Cetvel, kalem izi veya siyah çerçeve kenarı var mı?
    - vignette: Köşelerde kararma (vinyet) var mı?
    - gel_border: Jel sınırları veya sıvı baloncuk kenarları var mı?

    Cevabını SADECE şu JSON formatında ver, başka hiçbir şey yazma:
    {{
        "blurry": 0,
        "hairy": 0,
        "bubble": 0,
        "ruler": 0,
        "vignette": 0,
        "gel_border": 0
    }}
    """

    for attempt in range(retries):
        try:
            # Upload file
            sample_file = genai.upload_file(path=img_path, display_name="Skin Image")
            
            # Analyze
            response = model.generate_content([sample_file, prompt])
            
            # Parse JSON response
            text = response.text.replace("```json", "").replace("```", "").strip()
            result = json.loads(text)
            
            # Clean up cloud storage
            sample_file.delete()
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            # Check for quota errors
            if "429" in error_msg or "quota" in error_msg.lower():
                # Exponential backoff: 30s, 60s, 120s, 240s, 480s
                wait_time = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"\n⚠️  QUOTA AŞILDI! {wait_time} saniye bekleniyor... ({attempt+1}/{retries})")
                print(f"💡 İpucu: API kota limitleri aşıldı. Ücretli plana geçmeyi düşünün.")
                time.sleep(wait_time)
            else:
                # Other errors - shorter wait
                wait_time = 10 * (attempt + 1)
                print(f"\n❌ Hata: {error_msg}")
                print(f"↻ {wait_time} saniye sonra tekrar deneniyor... ({attempt+1}/{retries})")
                time.sleep(wait_time)
            
    return None


def get_unprocessed_count(df, artifacts):
    """Count unprocessed rows in dataframe"""
    unprocessed = 0
    for _, row in df.iterrows():
        if all(row[col] == 0 for col in artifacts):
            if os.path.exists(row['filename']):
                unprocessed += 1
    return unprocessed


def main(daily_limit=DEFAULT_DAILY_LIMIT):
    """
    Main processing function.
    
    Args:
        daily_limit: Maximum images to process per day
    """
    # Read CSV
    if not os.path.exists(METADATA_CSV):
        print(f"❌ CSV dosyası bulunamadı: {METADATA_CSV}")
        return
        
    df = pd.read_csv(METADATA_CSV)
    
    # Fix Windows paths
    df['filename'] = df['filename'].apply(lambda x: x.replace('\\', '/'))
    
    # Add artifact columns if missing
    for col in IMAGE_ARTIFACTS:
        if col not in df.columns:
            df[col] = 0

    # Resume from previous session
    if os.path.exists(METADATA_ENRICHED_CSV):
        print("📂 Önceki çalışma bulundu, kaldığı yerden devam ediliyor...")
        df_existing = pd.read_csv(METADATA_ENRICHED_CSV)
        df.update(df_existing)
    
    model = setup_model()
    
    # Show statistics
    unprocessed = get_unprocessed_count(df, IMAGE_ARTIFACTS)
    print(f"\n📊 İstatistik:")
    print(f"   Toplam resim: {len(df)}")
    print(f"   İşlenmemiş: {unprocessed}")
    print(f"   Günlük limit: {daily_limit}")
    print(f"   Kalan gün: ~{(unprocessed + daily_limit - 1) // daily_limit}")
    print(f"\n☕ Bugün analiz başlıyor...\n")

    processed_today = 0
    
    # Process images
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Stop if daily limit reached
        if processed_today >= daily_limit:
            print(f"\n✅ Bugünün limiti tamamlandı! ({processed_today}/{daily_limit})")
            print(f"⏰ Yarın tekrar çalıştırabilirsin.")
            break
        
        img_path = row['filename']
        
        # Skip if already processed
        if any(row[col] == 1 for col in IMAGE_ARTIFACTS):
            continue
        
        # Check if file exists
        if not os.path.exists(img_path):
            continue

        # Rate limit protection (RPM limit: ~15/min = 1 per 4 seconds, safe: 6 seconds)
        time.sleep(REQUEST_DELAY)
        
        # Analyze
        result = analyze_image_with_retry(model, img_path)
        
        if result:
            # Update dataframe
            for key in IMAGE_ARTIFACTS:
                val = result.get(key, 0)
                df.at[index, key] = val
            processed_today += 1
                
        # Save every 3 images (protect against power loss)
        if processed_today % 3 == 0:
            df.to_csv(METADATA_ENRICHED_CSV, index=False)
            print(f"   💾 İlerleme kaydedildi ({processed_today}/{daily_limit})")

    # Final save
    df.to_csv(METADATA_ENRICHED_CSV, index=False)
    print(f"\n✅ Günlük işlem tamamlandı: {processed_today} resim işlendi")
    print(f"📁 Dosya: {METADATA_ENRICHED_CSV}")


if __name__ == "__main__":
    # Allow custom daily limit from command line
    daily_limit = DEFAULT_DAILY_LIMIT
    if len(sys.argv) > 1:
        try:
            daily_limit = int(sys.argv[1])
            print(f"📌 Günlük limit: {daily_limit} resim\n")
        except ValueError:
            print("⚠️  Geçersiz limit. Örnek: python -m src.enrich_labels_with_llm 35")
    
    main(daily_limit=daily_limit)
