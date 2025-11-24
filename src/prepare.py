import os
import shutil
import random
from pathlib import Path

# ================= OTOMATİK AYARLAR =================

# 1. Bu dosyanın (prepare.py) nerede olduğunu bul
MEVCUT_DOSYA = Path(__file__).resolve()
# Bir üst klasöre çık (src) -> Bir üstüne daha çık (skin-noise-detection)
PROJE_ANA_DIZIN = MEVCUT_DOSYA.parent.parent

# 2. Hedef Klasörü Otomatik Belirle (skin-noise-detection/data)
HEDEF_KLASOR = PROJE_ANA_DIZIN / "data"

# 3. Kaynak Klasör (Bunu C:\Proje\Dataset olarak varsayıyorum)
# Eğer klasör adınız farklıysa sadece tırnak içini değiştirin.
KAYNAK_KLASOR = Path(r"C:\proje\dataset") 

ORANLAR = (0.70, 0.15, 0.15)

# ====================================================

def verileri_dagit():
    print(f"--- YOL KONTROLÜ ---")
    print(f"Kodun Çalıştığı Yer: {PROJE_ANA_DIZIN}")
    print(f"Kaynak Veri Yolu   : {KAYNAK_KLASOR}")
    print(f"Hedef Veri Yolu    : {HEDEF_KLASOR}")
    print(f"--------------------")

    # 1. Kaynak Kontrolü
    if not KAYNAK_KLASOR.exists():
        print(f"HATA: Kaynak klasör bulunamadı!\nAranan yer: {KAYNAK_KLASOR}")
        print("Lütfen C:\\Proje klasörünün içinde 'Dataset' adında klasör olduğundan emin olun.")
        return

    # 2. Hedef Klasörü Temizle (Varsa sil, yeniden oluştur)
    if HEDEF_KLASOR.exists():
        try:
            shutil.rmtree(HEDEF_KLASOR)
            print("Eski data klasörü temizlendi.")
        except Exception as e:
            print(f"Uyarı: Eski klasör silinemedi ({e}), üzerine yazılacak.")
    
    # 3. Sınıfları Bul
    siniflar = [x.name for x in KAYNAK_KLASOR.iterdir() if x.is_dir()]
    print(f"Bulunan Sınıflar: {siniflar}")

    if not siniflar:
        print("HATA: Kaynak klasörde hiç sınıf (alt klasör) bulunamadı!")
        return

    # 4. Dağıtım Başlıyor
    for sinif in siniflar:
        print(f"\n--> '{sinif}' sınıfı işleniyor...")
        
        kaynak_sinif_yolu = KAYNAK_KLASOR / sinif
        # Sadece resim dosyalarını al
        resimler = [f.name for f in kaynak_sinif_yolu.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        random.shuffle(resimler)
        
        toplam = len(resimler)
        train_end = int(toplam * ORANLAR[0])
        val_end = train_end + int(toplam * ORANLAR[1])
        
        splitler = {
            'train': resimler[:train_end],
            'validation': resimler[train_end:val_end],
            'test': resimler[val_end:]
        }
        
        for tip, liste in splitler.items():
            # Yolu path objesi ile oluştur (Hata riskini sıfırlar)
            hedef_yol = HEDEF_KLASOR / tip / sinif
            
            # Klasörü oluştur (parents=True demek, ara klasörleri de yarat demek)
            hedef_yol.mkdir(parents=True, exist_ok=True)
            
            for resim_adi in liste:
                src = kaynak_sinif_yolu / resim_adi
                dst = hedef_yol / resim_adi
                shutil.copy2(src, dst)
            
            print(f"   - {tip}: {len(liste)} adet.")

    print("\n✅ İŞLEM BAŞARIYLA TAMAMLANDI!")

if __name__ == "__main__":
    verileri_dagit()