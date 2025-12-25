import os
import shutil
import random
from pathlib import Path

# ========== CONFIGURATION ==========

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT_DIR = CURRENT_FILE.parent.parent

TARGET_FOLDER = PROJECT_ROOT_DIR / "data"

SOURCE_FOLDER = Path(r"C:\proje\datasetv2") 

SPLIT_RATIOS = (0.70, 0.15, 0.15)

# ====================================================

def distribute_data():
    print(f"--- PATH VERIFICATION ---")
    print(f"Working Directory  : {PROJECT_ROOT_DIR}")
    print(f"Source Data Path   : {SOURCE_FOLDER}")
    print(f"Target Data Path   : {TARGET_FOLDER}")
    print(f"--------------------")

    # 1. Verify source folder
    if not SOURCE_FOLDER.exists():
        print(f"ERROR: Source folder not found!\nLooking for: {SOURCE_FOLDER}")
        print("Please ensure 'Dataset' folder exists inside C:\\proje directory.")
        return

    # 2. Clean target folder (delete if exists, recreate)
    if TARGET_FOLDER.exists():
        try:
            shutil.rmtree(TARGET_FOLDER)
            print("Old data folder cleaned.")
        except Exception as e:
            print(f"Warning: Old folder could not be deleted ({e}), will overwrite.")
    
    # 3. Find classes
    classes = [x.name for x in SOURCE_FOLDER.iterdir() if x.is_dir()]
    print(f"Found Classes: {classes}")

    if not classes:
        print("ERROR: No classes (subfolders) found in source folder!")
        return

    # 4. Start distribution
    for class_name in classes:
        print(f"\nProcessing class '{class_name}' ...")
        
        source_class_path = SOURCE_FOLDER / class_name
        # Get only image files
        images = [f.name for f in source_class_path.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        random.shuffle(images)
        
        total = len(images)
        train_end = int(total * SPLIT_RATIOS[0])
        val_end = train_end + int(total * SPLIT_RATIOS[1])
        
        splits = {
            'train': images[:train_end],
            'validation': images[train_end:val_end],
            'test': images[val_end:]
        }
        
        for split_type, file_list in splits.items():
            # Create path using path object
            target_path = TARGET_FOLDER / split_type / class_name
            
            # Create folder
            target_path.mkdir(parents=True, exist_ok=True)
            
            for image_name in file_list:
                src = source_class_path / image_name
                dst = target_path / image_name
                shutil.copy2(src, dst)
            
            print(f"   - {split_type}: {len(file_list)} files.")

    print("\nPROCESS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    distribute_data()