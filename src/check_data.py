import os
import yaml
from pathlib import Path

# Путь к папке data
DATA_DIR = Path("data")

def check_dataset():
    print("🔍 Проверка структуры датасета...")
    
    # Проверяем наличие папок
    for split in ["train", "valid", "test"]:
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"
        
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"❌ Ошибка: не найдены папки для {split}")
            return False
            
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        labels = list(lbl_dir.glob("*.txt"))
        
        print(f"✅ {split}: найдено {len(images)} картинок и {len(labels)} файлов разметки.")
        
        # Проверяем, что у каждой картинки есть разметка
        missing_labels = 0
        for img in images:
            lbl_file = lbl_dir / (img.stem + ".txt")
            if not lbl_file.exists():
                missing_labels += 1
        
        if missing_labels > 0:
            print(f"⚠️ Внимание: у {missing_labels} картинок нет файлов разметки!")
        else:
            print(f"✅ У всех картинок в {split} есть разметка.")

    # Проверяем YAML файл
    yaml_path = DATA_DIR / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            data_config = yaml.safe_load(f)
        print(f"✅ Файл data.yaml найден. Классы: {data_config.get('names')}")
    else:
        print("❌ Файл data.yaml не найден в папке data!")

if __name__ == "__main__":
    check_dataset()