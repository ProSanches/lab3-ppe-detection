import cv2
import os
from pathlib import Path
from ultralytics import YOLO

def find_errors():
    print("🔍 Поиск ошибок модели на тестовой выборке...")
    model = YOLO("runs/detect/runs/ppe/e2_imgsz/weights/best.pt")
    
    # Создаём папку для галереи ошибок
    out_dir = Path("reports/error_gallery")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    test_images_dir = Path("data/test/images")
    image_paths = list(test_images_dir.glob("*.jpg"))
    
    false_positives = 0
    
    for img_path in image_paths:
        # Запускаем предсказание с порогом уверенности 0.5
        results = model.predict(str(img_path), conf=0.5, imgsz=640, verbose=False)[0]
        
        # Читаем оригинальное изображение
        img = cv2.imread(str(img_path))
        
        # Рисуем предсказания модели
        for box in results.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = results.names[cls_id]
            
            # Если модель очень уверена (conf > 0.8) в объекте, 
            # который редко встречается (например, если это не Person), сохраняем как потенциальную ошибку
            if conf > 0.8 and class_name in ['vehicle', 'NO-Mask', 'NO-Safety Vest']:
                # Сохраняем картинку с нарисованными рамками
                save_path = out_dir / f"error_{img_path.name}"
                results.save(filename=str(save_path))
                false_positives += 1
                break # Одна ошибка с картинки
                
        if false_positives >= 10:
            break
            
    print(f"✅ Найдено и сохранено {false_positives} примеров ошибок в папку reports/error_gallery")

if __name__ == "__main__":
    find_errors()