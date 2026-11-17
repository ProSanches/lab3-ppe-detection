import os
from ultralytics import YOLO

def run_e0():
    print("🚀 Запуск E0: Проверка предобученной модели yolov8n.pt...")
    
    # Путь к предобученной модели
    model_path = "models/yolov8n.pt"
    
    # Загружаем модель
    model = YOLO(model_path)
    
    # Путь к тестовым изображениям
    test_images_dir = "data/test/images"
    
    # Получаем список всех картинок и берём первые 10
    all_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.png'))]
    sample_images = all_images[:10]
    
    # Запускаем предсказание
    # save=True сохранит картинки с нарисованными рамками в папку runs/
    results = model.predict(
        source=sample_images,
        save=True,
        project="runs/ppe",
        name="e0_pretrained",
        exist_ok=True
    )
    
    print(f"\n✅ Готово! Обработано {len(sample_images)} изображений.")
    print(f"Результаты сохранены в папку: runs/ppe/e0_pretrained")
    print("\n💡 Задание для отчёта:")
    print("Открой сохранённые картинки и посмотри, какие объекты нашла модель.")
    print("Скорее всего, она найдёт 'Person' (человека), но НЕ найдёт 'Hardhat' (каску) или 'Safety Vest' (жилет),")
    print("потому что стандартная модель обучалась на других классах (COCO).")

if __name__ == "__main__":
    run_e0()