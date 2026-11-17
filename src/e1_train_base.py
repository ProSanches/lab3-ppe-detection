from ultralytics import YOLO

def run_e1():
    print("🚀 Запуск E1: Базовое обучение модели YOLOv8n...")
    
    # Загружаем предобученную модель
    model = YOLO("models/yolov8n.pt")
    
    # Запускаем обучение
    # Мы используем 10 эпох и imgsz=320 из-за ограничений CPU
    model.train(
        data="configs/ppe.yaml",
        epochs=10,          # Фиксированное решение: 10 эпох для CPU
        imgsz=320,          # Фиксированное решение: 320 вместо 640
        batch=8,            # Размер пакета
        seed=42,            # Для воспроизводимости (требование лабораторной)
        deterministic=True, # Для воспроизводимости
        project="runs",     # Папка для сохранения
        name="ppe/e1_base", # Имя эксперимента
        exist_ok=True,
        workers=2           # Количество потоков для загрузки данных
    )
    
    print("\n✅ Обучение E1 завершено!")
    print("Веса сохранены в: runs/ppe/e1_base/weights/best.pt")

if __name__ == "__main__":
    run_e1()