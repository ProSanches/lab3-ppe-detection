from ultralytics import YOLO

def run_e2():
    print("🚀 Запуск E2: Обучение с увеличенным размером изображения (imgsz=640)...")
    
    # Загружаем ту же предобученную модель
    model = YOLO("models/yolov8n.pt")
    
    # Запускаем обучение с изменённым параметром imgsz
    model.train(
        data="configs/ppe.yaml",
        epochs=10,          # Оставляем столько же эпох для чистоты эксперимента
        imgsz=640,          # 👈 ИЗМЕНЕНИЕ: увеличили с 320 до 640
        batch=8,            
        seed=42,            
        deterministic=True, 
        project="runs",     
        name="ppe/e2_imgsz", # Новое имя для сохранения
        exist_ok=True,
        workers=2           
    )
    
    print("\n✅ Обучение E2 завершено!")
    print("Веса сохранены в: runs/ppe/e2_imgsz/weights/best.pt")

if __name__ == "__main__":
    run_e2()