import time
import statistics
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def evaluate_test():
    print("🚀 Оценка лучшей модели (E2) на тестовой выборке...")
    
    # Путь к лучшей модели из E2
    model_path = "runs/detect/runs/ppe/e2_imgsz/weights/best.pt"
    model = YOLO(model_path)
    
    # 1. Оценка метрик на тесте
    print("\n📊 Считаем метрики на TEST...")
    metrics = model.val(
        data="configs/ppe.yaml",
        split="test",
        imgsz=640,
        project="runs",
        name="ppe/final_test",
        exist_ok=True
    )
    
    print(f"\nИТОГОВЫЕ МЕТРИКИ НА ТЕСТЕ:")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    
    # 2. Замер задержки (Latency)
    print("\n⏱️ Замер задержки на 20 картинках...")
    test_images_dir = Path("data/test/images")
    image_paths = list(test_images_dir.glob("*.jpg"))[:20]
    
    latencies = []
    for img_path in image_paths:
        start_time = time.perf_counter()
        _ = model.predict(str(img_path), imgsz=640, verbose=False)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # в миллисекундах
        
    median_latency = statistics.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"Медиана задержки: {median_latency:.2f} мс")
    print(f"95-й процентиль: {p95_latency:.2f} мс")
    print("\n✅ Оценка завершена. Результаты сохранены в runs/detect/runs/ppe/final_test")

if __name__ == "__main__":
    evaluate_test()