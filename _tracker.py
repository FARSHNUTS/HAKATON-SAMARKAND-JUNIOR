import os
import shutil
import json
from ultralytics import YOLO

# --- ГЛОБАЛЬНЫЕ НАСТРОЙКИ (КОНФИГУРАЦИЯ) ---

# ПУТЬ К ВИДЕО (Абсолютный путь для гарантии)
# Убедись, что файл input_video.mp4 лежит в C:\hackathon_sibintec
VIDEO_PATH_ABS = "C:/hackathon_sibintec/input_video.mp4" 

# ПУТЬ К МОДЕЛИ (Упрощенный, так как ты его скопировал в папку проекта)
MODEL_NAME = 'best.pt' 
# Список классов для JSON-отчета
CLASSES = ['signalman', 'janitor', 'lightmechanic', 'darkmechanic', 'train', 'manager']

OUTPUT_FOLDER = 'output_data_final'
JSON_NAME = 'final_report.json'

# --- ОСНОВНАЯ ЛОГИКА ---
# --- (ИЗМЕНЕНИЯ ВНОСИМ ТОЛЬКО ВНУТРИ ФУНКЦИИ main) ---

def main():
    # ... (начальные проверки и загрузка модели) ...

    print(f"🎥 Начинаю ТУРБО-ТРЕКИНГ и генерацию JSON...")
    
    json_results = []
    
    # 3. ЗАПУСК ТРЕКИНГА (УСКОРЕНИЕ)
    results_generator = model.track(
        source=VIDEO_PATH_ABS,         
        tracker='bytetrack.yaml',      
        save=True,                     
        save_txt=False,                
        project=os.path.join(script_dir, 'runs'),
        name='detect',
        exist_ok=True,
        
        # 👇 ФИНАЛЬНЫЕ НАСТРОЙКИ СТАБИЛЬНОСТИ И СКОРОСТИ 👇
        conf=0.35,                     
        iou=0.6,                       
        agnostic_nms=True,             
        
        imgsz=640,                     # 🔥 1. УСКОРЕНИЕ: СНИЖАЕМ РАЗРЕШЕНИЕ (x3-x4 быстрее)
        vid_stride=3,                  # 🔥 2. УСКОРЕНИЕ: ОБРАБАТЫВАЕМ ТОЛЬКО КАЖДЫЙ ТРЕТИЙ КАДР
        device='cuda',                 # 🔥 3. ГАРАНТИЯ: ЯВНО УКАЗЫВАЕМ GPU
        
        stream=True,                   
        verbose=False
    )

    # 4. Сбор данных в JSON
    # ... (Остальной код сбора данных и сохранения JSON остается без изменений) ...
    for frame_idx, result in enumerate(results_generator):
        if frame_idx % 100 == 0:
            print(f"   ...обработано {frame_idx * 3} кадров (виртуально)...") # Обновленный счетчик
        
        # ... (логика сбора данных) ...
        # ... (Код ниже полностью твой)

        for box in result.boxes:
            track_id = int(box.id.item()) if box.id is not None else -1
            cls_id = int(box.cls.item())
            
            # Назначение роли по классу
            role_name = CLASSES[cls_id] if cls_id < len(CLASSES) else "unknown"
            
            coords = box.xywhn[0].tolist() # x, y, w, h
            
            json_results.append({
                "frame": frame_idx,
                "id": track_id,
                "role": role_name,
                "bbox": coords
            })

    print("\n✅ Обработка видео завершена!")

    # 5. Сохранение JSON и уборка папок
    json_path = os.path.join(script_dir, JSON_NAME)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=4)

    final_output = os.path.join(script_dir, OUTPUT_FOLDER)
    if os.path.exists(final_output): shutil.rmtree(final_output)
    
    source_dir = os.path.join(script_dir, 'runs', 'detect')
    
    if os.path.exists(source_dir):
        os.rename(source_dir, final_output)
        shutil.move(json_path, os.path.join(final_output, JSON_NAME))
        print(f"------------------------------------------------")
        print(f"🎉 ПОБЕДА! Всё готово в папке: {OUTPUT_FOLDER}")
        print("------------------------------------------------")
    else:
        print("⚠️ Ошибка: Папка с результатами не найдена. Проверь 'runs'.")


if __name__ == "__main__":
    main()