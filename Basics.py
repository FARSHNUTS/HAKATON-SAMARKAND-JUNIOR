import cv2
import json
from ultralytics import YOLO

# --- НАСТРОЙКИ ---
video_path = 'input_video.mov'       # Имя твоего видео
output_file = 'fedya_pose.json' # Имя файла для Артёма
model_name = 'yolov8n-pose.pt' # 'n' - самая быстрая (Nano) модель

print(f"Загружаю модель {model_name} для скоростной обработки...")
model = YOLO(model_name)

# Карта точек тела (чтобы Артём понимал, где что, а не просто цифры)
KEYPOINT_NAMES = {
    0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
    9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
    13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
}

# Открываем видео
cap = cv2.VideoCapture(video_path)
data_for_json = [] # Сюда складываем всю инфу
frame_count = 0

print("Начинаю обработку видео...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # Запускаем YOLO. 
    # conf=0.3 -> игнорируем совсем мусорные детекции
    # verbose=False -> чтобы не засорять терминал
    results = model(frame, verbose=False, conf=0.3)
    
    # Структура данных для текущего кадра
    frame_data = {
        "frame_id": frame_count,
        "people": []
    }
    
    # Разбираем результаты
    for r in results:
        # Если есть скелеты
        if r.keypoints is not None and r.keypoints.xy is not None:
            # Получаем координаты скелетов (x, y)
            kps_list = r.keypoints.xy.cpu().numpy()
            # Получаем координаты боксов (x1, y1, x2, y2) - это нужно для сверки с Сашей
            boxes_list = r.boxes.xyxy.cpu().numpy()
            
            # Проходимся по каждому найденному человеку
            for i, kps in enumerate(kps_list):
                person_data = {
                    "person_index": i, # Просто порядковый номер на кадре
                    "box": boxes_list[i].tolist(), # [x1, y1, x2, y2]
                    "skeleton": {}
                }
                
                # Записываем точки тела
                for kp_id, (x, y) in enumerate(kps):
                    # Если x и y не равны 0 (значит точка найдена)
                    if x != 0 and y != 0:
                        part_name = KEYPOINT_NAMES.get(kp_id, f"point_{kp_id}")
                        person_data["skeleton"][part_name] = [float(x), float(y)]
                
                # Добавляем человека в список кадра, если у него нашлось хоть что-то
                if person_data["skeleton"]:
                    frame_data["people"].append(person_data)

    # Добавляем кадр в общий список
    data_for_json.append(frame_data)
    
    # Пишем прогресс каждые 100 кадров
    if frame_count % 100 == 0:
        print(f"Обработано кадров: {frame_count}...")

cap.release()

# --- СОХРАНЕНИЕ ---
print(f"Сохраняю данные в {output_file}...")
with open(output_file, 'w') as f:
    json.dump(data_for_json, f, indent=None) # indent=None для экономии места

print("Готово! JSON создан.")