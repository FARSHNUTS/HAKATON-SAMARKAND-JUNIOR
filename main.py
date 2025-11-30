import cv2
import pandas as pd
import json
import os
from logic import analyze_worker_activity, match_skeleton_to_box, get_box_center_pixels

# --- КОНФИГУРАЦИЯ ---
VIDEO_PATH = "input_video.mov" 
SASHA_FILE = "final_report.json"
FEDYA_FILE = "fedya_pose.json"
OUTPUT_CSV = "hackathon_result.csv"
OUTPUT_VIDEO = "final_output.avi"

# Настройка синхронизации
DATA_SPEED_FACTOR = 0.33333 
SYNC_OFFSET = 0             

# Цвета (BGR формат: Blue, Green, Red)
COLOR_GREEN = (0, 255, 0)      # Зеленый
COLOR_RED   = (0, 0, 255)      # Красный
COLOR_YELLOW = (0, 255, 255)   # Желтый
COLOR_TRAIN = (128, 128, 128)  # Серый

def main():
    print("🚀 ЗАПУСК СИСТЕМЫ 4.0 (Новые цвета)...")

    # 1. Загрузка Саши
    print(f"📂 Читаем {SASHA_FILE}...")
    try:
        with open(SASHA_FILE, 'r') as f:
            sasha_data = json.load(f)
    except FileNotFoundError:
        print("❌ Нет файла Саши!")
        return

    sasha_by_frame = {}
    for item in sasha_data:
        f_num = int(item['frame'])
        if f_num not in sasha_by_frame: sasha_by_frame[f_num] = []
        sasha_by_frame[f_num].append(item)

    # 2. Загрузка Феди
    print(f"📂 Читаем {FEDYA_FILE}...")
    try:
        with open(FEDYA_FILE, 'r') as f:
            fedya_data = json.load(f)
    except FileNotFoundError:
        print("❌ Нет файла Феди!")
        return
        
    fedya_by_frame = {}
    for item in fedya_data:
        f_id = item.get('frame_id')
        people = item.get('people', [])
        fedya_by_frame[f_id] = people

    # 3. Видео
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Ошибка видео!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    frame_counter = 0
    all_events = []

    print("⏳ Обработка видео...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_counter += 1
        
        # Индексы
        sasha_idx = int(frame_counter * DATA_SPEED_FACTOR) + SYNC_OFFSET
        fedya_idx = frame_counter 

        if frame_counter % 50 == 0:
            print(f"Frame {frame_counter} -> Sasha {sasha_idx}", end='\r')

        # Данные
        fedya_people = fedya_by_frame.get(fedya_idx, [])
        sasha_objects = sasha_by_frame.get(sasha_idx, [])

        for obj in sasha_objects:
            role = obj.get('role', 'worker')
            if role is None: role = 'worker'
            p_id = obj.get('id', -1)
            
            # Координаты
            bbox_coords, _ = get_box_center_pixels(obj['bbox'], width, height)
            x1, y1, x2, y2 = bbox_coords

            # Поезд
            if role == 'train':
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_TRAIN, 2)
                cv2.putText(frame, f"TRAIN", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TRAIN, 2)
                continue

            # Ищем скелет
            matched_skeleton = match_skeleton_to_box(bbox_coords, fedya_people, width, height)
            
            # Логика
            status = analyze_worker_activity(p_id, bbox_coords, matched_skeleton, height)
            
            # 🔥 ОБНОВЛЕННАЯ ЛОГИКА ЦВЕТОВ 🔥
            if status == "DANGEROUS SITUATION":
                color = COLOR_RED
            elif status == "WORKING":
                color = COLOR_GREEN   # Теперь это ЗЕЛЕНЫЙ
            else: # IDLE
                color = COLOR_YELLOW  # Теперь это ЖЕЛТЫЙ
            
            # Рисуем Квадрат
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Текст
            label = f"ID:{p_id} {role} {status}"
            # Плашка под текст чуть шире
            cv2.rectangle(frame, (x1, y1-25), (x1+300, y1), color, -1)
            # Текст черным цветом (на желтом/зеленом лучше видно черный)
            text_color = (0, 0, 0) if status != "DANGEROUS SITUATION" else (255, 255, 255)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Отрисовка точек скелета
            if matched_skeleton:
                for part_name, xy in matched_skeleton.items():
                    if xy and len(xy) >= 2:
                        px, py = int(xy[0]), int(xy[1])
                        if px < 2000 and py < 2000:
                            # Точки скелета рисуем контрастным цветом (Синим)
                            cv2.circle(frame, (px, py), 3, (255, 0, 0), -1)

            # Запись в CSV
            all_events.append({
                "frame": frame_counter,
                "time_sec": round(frame_counter / fps, 2),
                "id": p_id,
                "role": role,
                "status": status,
                "danger": 1 if status == "DANGEROUS SITUATION" else 0
            })

        out.write(frame)

    cap.release()
    out.release()
    
    if all_events:
        df = pd.DataFrame(all_events)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✅ Готово! Файл: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()