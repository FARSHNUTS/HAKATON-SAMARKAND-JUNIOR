import cv2
import os
import shutil
import math

# --- КОНФИГУРАЦИЯ ---
VIDEO_FILENAME = 'input_video.mp4'
OUTPUT_DIR_NAME = 'full_dataset_1500' 
TARGET_FRAME_COUNT = 1500 
# Среднее количество кадров для 12-минутного видео при 30 FPS: 21600.
# Шаг (STEP) рассчитывается на основе этой оценки, чтобы получить 1500 кадров.
# 21600 / 1500 = 14.4. Мы округляем до 15.
ESTIMATED_SKIP_INTERVAL = 15 # Будем пропускать 14 кадров, сохраняя 15-й.

def extract_frames():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path_full = os.path.join(script_dir, VIDEO_FILENAME)
    output_folder_full = os.path.join(script_dir, OUTPUT_DIR_NAME)

    # 1. Setup
    if os.path.exists(output_folder_full):
        shutil.rmtree(output_folder_full)
    os.makedirs(output_folder_full, exist_ok=True)
    
    if not os.path.exists(video_path_full):
        print(f"\n❌ ОШИБКА: Видео '{VIDEO_FILENAME}' не найдено.")
        return

    cap = cv2.VideoCapture(video_path_full)
    
    if not cap.isOpened():
        print(f"\n❌ ОШИБКА: Видео найдено, но не открывается. Проверьте кодеки.")
        return

    # 2. Расчет шага
    # Мы игнорируем сломанный счетчик кадров и используем оценку:
    step = ESTIMATED_SKIP_INTERVAL 
    print(f"\n✅ Читаю видео. Буду сохранять каждый {step}-й кадр, чтобы получить ~1500.")

    # 3. Extraction Loop (Медленный, но стабильный способ чтения)
    saved_count = 0
    read_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Сохраняем только каждый "step"-й кадр
        if read_count % step == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            save_path = os.path.join(output_folder_full, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            print('.', end="", flush=True)

        read_count += 1
        
        if saved_count >= TARGET_FRAME_COUNT:
            break

    cap.release()
    print("\n" + "-" * 35)
    print(f"🎉 УСПЕХ! Сохранено {saved_count} целевых кадров.")
    print("-" * 35)

if __name__ == "__main__":
    extract_frames()