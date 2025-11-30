import os
import shutil
import random
import yaml

# --- ОБНОВЛЕННЫЕ КЛАССЫ ---
CLASSES = ['signalman', 'janitor', 'lightmechanic', 'darkmechanic', 'train', 'manager']

RAW_IMAGES = 'dataset_images'
RAW_LABELS = 'labels_raw'
DEST_FOLDER = 'yolo_dataset'

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_source = os.path.join(base_dir, RAW_IMAGES)
    labels_source = os.path.join(base_dir, RAW_LABELS)
    dest_dir = os.path.join(base_dir, DEST_FOLDER)

    if not os.path.exists(labels_source):
        print(f"❌ ОШИБКА: Папка '{RAW_LABELS}' пуста или не найдена.")
        return

    # Чистим старый датасет перед созданием нового
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    # 1. Структура папок
    for split in ['train', 'val']:
        for content in ['images', 'labels']:
            os.makedirs(os.path.join(dest_dir, content, split), exist_ok=True)

    # 2. Файлы
    files = [f for f in os.listdir(images_source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)

    # 3. Копирование
    split_index = int(len(files) * 0.8)
    train_files = files[:split_index]
    val_files = files[split_index:]

    print(f"📦 Всего файлов: {len(files)}")

    def copy_files(file_list, split_name):
        for filename in file_list:
            src_img = os.path.join(images_source, filename)
            dst_img = os.path.join(dest_dir, 'images', split_name, filename)
            
            name_without_ext = os.path.splitext(filename)[0]
            txt_name = name_without_ext + ".txt"
            src_txt = os.path.join(labels_source, txt_name)
            dst_txt = os.path.join(dest_dir, 'labels', split_name, txt_name)
            
            if os.path.exists(src_txt):
                shutil.copy(src_img, dst_img) # Копируем картинку только если есть метка
                shutil.copy(src_txt, dst_txt)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    # 4. Config
    yaml_content = {
        'path': dest_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    yaml_path = os.path.join(dest_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=None)

    print("✅ Датасет обновлен с новыми классами!")

if __name__ == "__main__":
    main()