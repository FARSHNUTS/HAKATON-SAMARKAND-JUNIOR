import math

def get_box_center_pixels(bbox, vid_w, vid_h):
    """ Перевод координат Саши в пиксели """
    cx_norm, cy_norm, w_norm, h_norm = bbox
    
    cx = cx_norm * vid_w
    cy = cy_norm * vid_h
    w_px = w_norm * vid_w
    h_px = h_norm * vid_h
    
    x1 = int(cx - w_px / 2)
    y1 = int(cy - h_px / 2)
    x2 = int(cx + w_px / 2)
    y2 = int(cy + h_px / 2)
    
    return (x1, y1, x2, y2), (int(cx), int(cy))

def get_point(skeleton_dict, part_name):
    """ Безопасно достает точку [x, y] """
    if not skeleton_dict: return None
    pt = skeleton_dict.get(part_name)
    # Отсеиваем пустые точки или нули
    if not pt or len(pt) < 2 or (pt[0] == 0 and pt[1] == 0):
        return None
    return pt

def match_skeleton_to_box(bbox_coords, fedya_people, vid_w, vid_h):
    """ Ищет скелет внутри коробки """
    if not fedya_people: return None

    x1, y1, x2, y2 = bbox_coords
    box_center_x = (x1 + x2) / 2
    box_center_y = (y1 + y2) / 2

    best_person_data = None
    min_dist = float('inf')

    for person in fedya_people:
        skeleton = person.get('skeleton', {})
        
        # Проверяем основные точки
        anchors = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        anchor_point = None
        
        for part in anchors:
            pt = get_point(skeleton, part)
            if pt:
                anchor_point = pt
                break
        
        if not anchor_point: continue

        px, py = anchor_point[0], anchor_point[1]
        
        # Нормализация
        if px < 2.0: px *= vid_w
        if py < 2.0: py *= vid_h

        # Если точка попадает в коробку (с запасом 50px)
        margin = 50 
        if (x1 - margin) < px < (x2 + margin) and (y1 - margin) < py < (y2 + margin):
            dist = math.hypot(box_center_x - px, box_center_y - py)
            if dist < min_dist:
                min_dist = dist
                best_person_data = skeleton

    return best_person_data

def check_fall_by_skeleton(skeleton_dict, vid_h):
    """ ПРОВЕРКА ПАДЕНИЯ (По скелету) """
    if not skeleton_dict: return False
    
    # 1. Голова
    head_parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']
    y_head = None
    for part in head_parts:
        pt = get_point(skeleton_dict, part)
        if pt:
            y_head = pt[1]
            break 
            
    if y_head is None: return False

    # 2. Ноги
    y_feet = None
    l_ankle = get_point(skeleton_dict, 'left_ankle')
    r_ankle = get_point(skeleton_dict, 'right_ankle')
    if l_ankle and r_ankle: y_feet = max(l_ankle[1], r_ankle[1])
    elif l_ankle: y_feet = l_ankle[1]
    elif r_ankle: y_feet = r_ankle[1]
    
    # Если лодыжек нет - считаем, что не упал (безопасно)
    if y_feet is None: return False 

    # 3. Высота
    height_diff = abs(y_feet - y_head)
    if height_diff < (vid_h * 0.05): return False

    # 4. Пропорции
    all_x = []
    for k, v in skeleton_dict.items():
        if v and len(v) >= 2 and v[0] > 0: all_x.append(v[0])
    
    if not all_x: return False
    min_x, max_x = min(all_x), max(all_x)
    person_width = max_x - min_x
    
    is_wide = person_width > (height_diff * 1.2)
    is_low = height_diff < (vid_h * 0.15)
    
    if is_wide and is_low:
        return True
             
    return False

def check_working_by_skeleton(skeleton_dict):
    """ ПРОВЕРКА РАБОТЫ: Руки выше бедер """
    if not skeleton_dict: return False
    
    l_wrist = get_point(skeleton_dict, 'left_wrist')
    r_wrist = get_point(skeleton_dict, 'right_wrist')
    l_hip = get_point(skeleton_dict, 'left_hip')
    r_hip = get_point(skeleton_dict, 'right_hip')
    
    y_hip_level = None
    if l_hip and r_hip: y_hip_level = (l_hip[1] + r_hip[1]) / 2
    elif l_hip: y_hip_level = l_hip[1]
    elif r_hip: y_hip_level = r_hip[1]
    
    if y_hip_level is None: return False

    # Руки выше пояса (меньше Y)
    if l_wrist and l_wrist[1] < y_hip_level: return True
    if r_wrist and r_wrist[1] < y_hip_level: return True
    
    return False

def analyze_worker_activity(worker_id, bbox_coords, skeleton_dict, vid_h):
    x1, y1, x2, y2 = bbox_coords
    w = x2 - x1
    h = y2 - y1

    # --- 1. ЕСЛИ СКЕЛЕТ ЕСТЬ (Точная логика) ---
    if skeleton_dict:
        if check_fall_by_skeleton(skeleton_dict, vid_h):
            return "DANGEROUS SITUATION"
        
        if check_working_by_skeleton(skeleton_dict):
            return "WORKING"
            
        return "IDLE" # Скелет есть, стоит ровно, руки опущены

    # --- 2. ЕСЛИ СКЕЛЕТА НЕТ (Простейшая логика по квадрату) ---
    # Мы смотрим на геометрию квадрата Саши
    
    # Если ширина коробки в 1.5 раза больше высоты -> Лежит
    if w > h * 1.5:
        return "DANGEROUS SITUATION"
    
    # Если ширина обычная -> Стоит
    return "IDLE"