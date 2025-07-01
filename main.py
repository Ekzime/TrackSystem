import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient

def detect_head_rotation(keypoints, confidences):
    """
    Определяет угол поворота головы через анализ ушей
    
    Args:
        keypoints: массив координат ключевых точек [x, y]
        confidences: массив уверенности для каждой точки
    
    Returns:
        float: коэффициент поворота (-1.0 до 1.0)
               -1.0 = поворот вправо (видно левое ухо)
               0.0 = анфас (оба уха видны одинаково)  
               1.0 = поворот влево (видно правое ухо)
    """
    LEFT_EAR_IDX = 3
    RIGHT_EAR_IDX = 4
    NOSE_IDX = 0
    
    MIN_CONFIDENCE = 0.3
    
    try:
        left_ear_conf = confidences[LEFT_EAR_IDX]
        right_ear_conf = confidences[RIGHT_EAR_IDX]
        nose_conf = confidences[NOSE_IDX]
        
        # Если нос не виден, не можем определить поворот
        if nose_conf < MIN_CONFIDENCE:
            return 0.0
            
        # Разность уверенности ушей
        ear_conf_diff = right_ear_conf - left_ear_conf
        
        # Нормализация в диапазон [-1, 1]  
        max_diff = 1.0  # максимальная разность уверенности
        rotation_coeff = np.clip(ear_conf_diff / max_diff, -1.0, 1.0)
        
        return rotation_coeff
        
    except (IndexError, TypeError):
        return 0.0

def get_adaptive_ratio(rotation_coeff):
    """
    Вычисляет адаптивный коэффициент смещения на основе поворота головы
    
    Args:
        rotation_coeff: коэффициент поворота от detect_head_rotation (-1.0 до 1.0)
    
    Returns:
        float: адаптивный коэффициент для neck_to_head_ratio
    """
    # Базовые коэффициенты для разных поворотов
    FRONTAL_RATIO = 0.35    # анфас
    SLIGHT_TURN_RATIO = 0.25  # легкий поворот
    STRONG_TURN_RATIO = 0.15  # сильный поворот
    PROFILE_RATIO = 0.08      # профиль
    
    abs_rotation = abs(rotation_coeff)
    
    if abs_rotation < 0.2:  # анфас
        return FRONTAL_RATIO
    elif abs_rotation < 0.5:  # легкий поворот
        # Интерполяция между анфас и легким поворотом
        t = (abs_rotation - 0.2) / 0.3
        return FRONTAL_RATIO * (1 - t) + SLIGHT_TURN_RATIO * t
    elif abs_rotation < 0.8:  # сильный поворот
        # Интерполяция между легким и сильным поворотом  
        t = (abs_rotation - 0.5) / 0.3
        return SLIGHT_TURN_RATIO * (1 - t) + STRONG_TURN_RATIO * t
    else:  # профиль
        # Интерполяция между сильным поворотом и профилем
        t = (abs_rotation - 0.8) / 0.2
        return STRONG_TURN_RATIO * (1 - t) + PROFILE_RATIO * t

def get_neck_mount_point(keypoints, confidences, adaptive_ratio):
    """
    Вычисляет стабильную точку крепления между плечами по направлению к носу
    
    Args:
        keypoints: массив координат ключевых точек [x, y]
        confidences: массив уверенности для каждой точки
        adaptive_ratio: адаптивный коэффициент смещения от плеч к носу
    
    Returns:
        tuple (x, y) или None если данные недостаточны
    """
    # Индексы ключевых точек YOLO pose
    NOSE_IDX = 0
    LEFT_SHOULDER_IDX = 5  
    RIGHT_SHOULDER_IDX = 6
    
    # Минимальная уверенность для использования точки
    MIN_CONFIDENCE = 0.5
    
    try:
        # Проверяем уверенность для ключевых точек
        if (confidences[NOSE_IDX] < MIN_CONFIDENCE or 
            confidences[LEFT_SHOULDER_IDX] < MIN_CONFIDENCE or 
            confidences[RIGHT_SHOULDER_IDX] < MIN_CONFIDENCE):
            return None
            
        # Получаем координаты
        nose = np.array(keypoints[NOSE_IDX])
        left_shoulder = np.array(keypoints[LEFT_SHOULDER_IDX])  
        right_shoulder = np.array(keypoints[RIGHT_SHOULDER_IDX])
        
        # Средняя точка между плечами
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
        
        # Вектор от плеч к носу
        shoulder_to_nose = nose - shoulder_midpoint
        
        # Финальная точка крепления с адаптивным коэффициентом
        neck_mount_point = shoulder_midpoint + (shoulder_to_nose * adaptive_ratio)
        
        return tuple(neck_mount_point.astype(int))
        
    except (IndexError, TypeError):
        return None

class PointStabilizer:
    def __init__(self, alpha=0.7):
        """
        Стабилизатор координат с экспоненциальным сглаживанием
        
        Args:
            alpha: коэффициент сглаживания (0.0-1.0), чем больше - тем быстрее реакция
        """
        self.alpha = alpha
        self.stable_point = None
        self.last_stable_point = None
        
    def update(self, new_raw_point):
        """
        Обновляет стабилизированную точку
        
        Args:
            new_raw_point: новые координаты (x, y) или None
        """
        if new_raw_point is None:
            # Если детекция потеряна, возвращаем последнюю стабильную точку
            return self.last_stable_point
            
        if self.stable_point is None:
            # Первая точка - инициализируем
            self.stable_point = np.array(new_raw_point, dtype=float)
        else:
            # Экспоненциальное скользящее среднее
            new_point = np.array(new_raw_point, dtype=float)
            self.stable_point = self.alpha * new_point + (1 - self.alpha) * self.stable_point
            
        self.last_stable_point = self.get_point_as_int_tuple()
        return self.last_stable_point
    
    def get_point_as_int_tuple(self):
        """
        Возвращает стабилизированную точку как кортеж целых чисел
        
        Returns:
            tuple (x, y) или None
        """
        if self.stable_point is None:
            return None
        return tuple(self.stable_point.astype(int))

class ValueStabilizer:
    """
    Стабилизатор для числовых значений (например, коэффициента поворота)
    """
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.stable_value = None
        
    def update(self, new_value):
        if self.stable_value is None:
            self.stable_value = new_value 
        else:
            self.stable_value = self.alpha * new_value + (1 - self.alpha) * self.stable_value
        return self.stable_value

class PoseTracker:
    def __init__(self):
        # Принудительно используем только GPU
        if not torch.cuda.is_available():
            print("CUDA не доступна! Установите CUDA драйверы.")
            return
            
        self.device = 'cuda'
        print(f"Принудительно используем: {self.device}")
        
        # Загружаем модель на GPU
        self.model = YOLO('yolov8s-pose.pt')
        self.model.to(self.device)
        
        self.cap = cv2.VideoCapture(5)
        
        if not self.cap.isOpened():
            print("Не удалось открыть веб-камеру")
            return
            
        # Для подсчета FPS
        self.prev_time = time.time()
        
        # Устанавливаем разрешение 
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Уменьшено с 640
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)  # Уменьшено с 320
        
        # Данные для OSC
        self.osc_data = {'neck_mount': None}
        
        # Стабилизаторы
        self.neck_stabilizer = PointStabilizer(alpha=0.6)
        self.rotation_stabilizer = ValueStabilizer(alpha=0.5)
        self.ratio_stabilizer = ValueStabilizer(alpha=0.4)
        
        # OSC клиент
        self.osc_client = SimpleUDPClient("127.0.0.1", 9001)
        
    def run(self):
        print("Запуск трекинга позы...")
        print("Нажмите ESC для выхода")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Зеркальное отражение
            frame = cv2.flip(frame, 1)
            
            # Конвертируем в серый для нейросети (1 канал вместо 3)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Конвертируем обратно в 3-канальный для совместимости с YOLO
            gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            
            # Получаем размеры кадра
            frame_height, frame_width = frame.shape[:2]
                    
            # Детекция позы принудительно на GPU - используем серый кадр
            results = self.model(gray_frame_3ch, device='cuda')
            
            # Рисуем результаты (стандартная визуализация YOLO)
            annotated_frame = results[0].plot()
            
            # Извлекаем keypoints и confidences один раз за кадр
            keypoints_data = None
            confidences_data = None
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                keypoints_data = results[0].keypoints.xy[0].cpu().numpy()  # [17, 2]
                confidences_data = results[0].keypoints.conf[0].cpu().numpy()  # [17]
            
            # Обработка keypoints для расчета точки крепления
            raw_neck_mount = None
            rotation_coeff = 0.0
            adaptive_ratio = 0.3
            
            if keypoints_data is not None and confidences_data is not None:
                # Определяем поворот головы
                raw_rotation = detect_head_rotation(keypoints_data, confidences_data)
                rotation_coeff = self.rotation_stabilizer.update(raw_rotation)
                
                # Вычисляем адаптивный коэффициент
                raw_ratio = get_adaptive_ratio(rotation_coeff)
                adaptive_ratio = self.ratio_stabilizer.update(raw_ratio)
                
                # Вычисляем сырую точку крепления с адаптивным коэффициентом
                raw_neck_mount = get_neck_mount_point(keypoints_data, confidences_data, adaptive_ratio)
            
            # Стабилизируем координаты
            stable_neck_mount = self.neck_stabilizer.update(raw_neck_mount)
            
            if stable_neck_mount is not None:
                # Сохраняем стабилизированную точку для OSC
                self.osc_data['neck_mount'] = stable_neck_mount
                
                # Нормализуем координаты для OSC
                normalized_x = stable_neck_mount[0] / frame_width
                normalized_y = stable_neck_mount[1] / frame_height
                
                # Отправляем через OSC
                self.osc_client.send_message("/yolo/neck_anchor", [normalized_x, normalized_y])
                
                # Визуализация стабилизированной точки крепления
                cv2.circle(annotated_frame, stable_neck_mount, 8, (0, 0, 255), -1)
                
                # Рисуем соединительные линии со скелетом (если есть сырые данные)
                if raw_neck_mount is not None and keypoints_data is not None and confidences_data is not None:
                    nose = tuple(keypoints_data[0].astype(int))
                    left_shoulder = tuple(keypoints_data[5].astype(int))
                    right_shoulder = tuple(keypoints_data[6].astype(int))
                    
                    # Линии от стабилизированной точки к ключевым точкам
                    if confidences_data[0] > 0.5:  # к носу
                        cv2.line(annotated_frame, stable_neck_mount, nose, (0, 0, 255), 2)
                    if confidences_data[5] > 0.5:  # к левому плечу
                        cv2.line(annotated_frame, stable_neck_mount, left_shoulder, (0, 0, 255), 2)
                    if confidences_data[6] > 0.5:  # к правому плечу
                        cv2.line(annotated_frame, stable_neck_mount, right_shoulder, (0, 0, 255), 2)
                
                # Отображаем нормализованные координаты для OSC
                osc_text = f'OSC: ({normalized_x:.2f}, {normalized_y:.2f})'
                cv2.putText(annotated_frame, osc_text, (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                self.osc_data['neck_mount'] = None
            
            # Подсчет и отображение FPS
            current_time = time.time()
            fps = 1 / (current_time - self.prev_time)
            self.prev_time = current_time
            
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Показываем устройство
            cv2.putText(annotated_frame, 'Device: GPU', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Показываем статус точки крепления
            status = "TRACKING" if self.osc_data['neck_mount'] is not None else "NO TRACK"
            color = (0, 255, 0) if self.osc_data['neck_mount'] is not None else (0, 0, 255)
            cv2.putText(annotated_frame, f'Neck Mount: {status}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Показываем информацию о повороте и адаптивном коэффициенте
            rotation_text = f'Rotation: {rotation_coeff:.2f}'
            cv2.putText(annotated_frame, rotation_text, (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            ratio_text = f'Adaptive Ratio: {adaptive_ratio:.2f}'
            cv2.putText(annotated_frame, ratio_text, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('YOLO v8 Pose Tracking', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PoseTracker()
    tracker.run() 