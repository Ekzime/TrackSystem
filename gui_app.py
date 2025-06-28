import tkinter as tk
from tkinter import ttk, scrolledtext
import cv2
import time
import torch
import numpy as np
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient
from PIL import Image, ImageTk
import threading
from datetime import datetime
import platform
import asyncio
import queue
import subprocess
import re

def get_windows_camera_names():
    """Получение имен камер в Windows через PowerShell"""
    try:
        # PowerShell команда для получения видеоустройств
        cmd = ['powershell', '-Command', 
               'Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.PNPClass -eq "Camera" -or $_.PNPClass -eq "Image"} | Select-Object Name, DeviceID']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            devices = {}
            lines = result.stdout.split('\n')
            current_name = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('Name'):
                    name_match = re.search(r'Name\s+:\s+(.+)', line)
                    if name_match:
                        current_name = name_match.group(1).strip()
                elif line.startswith('DeviceID') and current_name:
                    # Извлекаем номер из DeviceID
                    device_match = re.search(r'VID_[0-9A-F]+&PID_[0-9A-F]+.*?\\([0-9]+)', line)
                    if device_match:
                        device_num = len(devices)  # Используем порядковый номер
                        devices[device_num] = current_name
                    current_name = None
            
            return devices
    except Exception as e:
        print(f"Ошибка получения имен камер: {e}")
    
    return {}

def get_neck_mount_point(keypoints, confidences, neck_to_head_ratio=0.3):
    """
    Вычисляет стабильную точку крепления между плечами по направлению к носу
    """
    NOSE_IDX = 0
    LEFT_SHOULDER_IDX = 5  
    RIGHT_SHOULDER_IDX = 6
    MIN_CONFIDENCE = 0.5
    
    try:
        if (confidences[NOSE_IDX] < MIN_CONFIDENCE or 
            confidences[LEFT_SHOULDER_IDX] < MIN_CONFIDENCE or 
            confidences[RIGHT_SHOULDER_IDX] < MIN_CONFIDENCE):
            return None
            
        nose = np.array(keypoints[NOSE_IDX])
        left_shoulder = np.array(keypoints[LEFT_SHOULDER_IDX])  
        right_shoulder = np.array(keypoints[RIGHT_SHOULDER_IDX])
        
        shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
        shoulder_to_nose = nose - shoulder_midpoint
        neck_mount_point = shoulder_midpoint + (shoulder_to_nose * neck_to_head_ratio)
        
        return tuple(neck_mount_point.astype(int))
        
    except (IndexError, TypeError):
        return None

class PointStabilizer:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.stable_point = None
        self.last_stable_point = None
        
    def update(self, new_raw_point):
        if new_raw_point is None:
            return self.last_stable_point
            
        if self.stable_point is None:
            self.stable_point = np.array(new_raw_point, dtype=float)
        else:
            new_point = np.array(new_raw_point, dtype=float)
            self.stable_point = self.alpha * new_point + (1 - self.alpha) * self.stable_point
            
        self.last_stable_point = self.get_point_as_int_tuple()
        return self.last_stable_point
    
    def get_point_as_int_tuple(self):
        if self.stable_point is None:
            return None
        return tuple(self.stable_point.astype(int))

class PoseTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Pose Tracker")
        self.root.geometry("1400x900")
        
        # Настройка современной темы
        self.setup_modern_theme()
        
        # Переменные состояния
        self.is_running = False
        self.current_frame = None
        self.cap = None
        self.model = None
        self.neck_stabilizer = PointStabilizer(alpha=0.6)
        self.osc_client = SimpleUDPClient("127.0.0.1", 9001)
        self.camera_list = []
        self.camera_backends = []
        self.windows_camera_names = {}
        
        # Очереди для async обработки
        self.frame_queue = queue.Queue(maxsize=2)  # Ограничиваем размер очереди
        self.ui_update_queue = queue.Queue()
        
        # Получаем имена камер Windows
        if platform.system() == "Windows":
            self.windows_camera_names = get_windows_camera_names()
            print("Windows camera names:", self.windows_camera_names)
        
        # Инициализация модели
        self.init_model()
        
        # Создание GUI
        self.create_widgets()
        
        # Поток для видео
        self.video_thread = None
        self.stop_thread = False
        
        # Запуск обработчика UI обновлений
        self.process_ui_updates()
        
    def setup_modern_theme(self):
        """Настройка современной темы в градациях серого"""
        style = ttk.Style()
        
        # Основные цвета
        self.colors = {
            'bg_main': '#1e1e1e',      # Основной фон - почти черный
            'bg_panel': '#2d2d2d',     # Панели - темно-серый
            'bg_widget': '#3c3c3c',    # Виджеты - серый
            'bg_input': '#404040',     # Поля ввода - светло-серый
            'fg_primary': '#ffffff',   # Основной текст - белый
            'fg_secondary': '#cccccc', # Второстепенный текст - светло-серый
            'fg_disabled': '#808080',  # Отключенный текст - средне-серый
            'accent': '#0078d4',       # Акцент - синий
            'accent_hover': '#106ebe', # Акцент при наведении
            'success': '#16c60c',      # Успех - зеленый
            'error': '#e74856'         # Ошибка - красный
        }
        
        # Настройка фона окна
        self.root.configure(bg=self.colors['bg_main'])
        
        # Создание современной темы
        style.theme_create('modern_flat', parent='clam', settings={
            'TLabel': {
                'configure': {
                    'background': self.colors['bg_panel'],
                    'foreground': self.colors['fg_primary'],
                    'font': ('Segoe UI', 9)
                }
            },
            'TButton': {
                'configure': {
                    'background': self.colors['bg_widget'],
                    'foreground': self.colors['fg_primary'],
                    'borderwidth': 0,
                    'focuscolor': 'none',
                    'font': ('Segoe UI', 9, 'bold'),
                    'padding': [20, 10]
                },
                'map': {
                    'background': [
                        ('active', self.colors['accent']),
                        ('pressed', self.colors['accent_hover'])
                    ],
                    'foreground': [
                        ('active', self.colors['fg_primary']),
                        ('pressed', self.colors['fg_primary'])
                    ]
                }
            },
            'TFrame': {
                'configure': {
                    'background': self.colors['bg_panel'],
                    'borderwidth': 0
                }
            },
            'TLabelFrame': {
                'configure': {
                    'background': self.colors['bg_panel'],
                    'foreground': self.colors['fg_secondary'],
                    'borderwidth': 1,
                    'relief': 'solid',
                    'bordercolor': self.colors['bg_widget'],
                    'font': ('Segoe UI', 10, 'bold')
                }
            },
            'TLabelFrame.Label': {
                'configure': {
                    'background': self.colors['bg_panel'],
                    'foreground': self.colors['fg_secondary'],
                    'font': ('Segoe UI', 10, 'bold')
                }
            },
            'TCombobox': {
                'configure': {
                    'fieldbackground': self.colors['bg_input'],
                    'background': self.colors['bg_input'],
                    'foreground': self.colors['fg_primary'],
                    'borderwidth': 0,
                    'selectbackground': self.colors['accent'],
                    'arrowcolor': self.colors['fg_secondary']
                }
            },
            'TEntry': {
                'configure': {
                    'fieldbackground': self.colors['bg_input'],
                    'foreground': self.colors['fg_primary'],
                    'borderwidth': 0,
                    'insertcolor': self.colors['fg_primary']
                }
            }
        })
        
        style.theme_use('modern_flat')
        
    def get_camera_name(self, index):
        """Получение названия камеры с улучшенным определением"""
        backends_to_try = []
        
        # Определяем backend'ы для тестирования в зависимости от ОС
        if platform.system() == "Windows":
            backends_to_try = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_ANY, "Default")
            ]
        else:
            backends_to_try = [
                (cv2.CAP_V4L2, "V4L2"),
                (cv2.CAP_ANY, "Default")
            ]
        
        best_info = None
        working_backend = None
        
        for backend_id, backend_name in backends_to_try:
            try:
                cap = cv2.VideoCapture(index, backend_id)
                if cap.isOpened():
                    # Получаем информацию о камере
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    # Пытаемся получить кадр для проверки работоспособности
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret and frame is not None:
                        # Получаем имя камеры из Windows если доступно
                        device_name = self.windows_camera_names.get(index, f"Camera {index}")
                        
                        # Камера работает и дает кадры
                        if width > 0 and height > 0:
                            camera_info = f"{device_name} - {backend_name} ({width}x{height}@{fps}fps)"
                        else:
                            camera_info = f"{device_name} - {backend_name} (Working)"
                        
                        best_info = camera_info
                        working_backend = backend_id
                        
                        # Если это DirectShow или Media Foundation на Windows - приоритет
                        if backend_id in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
                            break
                    else:
                        cap.release()
                else:
                    cap.release()
                    
            except Exception as e:
                try:
                    cap.release()
                except:
                    pass
                continue
        
        return best_info, working_backend
    
    def get_available_cameras(self):
        """Поиск доступных камер с улучшенным определением"""
        cameras = []
        self.camera_list = []
        self.camera_backends = []  # Сохраняем backend'ы для каждой камеры
        
        print("Поиск камер...")
        for i in range(15):  # Увеличил до 15 для поиска больше камер
            result = self.get_camera_name(i)
            if result and result[0] and result[1] is not None:
                camera_name, backend = result
                cameras.append(camera_name)
                self.camera_list.append(i)
                self.camera_backends.append(backend)
                print(f"Найдена: {camera_name}")
        
        if not cameras:
            cameras = ["No cameras found"]
            self.camera_list = [0]
            self.camera_backends = [cv2.CAP_ANY]
            
        return cameras
    
    def test_selected_camera(self):
        """Тестирование выбранной камеры"""
        selected_camera = self.camera_combo.current()
        if selected_camera >= 0 and selected_camera < len(self.camera_list):
            camera_index = self.camera_list[selected_camera]
            backend = self.camera_backends[selected_camera]
            
            self.log_osc(f"Testing camera {camera_index}...")
            self.test_single_camera(camera_index, backend)
    
    def test_single_camera(self, camera_index, backend):
        """Тестирование одной камеры"""
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Показываем информацию о камере
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    device_name = self.windows_camera_names.get(camera_index, f"Camera {camera_index}")
                    self.log_osc(f"✓ {device_name}: {width}x{height}@{fps}fps")
                    
                    # Показываем тестовый кадр на 3 секунды
                    cv2.putText(frame, f'{device_name} - INDEX {camera_index}', (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f'{width}x{height} @ {fps}fps', (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    self.ui_update_queue.put(('test_frame', frame))
                    
                else:
                    self.log_osc(f"✗ Camera {camera_index} opened but no frames")
                cap.release()
            else:
                self.log_osc(f"✗ Cannot open camera {camera_index}")
        except Exception as e:
            self.log_osc(f"✗ Camera {camera_index} test error: {e}")
    
    def test_all_cameras(self):
        """Тестирование всех камер подряд"""
        self.log_osc("=== TESTING ALL CAMERAS ===")
        
        def test_cameras_thread():
            for i in range(15):  # Тестируем до 15 камер
                # Пробуем разные backend'ы
                backends_to_try = [
                    (cv2.CAP_DSHOW, "DirectShow"),
                    (cv2.CAP_MSMF, "Media Foundation")
                ]
                
                camera_found = False
                for backend_id, backend_name in backends_to_try:
                    try:
                        cap = cv2.VideoCapture(i, backend_id)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                # Получаем информацию
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                fps = int(cap.get(cv2.CAP_PROP_FPS))
                                
                                device_name = self.windows_camera_names.get(i, f"Camera {i}")
                                
                                # Добавляем информацию на кадр
                                cv2.putText(frame, f'CAMERA INDEX: {i}', (50, 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                                cv2.putText(frame, f'{device_name}', (50, 100), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(frame, f'{backend_name} - {width}x{height}@{fps}fps', (50, 150), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                                cv2.putText(frame, 'Press SPACE to continue', (50, height-50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                self.ui_update_queue.put(('log', f"INDEX {i}: {device_name} - {backend_name} ({width}x{height}@{fps}fps)"))
                                self.ui_update_queue.put(('test_frame', frame))
                                
                                camera_found = True
                                break
                            cap.release()
                        else:
                            cap.release()
                    except Exception:
                        try:
                            cap.release()
                        except:
                            pass
                
                if camera_found:
                    time.sleep(2)  # Показываем каждую камеру 2 секунды
        
        # Запускаем в отдельном потоке
        test_thread = threading.Thread(target=test_cameras_thread, daemon=True)
        test_thread.start()
    
    def show_test_frame(self, frame):
        """Показать тестовый кадр"""
        # Отображаем через очередь
        self.ui_update_queue.put(('test_frame', frame))
        
        # Через 3 секунды убираем
        self.root.after(3000, self.clear_test_frame)
    
    def clear_test_frame(self):
        """Очистить тестовый кадр"""
        if not self.is_running:
            self.ui_update_queue.put(('clear_test', None))
    
    def refresh_cameras(self):
        """Обновить список камер"""
        self.log_osc("Refreshing camera list...")
        
        # Обновляем имена Windows камер
        if platform.system() == "Windows":
            self.windows_camera_names = get_windows_camera_names()
            print("Updated Windows camera names:", self.windows_camera_names)
        
        cameras = self.get_available_cameras()
        self.camera_combo['values'] = cameras
        if cameras and cameras[0] != "No cameras found":
            self.camera_combo.set(cameras[0])
        self.log_osc(f"Found {len([c for c in cameras if c != 'No cameras found'])} cameras")
    
    def init_model(self):
        """Инициализация YOLO модели"""
        if not torch.cuda.is_available():
            print("CUDA не доступна!")
            return
            
        self.device = 'cuda'
        self.model = YOLO('yolov8m-pose.pt')
        self.model.to(self.device)
        
    def create_widgets(self):
        """Создание GUI элементов"""
        
        # Основной контейнер
        main_container = tk.Frame(self.root, bg=self.colors['bg_main'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Левая панель управления
        left_panel = tk.Frame(main_container, bg=self.colors['bg_panel'], width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 1))
        left_panel.pack_propagate(False)
        
        # Правая область для видео и логов
        right_area = tk.Frame(main_container, bg=self.colors['bg_main'])
        right_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # === ЛЕВАЯ ПАНЕЛЬ ===
        
        # Заголовок
        title_label = tk.Label(left_panel, text="POSE TRACKER", 
                              bg=self.colors['bg_panel'], fg=self.colors['fg_primary'],
                              font=('Segoe UI', 16, 'bold'))
        title_label.pack(pady=(20, 30))
        
        # Блок выбора камеры
        camera_frame = tk.Frame(left_panel, bg=self.colors['bg_panel'])
        camera_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(camera_frame, text="Camera", bg=self.colors['bg_panel'], 
                fg=self.colors['fg_secondary'], font=('Segoe UI', 9)).pack(anchor=tk.W)
        
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(camera_frame, textvariable=self.camera_var, 
                                        values=self.get_available_cameras(), state="readonly",
                                        font=('Segoe UI', 9))
        self.camera_combo.pack(fill=tk.X, pady=(5, 0))
        if self.camera_combo['values']:
            self.camera_combo.set(self.camera_combo['values'][0])
        
        # Кнопки для камеры
        camera_buttons_frame = tk.Frame(camera_frame, bg=self.colors['bg_panel'])
        camera_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        test_camera_btn = ttk.Button(camera_buttons_frame, text="Test", 
                                    command=self.test_selected_camera)
        test_camera_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 1))
        
        refresh_btn = ttk.Button(camera_buttons_frame, text="Refresh", 
                                command=self.refresh_cameras)
        refresh_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(1, 0))
        
        # Кнопка Test All Cameras
        test_all_btn = ttk.Button(camera_frame, text="Test All Cameras", 
                                 command=self.test_all_cameras)
        test_all_btn.pack(fill=tk.X, pady=(5, 0))
        
        # Кнопки управления
        controls_frame = tk.Frame(left_panel, bg=self.colors['bg_panel'])
        controls_frame.pack(fill=tk.X, padx=20, pady=(0, 30))
        
        self.start_btn = ttk.Button(controls_frame, text="Start Tracking", 
                                   command=self.start_tracking)
        self.start_btn.pack(fill=tk.X, pady=(0, 10))
        
        self.stop_btn = ttk.Button(controls_frame, text="Stop Tracking", 
                                  command=self.stop_tracking, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)
        
        # Статистика
        stats_frame = tk.Frame(left_panel, bg=self.colors['bg_widget'], relief=tk.FLAT)
        stats_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(stats_frame, text="Statistics", bg=self.colors['bg_widget'], 
                fg=self.colors['fg_secondary'], font=('Segoe UI', 10, 'bold')).pack(pady=(15, 10))
        
        self.fps_label = tk.Label(stats_frame, text="FPS: 0", bg=self.colors['bg_widget'], 
                                 fg=self.colors['fg_primary'], font=('Segoe UI', 9))
        self.fps_label.pack(pady=2)
        
        self.status_label = tk.Label(stats_frame, text="Status: Stopped", bg=self.colors['bg_widget'], 
                                    fg=self.colors['fg_primary'], font=('Segoe UI', 9))
        self.status_label.pack(pady=2)
        
        self.track_label = tk.Label(stats_frame, text="Tracking: No", bg=self.colors['bg_widget'], 
                                   fg=self.colors['fg_primary'], font=('Segoe UI', 9))
        self.track_label.pack(pady=(2, 15))
        
        # OSC настройки
        osc_frame = tk.Frame(left_panel, bg=self.colors['bg_widget'], relief=tk.FLAT)
        osc_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(osc_frame, text="OSC Settings", bg=self.colors['bg_widget'], 
                fg=self.colors['fg_secondary'], font=('Segoe UI', 10, 'bold')).pack(pady=(15, 10))
        
        tk.Label(osc_frame, text="IP Address", bg=self.colors['bg_widget'], 
                fg=self.colors['fg_secondary'], font=('Segoe UI', 9)).pack(anchor=tk.W, padx=15)
        
        self.osc_ip_var = tk.StringVar(value="127.0.0.1")
        ip_entry = ttk.Entry(osc_frame, textvariable=self.osc_ip_var, font=('Segoe UI', 9))
        ip_entry.pack(fill=tk.X, padx=15, pady=(2, 10))
        
        tk.Label(osc_frame, text="Port", bg=self.colors['bg_widget'], 
                fg=self.colors['fg_secondary'], font=('Segoe UI', 9)).pack(anchor=tk.W, padx=15)
        
        self.osc_port_var = tk.IntVar(value=9001)
        port_entry = ttk.Entry(osc_frame, textvariable=self.osc_port_var, font=('Segoe UI', 9))
        port_entry.pack(fill=tk.X, padx=15, pady=(2, 10))
        
        apply_btn = ttk.Button(osc_frame, text="Apply Settings", command=self.update_osc_client)
        apply_btn.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        # === ПРАВАЯ ОБЛАСТЬ ===
        
        # Видео
        video_container = tk.Frame(right_area, bg=self.colors['bg_panel'], relief=tk.FLAT, bd=1)
        video_container.pack(fill=tk.BOTH, expand=True, padx=(0, 0), pady=(0, 1))
        
        video_header = tk.Frame(video_container, bg=self.colors['bg_panel'], height=40)
        video_header.pack(fill=tk.X)
        video_header.pack_propagate(False)
        
        tk.Label(video_header, text="Video Feed", bg=self.colors['bg_panel'], 
                fg=self.colors['fg_secondary'], font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=15, pady=10)
        
        self.video_canvas = tk.Canvas(video_container, bg='#000000', highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.video_text_id = self.video_canvas.create_text(
            400, 300, text="No video feed", 
            fill=self.colors['fg_disabled'], font=('Segoe UI', 14)
        )
        
        # Логи
        log_container = tk.Frame(right_area, bg=self.colors['bg_panel'], relief=tk.FLAT, bd=1, height=250)
        log_container.pack(fill=tk.X, padx=(0, 0), pady=(0, 0))
        log_container.pack_propagate(False)
        
        log_header = tk.Frame(log_container, bg=self.colors['bg_panel'], height=40)
        log_header.pack(fill=tk.X)
        log_header.pack_propagate(False)
        
        tk.Label(log_header, text="OSC Logs", bg=self.colors['bg_panel'], 
                fg=self.colors['fg_secondary'], font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=15, pady=10)
        
        log_content = tk.Frame(log_container, bg=self.colors['bg_panel'])
        log_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 15))
        
        self.log_text = tk.Text(log_content, height=10, state=tk.DISABLED,
                               bg=self.colors['bg_input'], fg=self.colors['fg_primary'],
                               insertbackground=self.colors['fg_primary'],
                               selectbackground=self.colors['accent'],
                               selectforeground=self.colors['fg_primary'],
                               font=('Consolas', 9), relief=tk.FLAT, bd=0)
        
        log_scrollbar = ttk.Scrollbar(log_content, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def update_osc_client(self):
        """Обновление OSC клиента"""
        try:
            ip = self.osc_ip_var.get()
            port = self.osc_port_var.get()
            self.osc_client = SimpleUDPClient(ip, port)
            self.log_osc(f"OSC client updated: {ip}:{port}")
        except Exception as e:
            self.log_osc(f"OSC error: {e}")
    
    def log_osc(self, message):
        """Добавление сообщения в лог OSC"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def start_tracking(self):
        """Запуск трекинга"""
        if self.is_running:
            return
            
        selected_camera = self.camera_combo.current()
        if selected_camera >= 0 and selected_camera < len(self.camera_list):
            camera_index = self.camera_list[selected_camera]
            backend = self.camera_backends[selected_camera]
        else:
            camera_index = 0
            backend = cv2.CAP_ANY
        
        try:
            self.cap = cv2.VideoCapture(camera_index, backend)
            if not self.cap.isOpened():
                self.log_osc(f"Failed to open camera {camera_index}")
                return
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            self.is_running = True
            self.stop_thread = False
            
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Running", fg=self.colors['success'])
            
            self.video_canvas.delete(self.video_text_id)
            
            # Очищаем очереди
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
            camera_name = self.camera_combo.get()
            self.log_osc(f"Tracking started: {camera_name}")
            
        except Exception as e:
            self.log_osc(f"Start error: {e}")
    
    def stop_tracking(self):
        """Остановка трекинга"""
        self.is_running = False
        self.stop_thread = True
        
        if self.cap:
            self.cap.release()
            
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped", fg=self.colors['fg_primary'])
        self.track_label.config(text="Tracking: No")
        self.fps_label.config(text="FPS: 0")
        
        self.video_canvas.delete("all")
        self.video_text_id = self.video_canvas.create_text(
            400, 300, text="Video stopped", 
            fill=self.colors['fg_disabled'], font=('Segoe UI', 14)
        )
        
        self.log_osc("Tracking stopped")
    
    def video_loop(self):
        """Основной цикл обработки видео"""
        prev_time = time.time()
        
        while self.is_running and not self.stop_thread:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            results = self.model(frame, device='cuda')
            annotated_frame = results[0].plot()
            
            keypoints_data = None
            confidences_data = None
            if results[0].keypoints is not None and len(results[0].keypoints) > 0:
                keypoints_data = results[0].keypoints.xy[0].cpu().numpy()
                confidences_data = results[0].keypoints.conf[0].cpu().numpy()
            
            raw_neck_mount = None
            if keypoints_data is not None and confidences_data is not None:
                raw_neck_mount = get_neck_mount_point(keypoints_data, confidences_data, neck_to_head_ratio=0.3)
            
            stable_neck_mount = self.neck_stabilizer.update(raw_neck_mount)
            
            if stable_neck_mount is not None:
                normalized_x = stable_neck_mount[0] / frame_width
                normalized_y = stable_neck_mount[1] / frame_height
                
                try:
                    self.osc_client.send_message("/yolo/neck_anchor", [normalized_x, normalized_y])
                    if int(time.time() * 10) % 30 == 0:
                        self.ui_update_queue.put(('log', f"OSC: ({normalized_x:.3f}, {normalized_y:.3f})"))
                except Exception as e:
                    self.ui_update_queue.put(('log', f"OSC send error: {e}"))
                
                cv2.circle(annotated_frame, stable_neck_mount, 8, (0, 0, 255), -1)
                
                if raw_neck_mount is not None and keypoints_data is not None:
                    nose = tuple(keypoints_data[0].astype(int))
                    left_shoulder = tuple(keypoints_data[5].astype(int))
                    right_shoulder = tuple(keypoints_data[6].astype(int))
                    
                    if confidences_data[0] > 0.5:
                        cv2.line(annotated_frame, stable_neck_mount, nose, (0, 0, 255), 2)
                    if confidences_data[5] > 0.5:
                        cv2.line(annotated_frame, stable_neck_mount, left_shoulder, (0, 0, 255), 2)
                    if confidences_data[6] > 0.5:
                        cv2.line(annotated_frame, stable_neck_mount, right_shoulder, (0, 0, 255), 2)
                
                self.ui_update_queue.put(('track_status', True))
            else:
                self.ui_update_queue.put(('track_status', False))
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            self.ui_update_queue.put(('fps', fps))
            
            # Помещаем кадр в очередь (неблокирующе)
            try:
                self.frame_queue.put_nowait(annotated_frame)
            except queue.Full:
                # Если очередь полная, пропускаем кадр
                try:
                    self.frame_queue.get_nowait()  # Убираем старый кадр
                    self.frame_queue.put_nowait(annotated_frame)  # Добавляем новый
                except queue.Empty:
                    pass
            
            time.sleep(0.01)
    
    def process_ui_updates(self):
        """Обработка обновлений UI из очередей"""
        # Обработка кадров
        try:
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
                self.display_frame(frame)
                break  # Обрабатываем только последний кадр
        except queue.Empty:
            pass
        
        # Обработка UI обновлений
        try:
            while not self.ui_update_queue.empty():
                update_type, data = self.ui_update_queue.get_nowait()
                
                if update_type == 'fps':
                    self.fps_label.config(text=f"FPS: {data:.1f}")
                elif update_type == 'track_status':
                    if data:
                        self.track_label.config(text="Tracking: Yes", fg=self.colors['success'])
                    else:
                        self.track_label.config(text="Tracking: No", fg=self.colors['fg_primary'])
                elif update_type == 'log':
                    self.log_osc(data)
                elif update_type == 'test_frame':
                    self.display_frame(data)
                elif update_type == 'clear_test':
                    self.video_canvas.delete("all")
                    self.video_text_id = self.video_canvas.create_text(
                        400, 300, text="No video feed", 
                        fill=self.colors['fg_disabled'], font=('Segoe UI', 14)
                    )
        except queue.Empty:
            pass
        
        # Планируем следующую обработку
        self.root.after(16, self.process_ui_updates)  # ~60 FPS UI updates
    
    def display_frame(self, frame):
        """Отображение кадра в GUI"""
        try:
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            frame_height, frame_width = frame.shape[:2]
            aspect_ratio = frame_width / frame_height
            
            if canvas_width / canvas_height > aspect_ratio:
                display_height = canvas_height - 20
                display_width = int(display_height * aspect_ratio)
            else:
                display_width = canvas_width - 20
                display_height = int(display_width / aspect_ratio)
            
            display_frame = cv2.resize(frame, (display_width, display_height))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            self.update_video_display(photo, display_width, display_height)
        except Exception as e:
            print(f"Display frame error: {e}")
    
    def update_video_display(self, photo, width, height):
        """Обновление видео дисплея"""
        try:
            self.video_canvas.delete("all")
            
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            x = (canvas_width - width) // 2
            y = (canvas_height - height) // 2
            
            self.video_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.video_canvas.photo = photo
        except Exception as e:
            print(f"Update video display error: {e}")
    
    def on_closing(self):
        """Обработка закрытия окна"""
        self.stop_tracking()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = PoseTrackerGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 