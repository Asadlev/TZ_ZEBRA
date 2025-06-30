from ultralytics import YOLO
import os


def train_yolo():
    config_path = os.path.join('configs', 'data.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError("Конфигурационный файл не найден.")

    model = YOLO("yolov8n.pt")

    train_params = {
        'data': config_path,
        'epochs': 30,
        'batch': 16,
        'imgsz': 640,
        'optimizer': 'Adam',
        'project': 'runs',
        'name': 'train',
        'exist_ok': True
    }

    results = model.train(**train_params)
    print("Обучение завершено.")
    return results



