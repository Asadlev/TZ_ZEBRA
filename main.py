import os
from src.extract_frames import extract_frames
from src.augment_data import augment_data
from src.split_dates import split_data
from src.train_yolov11 import train_yolo
from src.generate_annotations import generate_annotation


def main():
    try:
        # Извлечение кадров
        print("Извлечение кадров...")
        extract_frames(
            input_video=os.path.join('data', 'raw_video', '1.MOV'),
            output_dir=os.path.join('data', 'frames'),
            frame_interval=30
        )

        # Аугментация данных
        print("Аугментация данных...")
        augment_data(
            input_dir=os.path.join('data', 'frames'),
            output_dir=os.path.join('data', 'augmented')
        )

        # Генерация аннотаций
        print("Генерация аннотаций...")
        generate_annotation()

        # Разделение данных на train/val/test
        print("Разделение данных...")
        split_data(
            input_dir=os.path.join('data', 'frames', 'annotations'),
            output_dir=os.path.join('data', 'dataset'),
            ratios=(0.7, 0.15, 0.15)
        )

        # Обучение модели
        print("Запуск обучения модели...")
        train_yolo()
        print("Все этапы выполнены успешно!")
    except Exception as e:
        print(f"Произошла ошибка: {e}")



if __name__ == "__main__":
    main()
