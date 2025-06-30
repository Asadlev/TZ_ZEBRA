import os
import random


def generate_annotations(image_dir, annotations_dir, num_classes=5):
    """
    Генерирует фиктивные аннотации в формате YOLO.

    (Из за того что при запуске labelImg во время обучения модели, он не генерирует аннотацию, потому что версий labelImg с новой версий Python не совпадают)

    и вот, решил написать ручной скрипт

    :param images_dir: Директория с изображениями.
    :param annotations_dir: Директория для сохранения аннотаций.
    :param num_classes: Количество классов.
    """

    os.makedirs(annotations_dir, exist_ok=True)
    images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image in images:
        image_path = os.path.join(image_dir, image)
        annotations = os.path.join(annotations_dir, os.path.splitext(image)[0] + '.txt')

        with open(annotations, 'w') as f:
            for _ in range(random.randint(1, 5)):   # 1–5 объектов на изображение
                class_id = random.randint(0, num_classes - 1)
                x_center = random.uniform(0, 1)
                y_center = random.uniform(0, 1)
                width = random.uniform(0, 1)
                height = random.uniform(0, 1)
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        print(f"Аннотация сгенерирована для {image_path}")


def generate_annotation():
    images_dir = os.path.join('data', 'frames')
    annotations_dir = os.path.join('data', 'frames', 'annotations')
    generate_annotations(images_dir, annotations_dir)
