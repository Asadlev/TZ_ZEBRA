import os
import shutil
from sklearn.model_selection import train_test_split
import glob


def split_data(input_dir, output_dir, ratios=(0.7, 0.15, 0.15)):
    os.makedirs(output_dir, exist_ok=True)

    annotations = glob.glob(f"{input_dir}/*.txt")

    if len(annotations) == 0:
        raise Exception("Нет аннотаций.")

    train, temp = train_test_split(annotations, test_size=1 - ratios[0], random_state=42)
    val, test = train_test_split(temp, test_size=ratios[2] / (ratios[1] + ratios[2]), random_state=42)

    splits = {'train': train, 'val': val, 'test': test}

    print(f"Обучающая выборка: {len(train)}, Валидационная выборка: {len(val)}, Тестовая выборка: {len(test)}")

    for split_name, split_files in splits.items():
        split_dir_images = os.path.join(output_dir, split_name, 'images')
        split_dir_labels = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(split_dir_images, exist_ok=True)
        os.makedirs(split_dir_labels, exist_ok=True)

        for file_path in split_files:
            image_name = os.path.splitext(os.path.basename(file_path))[0] + ".jpg"
            image_path = os.path.join(str(input_dir.replace('annotations', '')), str(image_name))

            # Проверка наличия изображения
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Изображение {image_path} для аннотации {file_path} не найдено.")

            # Копирование файла аннотации
            shutil.copy(file_path, os.path.join(split_dir_labels, os.path.basename(file_path)))
            # Копирование соответствующего изображения
            shutil.copy(image_path, os.path.join(split_dir_images, image_name))

    print(f"Данные разделены. Train: {len(train)}, Val: {len(val)}, Test: {len(test)}.")
