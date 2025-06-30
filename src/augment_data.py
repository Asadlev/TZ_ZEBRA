import os
import shutil

import cv2
import albumentations as A


def augment_data(input_dir, output_dir):
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    ])

    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for img_name in image_files:
        # Обработка изображения
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Аугментация
        augmented = augmentation(image=img)
        cv2.imwrite(os.path.join(output_dir, f"aug_{img_name}"), augmented["image"])

        # Копируем соответствующую аннотацию (если существует)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(input_dir, txt_name)
        if os.path.exists(txt_path):
            shutil.copy(txt_path, os.path.join(output_dir, f"aug_{txt_name}"))

    print(f"Аугментация завершена. Данные сохранены в {output_dir}")

