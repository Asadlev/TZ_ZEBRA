import cv2
import os


def extract_frames(input_video, output_dir, frame_interval):
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Видео {input_video} не найдено.")

    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        raise Exception("Не удалось открыть видео.")

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    while video.isOpened():
        ret, frame = video.read()
        if not ret: break
        if count % 30 == 0:
            cv2.imwrite(f"{output_dir}/{count:04d}.jpg", frame)
        count += 1

    print(f"Кадры успешно извлечены в {output_dir}.")



