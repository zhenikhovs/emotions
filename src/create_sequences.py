import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def create_sequences(frames_dir, seq_len, out_path, img_size=96):
    """
    Создаёт последовательности кадров для обучения модели.
    Сохраняет метаданные: эмоцию, видео, актёра.
    """
    X = []
    y = []
    actors = []
    videos = []

    # Список эмоций
    emotions = sorted([
        d for d in os.listdir(frames_dir)
        if os.path.isdir(os.path.join(frames_dir, d))
    ])
    encoder = LabelEncoder()
    encoder.fit(emotions)

    for emotion in emotions:
        emotion_path = os.path.join(frames_dir, emotion)
        video_folders = [
            d for d in os.listdir(emotion_path)
            if os.path.isdir(os.path.join(emotion_path, d))
        ]

        for video_folder in tqdm(video_folders, desc=f"Processing {emotion}", unit="video"):
            video_path = os.path.join(emotion_path, video_folder)
            frames = sorted([
                f for f in os.listdir(video_path) if f.endswith(".jpg")
            ], key=lambda x: int(x.split(".")[0]))

            if len(frames) < seq_len:
                continue  # короткие видео пропускаем

            frames = frames[:seq_len]
            seq = []
            for f in frames:
                img_path = os.path.join(video_path, f)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # Resize на всякий случай, кадры уже 96x96
                img = cv2.resize(img, (img_size, img_size))
                # Нормализация
                img = img.astype(np.float32) / 255.0
                seq.append(img)

            if len(seq) != seq_len:
                continue  # если кадров меньше, чем seq_len, пропускаем

            X.append(seq)
            y.append(emotion)
            videos.append(video_folder)
            # Из имени видео_folder извлекаем актёра (actorXX)
            actor = video_folder.split("_")[0] if "_" in video_folder else "unknown"
            actors.append(actor)

    X = np.array(X, dtype=np.float32)
    y = encoder.transform(y)
    np.savez(
        out_path,
        X=X,
        y=y,
        classes=encoder.classes_,
        videos=np.array(videos),
        actors=np.array(actors)
    )
    print(f"Sequences saved: {out_path}, total sequences: {len(X)}")
