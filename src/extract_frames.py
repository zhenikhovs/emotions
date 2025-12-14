import os
import cv2
from tqdm import tqdm

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

def extract_frames(input_dir, out_dir, fps_target=5, face_crop=False, test_actors=None):
    if test_actors is None:
        test_actors = []

    os.makedirs(out_dir, exist_ok=True)

    actors = sorted([a for a in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, a))])

    for actor in tqdm(actors, desc="Actors"):
        if actor in test_actors:
            continue  # пропускаем тестовых актёров

        actor_path = os.path.join(input_dir, actor)
        videos = [v for v in os.listdir(actor_path) if v.endswith(".mp4")]

        for video_name in tqdm(videos, desc=f"{actor}", leave=False):
            parts = video_name.replace(".mp4", "").split("-")
            if len(parts) < 7:
                print(f"Skipping unexpected filename format: {video_name}")
                continue

            emotion = EMOTION_MAP.get(parts[2], "unknown")
            intensity = parts[3]
            statement = parts[4]
            repetition = parts[5]
            actor_num = parts[6]

            out_folder = os.path.join(
                out_dir,
                emotion,
                f"{actor_num}_int{intensity}_stmt{statement}_rep{repetition}"
            )
            os.makedirs(out_folder, exist_ok=True)

            video_path = os.path.join(actor_path, video_name)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            step = max(int(fps / fps_target), 1)
            frame_id = 0
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if count % step == 0:
                    if face_crop:
                        # место для детекции лица, пока пропускаем
                        pass
                    frame_path = os.path.join(out_folder, f"{frame_id}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_id += 1
                count += 1

            cap.release()
