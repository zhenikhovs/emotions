import os
import cv2
import mediapipe as mp
from tqdm import tqdm


def extract_faces_mediapipe(input_dir, output_dir, img_size=96):
    """
    Извлекает лица из кадров с помощью MediaPipe.

    input_dir/
      emotion/
        video_folder/
          0.jpg

    output_dir/
      emotion/
        video_folder/
          0.jpg (96x96 лицо)
    """

    os.makedirs(output_dir, exist_ok=True)

    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.6
    )

    emotions = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    for emotion in emotions:
        emotion_path = os.path.join(input_dir, emotion)

        video_folders = [
            d for d in os.listdir(emotion_path)
            if os.path.isdir(os.path.join(emotion_path, d))
        ]

        for video_folder in tqdm(
            video_folders,
            desc=f"MediaPipe | {emotion}",
            unit="video"
        ):
            video_path = os.path.join(emotion_path, video_folder)
            out_folder = os.path.join(output_dir, emotion, video_folder)
            os.makedirs(out_folder, exist_ok=True)

            frames = sorted(
                os.listdir(video_path),
                key=lambda x: int(x.split(".")[0])
            )

            for f in frames:
                img_path = os.path.join(video_path, f)
                img = cv2.imread(img_path)

                if img is None:
                    continue

                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb)

                if results.detections:
                    det = results.detections[0]
                    box = det.location_data.relative_bounding_box

                    x1 = int(box.xmin * w)
                    y1 = int(box.ymin * h)
                    bw = int(box.width * w)
                    bh = int(box.height * h)

                    cx = x1 + bw // 2
                    cy = y1 + bh // 2
                    size = max(bw, bh)

                    x1 = max(cx - size // 2, 0)
                    y1 = max(cy - size // 2, 0)
                    x2 = min(x1 + size, w)
                    y2 = min(y1 + size, h)

                    face = img[y1:y2, x1:x2]
                else:
                    # fallback — центр кадра
                    min_dim = min(h, w)
                    cx, cy = w // 2, h // 2
                    x1 = cx - min_dim // 2
                    y1 = cy - min_dim // 2
                    face = img[y1:y1+min_dim, x1:x1+min_dim]

                face = cv2.resize(face, (img_size, img_size))
                cv2.imwrite(os.path.join(out_folder, f), face)

    face_detector.close()
