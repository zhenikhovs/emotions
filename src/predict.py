import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}


def _load_meta(sequences_path: str):
    data = np.load(sequences_path, allow_pickle=True)
    classes = data["classes"]
    seq_len = data["X"].shape[1]
    img_size = data["X"].shape[2]
    return classes, seq_len, img_size


def _emotion_from_filename(video_name: str):
    parts = video_name.replace(".mp4", "").split("-")
    if len(parts) < 3:
        return None
    return EMOTION_MAP.get(parts[2], None)


def _detect_and_crop_face(frame, detector, img_size):
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

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

        face = frame[y1:y2, x1:x2]
    else:
        min_dim = min(h, w)
        cx, cy = w // 2, h // 2
        x1 = cx - min_dim // 2
        y1 = cy - min_dim // 2
        face = frame[y1:y1 + min_dim, x1:x1 + min_dim]

    return cv2.resize(face, (img_size, img_size))


def _video_to_sequence(video_path, seq_len, img_size, fps_target=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    mp_face = mp.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.6)

    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(cap_fps / fps_target), 1)

    frames = []
    count = 0
    while len(frames) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            face = _detect_and_crop_face(frame, detector, img_size)
            frames.append(face.astype(np.float32) / 255.0)
        count += 1

    cap.release()
    detector.close()

    if len(frames) < seq_len:
        return None

    return np.array(frames, dtype=np.float32)



def predict_and_evaluate(
    model_path: str,
    data_dir: str,
    test_actors: list,
    sequences_path: str,
):
    classes, seq_len, img_size = _load_meta(sequences_path)
    class_to_id = {c: i for i, c in enumerate(classes)}

    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []

    print("\n=== PREDICTIONS ===\n")

    for actor in test_actors:
        actor_dir = os.path.join(data_dir, actor)
        videos = [v for v in os.listdir(actor_dir) if v.endswith(".mp4")]

        for video in videos:
            gt = _emotion_from_filename(video)
            if gt not in class_to_id:
                continue

            seq = _video_to_sequence(
                os.path.join(actor_dir, video),
                seq_len,
                img_size
            )
            if seq is None:
                continue

            seq = np.expand_dims(seq, axis=0)
            probs = model.predict(seq, verbose=0)
            pred_id = int(np.argmax(probs))
            pred_label = classes[pred_id]

            print(f"{video} → {pred_label}")

            y_true.append(class_to_id[gt])
            y_pred.append(pred_id)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\n=== METRICS (TEST ACTORS) ===\n")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=classes))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix — TEST ACTORS")
    plt.tight_layout()

    out_path = "models/confusion_matrix_test_actors.png"
    plt.savefig(out_path, dpi=300)
    print(f"Confusion matrix saved to {out_path}")

    plt.show()

