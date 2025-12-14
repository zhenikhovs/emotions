import cv2
import numpy as np
import tensorflow as tf


def predict_on_video(model_path, video_path):
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    IMG_SIZE = 96

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frames.append(frame)

    cap.release()

    if len(frames) < 20:
        print("Video too short (<20 frames)")
        return

    frames = np.array(frames[:20])
    frames = np.expand_dims(frames, axis=0)

    pred = model.predict(frames)
    emotion_id = np.argmax(pred)

    print("Predicted class index:", emotion_id)
