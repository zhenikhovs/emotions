import os

from src.extract_frames import extract_frames
from src.extract_faces import extract_faces_mediapipe
from src.create_sequences import create_sequences
from src.train import train_model
from src.predict import predict_on_video

MODE = "train"  # prepare | faces | sequences | train | predict

DATA_DIR = "data"
FRAMES_DIR = "frames"
FACES_DIR = "faces_frames"
SEQUENCES_PATH = "sequences/data_sequences.npz"
MODEL_PATH = "models/cnn_lstm_model.h5"
WEIGHTS_PATH = "models/efficientnetb1_notop.h5"  # веса EfficientNetB1

TEST_ACTORS = ["Actor_23", "Actor_24"]

if MODE == "prepare":
    print("STEP 1: Extracting frames from videos...")
    extract_frames(
        input_dir=DATA_DIR,
        output_dir=FRAMES_DIR,
        fps_target=5,
        test_actors=TEST_ACTORS
    )

elif MODE == "faces":
    print("STEP 2: Extracting faces...")
    extract_faces_mediapipe(
        input_dir=FRAMES_DIR,
        output_dir=FACES_DIR,
        img_size=96
    )

elif MODE == "sequences":
    print("STEP 3: Creating sequences...")
    create_sequences(
        frames_dir=FACES_DIR,
        seq_len=22,
        out_path=SEQUENCES_PATH,
        img_size=96
    )

elif MODE == "train":
    print("STEP 4: Training CNN + LSTM model...")
    train_model(
        sequences_path=SEQUENCES_PATH,
        model_path=MODEL_PATH,
        weights_path=WEIGHTS_PATH
    )

elif MODE == "predict":
    print("STEP 5: Predicting emotion...")
    for actor in TEST_ACTORS:
        predict_on_video(
            model_path=MODEL_PATH,
            video_dir=os.path.join(DATA_DIR, actor)
        )

else:
    raise ValueError("Unknown MODE")
