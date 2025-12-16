import os

from src.extract_frames import extract_frames
from src.extract_faces import extract_faces_mediapipe
from src.create_sequences import create_sequences
from src.train import train_model
from src.predict import predict_and_evaluate


MODE = "predict"  # prepare | faces | sequences | train | predict

DATA_DIR = "data"
FRAMES_DIR = "frames"
FACES_DIR = "faces_frames"
SEQUENCES_PATH = "sequences/data_sequences_96.npz"
MODEL_PATH = "models/cnn_lstm_model_96_99_83p.keras"

TEST_ACTORS = ["Actor_02", "Actor_24"]

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
        img_size=96  # Используем полный размер 96x96 вместо уменьшения до 48x48
    )

elif MODE == "train":
    print("STEP 4: Training CNN + LSTM model...")
    train_model(
        sequences_path=SEQUENCES_PATH,
        model_path=MODEL_PATH,
    )

elif MODE == "predict":
    predict_and_evaluate(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        test_actors=TEST_ACTORS,
        sequences_path=SEQUENCES_PATH,
    )

else:
    raise ValueError("Unknown MODE")