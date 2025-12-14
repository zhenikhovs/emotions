import numpy as np
import tensorflow as tf
from src.model_cnn_lstm import build_model
from tqdm.keras import TqdmCallback  # прогресс-бар для Keras


def train_model(sequences_path, model_path, weights_path="efficientnetb1_notop.h5"):
    """
    Обучение CNN+LSTM модели на последовательностях кадров.

    Args:
        sequences_path (str): путь к .npz с последовательностями
        model_path (str): путь для сохранения обученной модели
        weights_path (str): путь к весам EfficientNetB1 без верхнего слоя
    """
    # Загружаем последовательности
    data = np.load(sequences_path, allow_pickle=True)
    X, y = data["X"], data["y"]

    IMG_SIZE = X.shape[2]
    SEQ_LEN = X.shape[1]
    N_CLASSES = len(np.unique(y))

    # Строим модель
    model = build_model(SEQ_LEN, IMG_SIZE, N_CLASSES, weights_path=weights_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Тренируем с прогресс-баром
    model.fit(
        X, y,
        batch_size=8,
        epochs=20,
        validation_split=0.2,
        verbose=1,
        callbacks=[TqdmCallback(verbose=1)]
    )

    # Сохраняем модель
    model.save(model_path)
    print(f"Model saved to {model_path}")
