import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.model_cnn_lstm import build_model
from tqdm.keras import TqdmCallback  # прогресс-бар для Keras


def train_model(sequences_path, model_path, test_size: float = 0.2, random_state: int = 42):
    """
    Обучение модели на заранее подготовленных последовательностях.
    Делает стратифицированный train/val split, чтобы валидация
    содержала все классы и метрика считалась корректно.
    """
    data = np.load(sequences_path, allow_pickle=True)
    X, y = data["X"], data["y"]
    classes = data["classes"]

    # Стратифицированное разбиение, чтобы валидация не потеряла редкие классы
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )

    IMG_SIZE = X.shape[2]
    SEQ_LEN = X.shape[1]
    N_CLASSES = len(np.unique(y))
    
    # Вычисляем class_weight для балансировки классов
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {dict(zip(classes, class_weights))}")

    model = build_model(SEQ_LEN, IMG_SIZE, N_CLASSES)
    print("MODEL LOADED")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  # Вернул к более консервативному LR
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True,
        mode="max",
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode="min",
    )

    print("Start fit")
    model.fit(
        X_train,
        y_train,
        batch_size=8,
        epochs=60,
        validation_data=(X_val, y_val),
        verbose=0,
        callbacks=[TqdmCallback(verbose=1), early_stop, reduce_lr],
        shuffle=True,
        class_weight=class_weight_dict,  # Балансировка классов
    )

    # Сохраняем в современном формате .keras
    model.save(model_path)
    print(f"Model saved to {model_path}")