import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, GlobalAveragePooling2D


def build_model(seq_len, img_size, n_classes, weights_path="efficientnetb1_notop.h5"):
    """
    Строит CNN+LSTM модель для распознавания эмоций.

    Args:
        seq_len (int): длина последовательности кадров
        img_size (int): размер кадра (img_size x img_size x 3)
        n_classes (int): количество классов (эмоций)
        weights_path (str): путь к pre-trained весам EfficientNetB1 без верхнего слоя
    Returns:
        tf.keras.Model
    """
    # Проверка файла весов
    import os
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}")

    # Входной слой для последовательности
    input_layer = Input(shape=(seq_len, img_size, img_size, 3))

    # CNN для извлечения признаков из каждого кадра
    base_cnn = EfficientNetB1(
        include_top=False,
        weights=None,  # мы будем загружать свои веса вручную
        input_shape=(img_size, img_size, 3)
    )
    base_cnn.load_weights(weights_path)

    # Применяем CNN ко всем кадрам с TimeDistributed
    cnn_out = TimeDistributed(base_cnn)(input_layer)
    cnn_out = TimeDistributed(GlobalAveragePooling2D())(cnn_out)  # сжимаем в вектор признаков

    # LSTM для анализа последовательности
    lstm_out = LSTM(128, return_sequences=False)(cnn_out)

    # Dense слои для классификации
    dense = Dense(64, activation='relu')(lstm_out)
    output = Dense(n_classes, activation='softmax')(dense)

    model = Model(inputs=input_layer, outputs=output)
    return model
