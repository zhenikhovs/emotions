import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    Dense,
    LSTM,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.applications import MobileNetV2


def build_model(seq_len, img_size, n_classes, freeze_backbone=True):
    """
    CNN+LSTM модель с MobileNetV2 (предобучена на ImageNet) для распознавания эмоций.
    Использует transfer learning для лучших результатов при меньшем количестве данных.
    """
    input_layer = Input(shape=(seq_len, img_size, img_size, 3))

    # MobileNetV2 предобучена на ImageNet - лёгкая и эффективная
    def cnn_block():
        # Загружаем MobileNetV2 без верхнего слоя (include_top=False)
        # Веса уже скачаны локально, поэтому загрузка должна пройти успешно
        base_model = MobileNetV2(
            input_shape=(img_size, img_size, 3),
            include_top=False,
            weights='imagenet',  # Используем веса, предобученные на ImageNet
            pooling=None,
        )
        print("✓ MobileNetV2 веса загружены с ImageNet")
        
        # Замораживаем часть слоёв для transfer learning
        if freeze_backbone:
            # Замораживаем все слои кроме последних 20 (можно дообучить)
            for layer in base_model.layers[:-20]:
                layer.trainable = False
        else:
            # Размораживаем все слои для fine-tuning
            for layer in base_model.layers:
                layer.trainable = True
        
        # Добавляем GlobalAveragePooling2D для получения фичевых векторов
        model_input = Input(shape=(img_size, img_size, 3))
        x = base_model(model_input)
        x = GlobalAveragePooling2D()(x)
        return Model(model_input, x)

    cnn = cnn_block()

    # TimeDistributed CNN - применяем к каждому кадру в последовательности
    td_cnn = TimeDistributed(cnn)(input_layer)

    # LSTM - умеренный размер с умеренной регуляризацией
    lstm_out = LSTM(96, return_sequences=False)(td_cnn)
    lstm_out = Dropout(0.35)(lstm_out)  # Чуть больше чем 0.3, но не так сильно как 0.5

    # Dense слой с умеренным Dropout
    dense = Dense(128, activation="relu")(lstm_out)
    dense = Dropout(0.5)(dense)  # Чуть больше чем 0.3, но не так сильно как 0.5

    output = Dense(n_classes, activation="softmax")(dense)

    model = Model(inputs=input_layer, outputs=output)
    return model
