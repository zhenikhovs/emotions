import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    LSTM,
    Dropout,
    BatchNormalization,
)


def build_model(seq_len, img_size, n_classes):
    """
    Оптимизированная CNN+LSTM модель для распознавания эмоций.
    Баланс между сложностью и возможностью обучиться на доступных данных.
    Умеренная регуляризация для борьбы с переобучением без потери точности.
    """
    input_layer = Input(shape=(seq_len, img_size, img_size, 3))

    # CNN блок - 3 слоя (проверенная конфигурация)
    def cnn_block():
        model_input = Input(shape=(img_size, img_size, 3))
        x = Conv2D(32, (3, 3), activation="relu", padding="same")(model_input)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        return Model(model_input, x)

    cnn = cnn_block()

    # TimeDistributed CNN
    td_cnn = TimeDistributed(cnn)(input_layer)

    # LSTM - умеренный размер с умеренной регуляризацией
    lstm_out = LSTM(96, return_sequences=False)(td_cnn)
    lstm_out = Dropout(0.4)(lstm_out)  # Чуть больше чем 0.3, но не так сильно как 0.5

    # Dense слой с умеренным Dropout
    dense = Dense(128, activation="relu")(lstm_out)
    dense = Dropout(0.5)(dense)  # Чуть больше чем 0.3, но не так сильно как 0.5

    output = Dense(n_classes, activation="softmax")(dense)

    model = Model(inputs=input_layer, outputs=output)
    return model
