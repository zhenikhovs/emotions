import numpy as np
import cv2
import matplotlib.pyplot as plt

# Путь к файлу .npz
npz_path = "sequences/data_sequences.npz"

# Загружаем данные
data = np.load(npz_path, allow_pickle=True)
X = data['X']  # изображения
y = data['y']  # метки
classes = data['classes']  # названия эмоций

print("Файл содержит массивы:", data.files)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("classes:", classes)

# Покажем несколько кадров из первых 5 последовательностей
for i in range(5):
    seq = X[i]  # последовательность кадров
    label = classes[y[i]]  # эмоция

    fig, axes = plt.subplots(1, seq.shape[0], figsize=(20, 3))
    fig.suptitle(f"Sequence {i} | Emotion: {label}", fontsize=16)

    for j, ax in enumerate(axes):
        ax.imshow(seq[j])
        ax.axis('off')

    plt.show()
