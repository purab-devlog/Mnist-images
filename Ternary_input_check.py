import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist


(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.astype('float32') / 255.0
x_test = np.expand_dims(x_test, axis=-1)


def ternarize_input(img):
    mean = np.mean(img)
    std = np.std(img)
    upper = mean + 0.25 * std
    lower = mean - 0.25 * std
    return np.where(img > upper, 1.0, np.where(img < lower, -1.0, 0.0))

for i in range(5):
    original = x_test[i].squeeze()
    ternary = ternarize_input(original)

    print(f"Image {i} - unique values in ternary images:", np.unique(ternary))

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('original')
    axes[0].axis('off')

    axes[1].imshow(ternary, cmap='bwr', vmin=-1, vmax=1)
    axes[1].set_title('Ternarized')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
