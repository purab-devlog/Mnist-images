import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Step 2: Create a simple CNN model
model = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=3, validation_split=0.1)

# Step 3: Apply ternary quantization using global thresholding
def ternarize_weights(model, threshold_ratio=0.05):
    all_weights = []
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()
            for w in weights:
                all_weights.extend(w.flatten())

    all_weights = np.array(all_weights)
    global_thresh = threshold_ratio * np.max(np.abs(all_weights))

    # Update model weights
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            weights = layer.get_weights()
            ternary_weights = []
            for w in weights:
                ternary_w = np.where(w > global_thresh, 1,
                            np.where(w < -global_thresh, -1, 0))
                ternary_weights.append(ternary_w.astype(np.float32))
            layer.set_weights(ternary_weights)

# Apply ternarization
ternarize_weights(model, threshold_ratio=0.05)

# Step 4: Convert the ternary model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Step 5: Save the TFLite model
with open("mnist_ternary.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Ternary TFLite model saved as mnist_ternary.tflite")
