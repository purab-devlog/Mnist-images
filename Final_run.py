import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load MNIST test data
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test[:5].astype('float32') / 255.0
x_test = np.expand_dims(x_test, -1)

#Function to ternarize input using mean ± 0.25 * std
def ternarize_input(img):
    mean = np.mean(img)
    std = np.std(img)
    upper = mean + 0.25 * std
    lower = mean - 0.25 * std
    return np.where(img > upper, 1.0, np.where(img < lower, -1.0, 0.0)).astype(np.float32)

ternary_inputs = np.array([ternarize_input(img) for img in x_test])

#Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="mnist_ternary.tflite")
interpreter.allocate_tensors()

#Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Function to ternarize weights using global threshold
def ternarize_model_weights(interpreter, threshold_ratio=0.05):
    for detail in interpreter.get_tensor_details():
        if 'kernel' in detail['name'] or 'weights' in detail['name']:
            tensor = interpreter.get_tensor(detail['index'])
            thresh = threshold_ratio * np.max(np.abs(tensor))
            ternary_tensor = np.where(tensor > thresh, 1.0, np.where(tensor < -thresh, -1.0, 0.0)).astype(np.float32)
            interpreter.set_tensor(detail['index'], ternary_tensor)

#Apply weight ternarization
ternarize_model_weights(interpreter)

#Run inference on 5 images and plot
for i in range(5):
    input_tensor = np.expand_dims(ternary_inputs[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output)

    #Plot original image with prediction
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title(f"Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
