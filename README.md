# Ternary Neural Network on MNIST

This project presents a neural network implementation trained on the MNIST dataset, with both **inputs and weights quantized to ternary values** (-1, 0, +1). The model is designed and tested using Google Colab and TensorFlow, with an emphasis on reducing computational complexity while maintaining a competitive classification accuracy.

---

## 📌 Motivation

In resource-constrained environments such as edge devices, minimizing power consumption and memory usage is critical. Ternary Neural Networks (TNNs) provide a lightweight alternative to full-precision models by using a discrete representation for weights and inputs. This project is part of a broader exploration into efficient AI for embedded systems, particularly targeting platforms like ARM and ESP32.

---

## 🧠 What is a Ternary Neural Network?

A Ternary Neural Network is a type of neural network where weights and/or activations are restricted to three possible values:

- **+1** (positive influence)
- **0** (no influence)
- **–1** (negative influence)

This drastically simplifies multiplication operations to additions and subtractions, and removes unnecessary computation when a value is zero.

---

## 🔍 Project Highlights

- ✅ Uses the MNIST dataset for handwritten digit recognition.
- ✅ Converts both input images and model weights to ternary format.
- ✅ Implements custom ternarization functions.
- ✅ Trains and evaluates the model in Google Colab.
- ✅ Offers analysis of performance vs. precision trade-off.
- ✅ Optimized for low-resource hardware inference.

---

## 🗂️ Dataset Information

- **Name:** MNIST
- **Size:** 60,000 training images, 10,000 test images
- **Content:** 28x28 grayscale images of handwritten digits (0–9)
- **Source:** [Yann LeCun's MNIST Database](http://yann.lecun.com/exdb/mnist/)

The dataset is automatically downloaded using TensorFlow’s dataset loader.

---
