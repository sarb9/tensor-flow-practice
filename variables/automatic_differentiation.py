import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2

# dy = 2x * dx
dy_dx = tape.gradient(y, x)
dy_dx.numpy()

w = tf.Variable(tf.random.normal((3, 2)), name="w")
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name="b")
x = [[1.0, 2.0, 3.0]]

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y ** 2)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])

# Gradiant tensors have the same shapes as their initial tensors
print(w.shape)
print(dl_dw.shape)


my_vars = {
    "w": tf.Variable(tf.random.normal((3, 2)), name="w"),
    "b": tf.Variable(tf.zeros(2, dtype=tf.float32), name="b"),
}

grad = tape.gradient(loss, my_vars)
print("expected None as each tape could only calculate one grad:", grad["b"])

layer = tf.keras.layers.Dense(2, activation="relu")
x = tf.constant([[1.0, 2.0, 3.0]])

with tf.GradientTape() as tape:
    # Forward pass
    y = layer(x)
    loss = tf.reduce_mean(y ** 2)

# Calculate gradients with respect to every trainable variable
grad = tape.gradient(loss, layer.trainable_variables)
for var, g in zip(layer.trainable_variables, grad):
    print(f"{var.name}, shape: {g.shape}")

