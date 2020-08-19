from tensorflow import optimizers
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

plt.imshow(train_images[0])

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy')

model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images, test_labels)

# predict single test data
classifications = model.predict(test_images)
print(classifications)
