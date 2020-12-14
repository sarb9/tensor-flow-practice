import tensorflow as tf
from keras_layer import FlexibleDense

inputs = tf.keras.Input(shape=[3,])

x = FlexibleDense(3)(inputs)
x = FlexibleDense(2)(x)

my_functional_model = tf.keras.Model(inputs=inputs, outputs=x)

print(my_functional_model.summary())
