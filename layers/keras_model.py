import tensorflow as tf

from keras_layer import FlexibleDense


class MySequentialModel(tf.keras.Model):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)

        self.dense_1 = FlexibleDense(out_features=3)
        self.dense_2 = FlexibleDense(out_features=2)

    def call(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


# You have made a Keras model!
my_sequential_model = MySequentialModel(name="the_model")

# Call it on a tensor, with random results
print("Model results:", my_sequential_model(tf.constant([[2.0, 2.0, 2.0]])))


# A raw tf.Module nested inside a Keras layer or model will not get its variables
# collected for training or saving. Instead, nest Keras layers inside of Keras layers.
print(my_sequential_model.variables)
print(my_sequential_model.submodules)
