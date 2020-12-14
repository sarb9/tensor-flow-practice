import tensorflow as tf


class MyDense(tf.keras.layers.Layer):
    # Adding **kwargs to support base Keras layer arguemnts
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)

        # This will soon move to the build step; see below
        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name="w")
        self.b = tf.Variable(tf.zeros([out_features]), name="b")

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


simple_layer = MyDense(name="simple", in_features=3, out_features=3)


class FlexibleDense(tf.keras.layers.Layer):
    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, out_features, **kwargs):
        super().__init__(**kwargs)
        self.out_features = out_features

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.w = tf.Variable(
            tf.random.normal([input_shape[-1], self.out_features]), name="w"
        )
        self.b = tf.Variable(tf.zeros([self.out_features]), name="b")

    def call(self, inputs):  # Defines the computation from inputs to outputs
        return tf.matmul(inputs, self.w) + self.b


# Create the instance of the layer
flexible_dense = FlexibleDense(out_features=3)

# No vairable because no input has been passed
print(flexible_dense.variables)

# Call it, with predictably random results
print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])))

# This call will have some output because the build function of the model has
# been called previously
print(flexible_dense.variables)

# This will be failed because build method has been called already and the
# shape of the inputs are determined
try:
    print("Model results:", flexible_dense(tf.constant([[2.0, 2.0, 2.0, 2.0]])))
except tf.errors.InvalidArgumentError as e:
    print("Failed:", e)
