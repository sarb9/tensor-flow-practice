import tensorflow as tf

# Usefull notes from official site:

# 1.Basically tf.Variable objects store mutable tf.Tensor-like values
# accessed during training to make automatic differentiation easier.

# 2.The collections of variables can be encapsulated into layers
# or models, along with methods that operate on them.

# 3.The main difference between layers and models is that models
# add methods like Model.fit, Model.evaluate, and Model.save.


class Linear(tf.keras.Model):
    def __init__(self):
        super(Linear, self).__init__()
        self.W = tf.Variable(5.0, name="weight")
        self.B = tf.Variable(10.0, name="bias")

    def call(self, inputs):
        return inputs * self.W + self.B

    def trainable_variables(self):
        return [self.W, self.B]


# A toy dataset of points around 3 * x + 2
NUM_EXAMPLES = 2000
training_inputs = tf.random.normal([NUM_EXAMPLES])
noise = tf.random.normal([NUM_EXAMPLES])
training_outputs = training_inputs * 3 + 2 + noise

# The loss function to be optimized.
def loss(model, inputs, targets):
    error = model(inputs) - targets
    return tf.reduce_mean(tf.square(error))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.trainable_variables())


model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

print("Initial loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))

steps = 300
for i in range(steps):
    grads = grad(model, training_inputs, training_outputs)
    optimizer.apply_gradients(zip(grads, model.trainable_variables()))
    if i % 20 == 0:
        print(
            "Loss at step {:03d}: {:.3f}".format(
                i, loss(model, training_inputs, training_outputs)
            )
        )

print("Final loss: {:.3f}".format(loss(model, training_inputs, training_outputs)))
print("W = {}, B = {}".format(model.W.numpy(), model.B.numpy()))


# Create a checkpoint:
model.save_weights("weights")
status = model.load_weights("weights")

x = tf.Variable(10.0)
checkpoint = tf.train.Checkpoint(x=x)
