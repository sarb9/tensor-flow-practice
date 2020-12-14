import tensorflow as tf
from datetime import datetime


class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(
            5.0, trainable=False, name="do_not_train_me"
        )

    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_variable


simple_module = SimpleModule(name="simple")

print(simple_module(tf.constant(5.0)))
# All trainable variables
print("trainable variables:", simple_module.trainable_variables)
# Every variable
print("all variables:", simple_module.variables)
