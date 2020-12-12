import os

import tensorflow as tf
import numpy as np
import cProfile

tf.executing_eagerly()

x = [[2.0]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

a = tf.constant([[1, 2], [3, 4]])
print(a)

# Broadcasting support
b = tf.add(a, 1)
print(b)

# Operator overloading is supported
print(a * b)

# Use NumPy values
c = np.multiply(a, b)
print(c)

# Obtain numpy value from a tensor:
print(a.numpy())
