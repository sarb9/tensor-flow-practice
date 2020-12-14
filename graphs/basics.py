import tensorflow as tf
import timeit
from datetime import datetime


# Define a Python function
def function_to_get_faster(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


# Create a `Function` object that contains a graph
a_function_that_uses_a_graph = tf.function(function_to_get_faster)

# Make some tensors
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

# It just works!
a_function_that_uses_a_graph(x1, y1, b1).numpy()


def inner_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


# Use the decorator
@tf.function
def outer_function(x):
    y = tf.constant([[2.0], [3.0]])
    b = tf.constant(4.0)

    return inner_function(x, y, b)


# Note that the callable will create a graph that
# includes inner_function() as well as outer_function()
outer_function(tf.constant([[1.0, 2.0]])).numpy()


def my_function(x):
    if tf.reduce_sum(x) <= 1:
        return x * x
    else:
        return x - 1


a_function = tf.function(my_function)

print("First branch, with graph:", a_function(tf.constant(1.0)).numpy())
print("Second branch, with graph:", a_function(tf.constant([5.0, 5.0])).numpy())

# Don't read the output too carefully.
print(tf.autograph.to_code(my_function))
