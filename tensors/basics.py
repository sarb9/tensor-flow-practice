import tensorflow as tf
import numpy as np

# This will be an int32 tensor by default
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)


# This one is a float tensor.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)


# Create a matrix tensor with specified dtype
rank_2_tensor = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

# Create a rank 3 tensor
rank_3_tensor = tf.constant(
    [
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29]],
    ]
)
print(rank_3_tensor)


# Convert a tensor to a NumPy array
assert (rank_2_tensor.numpy() == np.array(rank_2_tensor)).all()


# Multiplication
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[1, 1], [1, 1]])

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")  # element-wise
print(tf.matmul(a, b), "\n")

print(a + b, "\n")
print(a * b, "\n")  # element-wise
print(a @ b, "\n")

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Some other usages of tensors

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))


# Create a tensor with zeros as initial values
rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# ** normally, axes are ordered from global to local i.e. batch, width, height, features.

# Shaping:
x = tf.constant([[1], [2], [3]])
print(x.shape)
print(x.shape.as_list())

reshaped = tf.reshape(x, [1, 3])
print(x.shape)
print(reshaped.shape)

# Data actually don't get replaced by reshaping
# If you want to see how data reside in memory you can flatten the array
# A -1 passed in the shape argument says "Whatever fits".
print(tf.reshape(rank_3_tensor, [-1]))

# Usually the only reasonable thing to do with reshape is combine and split adjacent axes
print(tf.reshape(rank_3_tensor, [3 * 2, 5]), "\n")
print(tf.reshape(rank_3_tensor, [3, -1]))
