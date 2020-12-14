import tensorflow as tf

# A tf.Variable represents a tensor whose value can be changed by running ops on it.

# For watching where variables get placed
tf.debugging.set_log_device_placement(True)

my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
my_variable = tf.Variable(my_tensor)
bool_variable = tf.Variable([False, False, False, True])
complex_variable = tf.Variable([5 + 4j, 6 + 1j])

print("Shape: ", my_variable.shape)
print("DType: ", my_variable.dtype)
print("As NumPy: ", my_variable.numpy())

print("A variable:", my_variable)
print("Viewed as a tensor:", tf.convert_to_tensor(my_variable))
print("Index of highest value:", tf.argmax(my_variable))

# This creates a new tensor; it does not reshape the variable.
print("Copying and reshaping: ", tf.reshape(my_variable, ([1, 4])))

a = tf.Variable([2.0, 3.0])
# This will keep the same dtype, float32
a.assign([1, 2])
# Not allowed as it resizes the variable:
try:
    a.assign([1.0, 2.0, 3.0])
except Exception as e:
    print(f"{type(e).__name__}: {e}")


# Create b based on the value of a
a = tf.Variable([2.0, 3.0])
b = tf.Variable(a)
a.assign([5, 6])
# a and b are different
print(a.numpy())
print(b.numpy())

# There are other versions of assign
print(a.assign_add([2, 3]).numpy())  # [7. 9.]
print(a.assign_sub([7, 9]).numpy())  # [0. 0.]

# Create a and b; they will have the same name but will be backed by
# different tensors.
a = tf.Variable(my_tensor, name="Mark")
# A new variable with the same name, but different value
# Note that the scalar add is broadcast
b = tf.Variable(my_tensor + 1, name="Mark")

# These are elementwise-unequal, despite having the same name
print(a == b)

# Disable automatic differentiation
step_counter = tf.Variable(1, trainable=False)

# If you had multiple GPU workers but only want one copy of the variables
with tf.device("CPU:0"):
    a = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.Variable([[1.0, 2.0, 3.0]])

with tf.device("GPU:0"):
    # Element-wise multiply
    k = a * b
print(k)
