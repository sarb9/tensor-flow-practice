import tensorflow as tf


x = tf.constant(1.0)

v0 = tf.Variable(2.0)
v1 = tf.Variable(2.0)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    if x > 0.0:
        result = v0
    else:
        result = v1 ** 2

dv0, dv1 = tape.gradient(result, [v0, v1])

print(dv0)
print(dv1)


# Getting a gradient of None

# When the targets and variables loose their connection to each other,
# the gradiant will be None.

# Fault number one:
x = tf.Variable(2.0)
for epoch in range(2):
    with tf.GradientTape() as tape:
        y = x + 1

    print(type(x).__name__, ":", tape.gradient(y, x))
    x = x + 1  # This should be x.assign_add(1)

# Always use assign with variables

# Fault number two:
# Integers and strings are not differentiable!

# The x0 variable has an int dtype.
x = tf.Variable([[2, 2], [2, 2]])

with tf.GradientTape() as tape:
    # The path to x1 is blocked by the int dtype here.
    y = tf.cast(x, tf.float32)
    y = tf.reduce_sum(x)

print(tape.gradient(y, x))

# Fault number three:
# taking gradiant from stateful object
x0 = tf.Variable(3.0)
x1 = tf.Variable(0.0)

with tf.GradientTape() as tape:
    # Update x1 = x1 + x0.
    x1.assign_add(x0)
    # The tape starts recording from x1.
    y = x1 ** 2  # y = (x1 + x0)**2

# This doesn't work.
print(tape.gradient(y, x0))  # dy/dx0 = 2*(x1 + x2)

