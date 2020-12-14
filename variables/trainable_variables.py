import tensorflow as tf
import matplotlib.pyplot as plt

# A trainable variable
x0 = tf.Variable(3.0, name="x0")
# Not trainable
x1 = tf.Variable(3.0, name="x1", trainable=False)
# Not a Variable: A variable + tensor returns a tensor.
x2 = tf.Variable(2.0, name="x2") + 1.0
# Not a variable
x3 = tf.constant(3.0, name="x3")

with tf.GradientTape() as tape:
    y = (x0 ** 2) + (x1 ** 2) + (x2 ** 2)

grad = tape.gradient(y, [x0, x1, x2, x3])

for g in grad:
    print(g)

print([var.name for var in tape.watched_variables()])

# Example of watch method
x = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2

# dy = 2x * dx
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())


# Disable default mode for watching all variables
x0 = tf.Variable(0.0)
x1 = tf.Variable(10.0)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.nn.softplus(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)
# dy = 2x * dx
grad = tape.gradient(ys, {"x0": x0, "x1": x1})

print("dy/dx0:", grad["x0"])
print("dy/dx1:", grad["x1"].numpy())

# Persistant gradiant tape
x = tf.constant([1, 3.0])
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x * x
    z = y * y

print(tape.gradient(z, x).numpy())  # 108.0 (4 * x**3 at x = 3)
print(tape.gradient(y, x).numpy())  # 6.0 (2 * x)

del tape


# Calling gradiant on multiple targets returns sum of each target's gradiants
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y0 = x ** 2
    y1 = 1 / x

print(tape.gradient({"y0": y0, "y1": y1}, x).numpy())

# Same will happen for non-scaller targets
x = tf.Variable(2.0)
with tf.GradientTape() as tape:
    y = x * [3.0, 4.0]

print(tape.gradient(y, x).numpy())

# This makes it simple to take the gradient of the sum of a collection of losses,
# or the gradient of the sum of an element-wise loss calculation.

# But if the shapes are matched with each other, the jacobians will be computed.
x = tf.linspace(-10.0, 10.0, 200 + 1)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.nn.sigmoid(x)

dy_dx = tape.gradient(y, x)

plt.plot(x, y, label="y")
plt.plot(x, dy_dx, label="dy/dx")
plt.legend()
plt.xlabel("x")
plt.show()
