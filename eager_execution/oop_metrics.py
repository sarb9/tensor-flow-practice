import tensorflow as tf

m = tf.keras.metrics.Mean("test")

# Update a metric by passing the new data to the callable.
m(0)
m(5)
print(m.result())  # => 2.5
m([8, 9])
print(m.result())  # => 5.5
