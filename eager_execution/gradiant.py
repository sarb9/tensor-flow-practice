import tensorflow as tf

w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w * w

# A particular tf.GradientTape can only compute one gradient;
# subsequent calls throw a runtime error!!
grad = tape.gradient(loss, w)
print(grad)
