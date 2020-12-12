import os

import tensorflow as tf


x = tf.Variable(10.0)
checkpoint = tf.train.Checkpoint(x=x)

x.assign(2.0)  # Assign a new value to the variables and save.
checkpoint_path = "./ckpt/"
checkpoint.save("./ckpt/")

x.assign(11.0)  # Change the variable after saving.

# Restore values from the checkpoint
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))

print(x)  # => 2.0


# To save and load models, tf.train.Checkpoint stores the
# internal state of objects, without requiring hidden variables.
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(16, [3, 3], activation="relu"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10),
    ]
)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

checkpoint_dir = "ckpt/model/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

root = tf.train.Checkpoint(optimizer=optimizer, model=model)
root.save(checkpoint_prefix)

# Restore the model as well as optimizer state
root.restore(tf.train.latest_checkpoint(checkpoint_dir))
