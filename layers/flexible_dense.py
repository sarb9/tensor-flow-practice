import tensorflow as tf


class FlexibleDenseModule(tf.Module):
    # Note: No need for `in+features`
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    def __call__(self, x):
        # Create variables on first call.
        if not self.is_built:
            self.w = tf.Variable(
                tf.random.normal([x.shape[-1], self.out_features]), name="w"
            )
            self.b = tf.Variable(tf.zeros([self.out_features]), name="b")
            self.is_built = True

        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)


# Used in a module
class MySequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = FlexibleDenseModule(out_features=3)
        self.dense_2 = FlexibleDenseModule(out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


my_model = MySequentialModule(name="the_model")
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))


chkp_path = "my_checkpoint/checkk"
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(chkp_path)
checkpoint.write(chkp_path)

# Look inside a checkpoint
print(tf.train.list_variables(chkp_path))

new_model = MySequentialModule()
new_checkpoint = tf.train.Checkpoint(model=new_model)
new_checkpoint.restore(chkp_path)

# Should be the same result as above
new_model(tf.constant([[2.0, 2.0, 2.0]]))

tf.saved_model.save(my_model, "the_saved_model")
new_model = tf.saved_model.load("the_saved_model")
print(isinstance(new_model, MySequentialModule))
