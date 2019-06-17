from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)


# Create a sigmoid layer:
x = layers.Dense(64, activation='sigmoid')
print(x)
# Or:
x = layers.Dense(64, activation=tf.sigmoid)
print(x)
# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
x = layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
print(x)
# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
x = layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
print(x)
# A linear layer with a kernel initialized to a random orthogonal matrix:
x = layers.Dense(64, kernel_initializer='orthogonal')
print(x)
# A linear layer with a bias vector initialized to 2.0s:
x = layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
print(dir(x))
print(len(tf.get_default_graph().get_operations()))


model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))

print(dir(model))
print(dir(layers))
print(len(tf.get_default_graph().get_operations()))

def print_graph():
  print("Default graph")
  print(len(tf.get_default_graph().get_operations()))
  # for op in tf.get_default_graph().get_operations():
  #   print(op.node_def)

# Compile the model.
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print_graph()
# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error
print_graph()
# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
print_graph()


import numpy as np

def random_one_hot_labels(shape):
  n, n_class = shape
  classes = np.random.randint(0, n_class, n)
  labels = np.zeros((n, n_class))
  labels[np.arange(n), classes] = 1
  return labels

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))

val_data = np.random.random((100, 32))
val_labels = random_one_hot_labels((100, 10))

# Fit without validation
# model.fit(data, labels, epochs=10, batch_size=32)

# Fit with validation
# model.fit(data, labels, epochs=10, batch_size=32,
#           validation_data=(val_data, val_labels))


# # Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

# # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
# model.fit(dataset, epochs=10, steps_per_epoch=30)
model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

data = np.random.random((1000, 32))
print(data.shape)
labels = np.random.random((1000, 10))

# model.evaluate(data, labels, batch_size=32)
model.evaluate(dataset, steps=50)

result = model.predict(data, batch_size=32)
print(result)
print(result.shape)


class MyLayer(layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    shape = tf.TensorShape((input_shape[1], self.output_dim))
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=shape,
                                  initializer='uniform',
                                  trainable=True)
    # Make sure to call the `build` method at the end
    super(MyLayer, self).build(input_shape)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.output_dim
    return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

# The compile step specifies the training configuration
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)



class MyModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)


model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
