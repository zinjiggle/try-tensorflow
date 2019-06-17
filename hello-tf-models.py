
import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()


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

print(data)
print(labels)
print(len(labels))
print(len(labels[0]))
print(labels[0])


dataset = tf.data.Dataset.from_tensor_slices((data, labels))
print(dataset)
dataset = dataset.batch(32)
print(dataset)
# iterator = dataset.make_one_shot_iterator()
# for x, y in iterator:
#   print(x)
#   print(y)
#   exit(0)
dataset = dataset.repeat()
print(dataset)



