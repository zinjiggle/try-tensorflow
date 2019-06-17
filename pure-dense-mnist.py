
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Bidirectional, Flatten, Conv1D

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)
print(x_train[0].shape)
print(x_train.shape[1:])

model = Sequential()

# Adding conv layer, but this does not help the accuracy though.
# model.add(Conv1D(filters=128, kernel_size=(3), use_bias=True, activation='relu'))
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu',use_bias=True))
model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],)

tf.keras.utils.plot_model(model,
                          to_file='simple-bi-lstm.svg',
                          show_shapes=True,
                          show_layer_names=True)


model.summary()

print(model.to_json())

model.fit(x_train, y_train,
          epochs=10,
          validation_data=(x_test, y_test),
          # callbacks=[tensorboard_callback],
          )

model.save('simple-mnist.h5')



