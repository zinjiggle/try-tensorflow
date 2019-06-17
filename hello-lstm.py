
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, CuDNNLSTM

print(tf.VERSION)
print(tf.keras.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)
print(x_train[0].shape)
print(x_train.shape[1:])

model = Sequential()
model.add(LSTM(128, input_shape=(28, 28), activation='relu', return_sequences=True))
# Train using GPU, I have Error:  OpKernel was registered to support Op 'CudnnRNN'
# model.add(CuDNNLSTM(128, input_shape=(28, 28), return_sequences=True))
# model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
# model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'],)

tf.keras.utils.plot_model(model,
                          to_file='simple-lstm.svg',
                          show_shapes=True,
                          show_layer_names=True)


model.summary()

print(model.to_json())

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./lstm-logs', write_graph=True, write_images=True)

# model.fit(x_train, y_train,
#           epochs=1,
#           validation_data=(x_test, y_test),
#           # callbacks=[tensorboard_callback],
#           )

model.save('hello-lstm.h5')




