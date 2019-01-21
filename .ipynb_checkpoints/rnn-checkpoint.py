import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import TensorBoard

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
print(f"Shape of x_train is: {x_train.shape}")
print(f"For each row, the shape is: {x_train[0].shape}")

# treat the row pixels as a sequence for the RNN to learn an to feed it to

NAME = "First RNN"
tensorboard = TensorBoard(log_dir = f"rnnlogs/{NAME}")

model = Sequential()
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation = 'relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(10, activation = "softmax"))

opt = tf.keras.optimizers.Adam(lr = 1e-3, decay = 1e-5)

model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 5, validation_data = (x_test, y_test), callbacks = [tensorboard])