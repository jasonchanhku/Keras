import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# normalize X
X = tf.keras.utils.normalize(X, axis =1)

# after identifying model of best fit
dense_layers = [0]
layer_sizes = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            print(f"Training: {NAME}")
            tensorboard = TensorBoard(log_dir = f"logs/{NAME}")
            # Building the model
            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2,2)))
            
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())
            
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))
                
            model.add(Dense(64))
            model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss = "binary_crossentropy",
                         optimizer = "adam",
                         metrics = ['accuracy'])

            model.fit(X, y, batch_size = 32, epochs = 10, validation_split = 0.3, callbacks = [tensorboard])
model.save('final.model')