import numpy as np
import pickle
from tensorflow.python.ops.gen_array_ops import batch_matrix_set_diag
import keras.backend
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D
import sklearn

# load train data
new_training_data = pickle.load(open("../../new_training_data.pickle", "rb"))

# Separate training_images and training_labels
training_images = []
training_labels = []
for features, label in new_training_data:
  training_images.append(features)
  training_labels.append(label)
   
# Convert to numpy array of type float32 in order to feed into the neural net
training_images = np.array(training_images).astype(np.float32)
training_labels = np.array(training_labels).astype(np.float32)

# shuffle train data: https://stackoverflow.com/questions/38190476/use-of-random-state-parameter-in-sklearn-utils-shuffle
training_images, training_labels = sklearn.utils.shuffle(training_images, training_labels, random_state=10)

model = tf.keras.models.Sequential() # define model type

#block 1
model.add(Conv2D(16, (7, 7), padding="same"))
model.add(Conv2D(16, (7, 7), padding="same"))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

#block 2
# model.add(Conv2D(32, (5, 5), padding="same"))
# model.add(Conv2D(32, (5, 5), padding="same"))
# model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))

#block 3
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))

#block 4
# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(AveragePooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))


# model.add(Conv2D(256, (3, 3), padding="same"))
# model.add(Conv2D(10, (3, 3), padding="same"))
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation=tf.nn.softmax)) # add output layer

optimizer = tf.keras.optimizers.Adam(0.0001) #learning rate

model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

print("Learning rate:", tf.keras.backend.eval(model.optimizer.lr))

for i in range(1, 110):
    model.fit(training_images, training_labels, epochs=10, batch_size=16)
    model.save("original_model_{}_epochs".format(i*10))
    print("saved model with epoch " + str(i*10))
               
status = "all_epochs_saved"
f = open("save_status.txt", "w")
f.write(status)
f.close()
