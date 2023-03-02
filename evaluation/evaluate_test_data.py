import tensorflow as tf
import pickle
import numpy as np

# load test data
test_data = pickle.load(open("../test_data.pickle","rb"))

test_images = []
test_labels = []
for features, label in test_data:
  test_images.append(features)
  test_labels.append(label)

# Convert to numpy array of type float32 in order to feed into the neural net
test_images = np.array(test_images).astype(np.float32)
test_labels = np.array(test_labels).astype(np.float32)

print("original model")
model = tf.keras.models.load_model('original_model_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("depthwise of mod_10")
model = tf.keras.models.load_model('depthwise_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)