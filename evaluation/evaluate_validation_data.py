import tensorflow as tf
import pickle
import numpy as np

# load validation data
validation_data = pickle.load(open("../validation_data.pickle","rb"))

validation_images = []
validation_labels = []
for features, label in validation_data:
  validation_images.append(features)
  validation_labels.append(label)

# Convert to numpy array of type float32 in order to feed into the neural net
validation_images = np.array(validation_images).astype(np.float32)
validation_labels = np.array(validation_labels).astype(np.float32)

print("original without GAP")
model = tf.keras.models.load_model('original_model_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("original model without GAP")
model = tf.keras.models.load_model('original_model_no_GAP_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_1")
model = tf.keras.models.load_model('mod_1_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_2")
model = tf.keras.models.load_model('mod_2_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_3")
model = tf.keras.models.load_model('mod_3_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_4")
model = tf.keras.models.load_model('mod_4_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_5")
model = tf.keras.models.load_model('mod_5_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_6")
model = tf.keras.models.load_model('mod_6_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_7")
model = tf.keras.models.load_model('mod_7_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_8")
model = tf.keras.models.load_model('mod_8_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_9")
model = tf.keras.models.load_model('mod_9_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("mod_10")
model = tf.keras.models.load_model('mod_10_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)

print("depthwise")
model = tf.keras.models.load_model('depthwise_200_epochs')
val_loss, val_acc = model.evaluate(validation_images, validation_labels, batch_size=1)