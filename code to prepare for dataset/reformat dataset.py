import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import pickle

# Create train data
DATADIR = "imgs/train"
CATEGORIES = ["c0", "c1","c2","c3","c4","c5","c6","c7","c8","c9"]
training_data = []
def create_training_data():
  for category in CATEGORIES:
    path = os.path.join(DATADIR, category) #  path to different types of drivers
    class_num = CATEGORIES.index(category)
    print(category)
    for img in os.listdir(path):
      try:
        # img_array = cv2.imread(os.path.join(path, img))
        # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize to 224 x 244
        img_array = Image.open(os.path.join(path, img))
        img_array = img_array.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_array)
        training_data.append([img_array, class_num])
      except Exception as e:
        print(e)
        pass
create_training_data()

# save training_data
pickle.dump(training_data, open("training_data.pickle","wb"))