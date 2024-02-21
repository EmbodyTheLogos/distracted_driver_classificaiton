import numpy as np
import pickle

# load training data
training_data = pickle.load(open("training_data.pickle", "rb"))

# count the number of each categories so we can create test data.
categories_counts = []
def get_category_count():
  count = 0
  current_class_num = training_data[0][1]
  for i in range(len(training_data)):
    if training_data[i][1] == current_class_num:
      count+=1
    else:
      categories_counts.append(count)
      count = 1
      current_class_num = training_data[i][1]
  categories_counts.append(count)
get_category_count()

# Split data: 60% for training, 20% for validation, and 20% for testing
validation_data = []
test_data = []
new_training_data = []

start_index = 0
for count in categories_counts:
  test_count = count * 20 / 100
  if test_count - int(test_count) >= 0.5:
    test_count = int(test_count) + 1
  else:
    test_count = int(test_count)
  test_data.extend(training_data[start_index:(start_index + test_count)])
  validation_data.extend(training_data[(start_index + test_count):(start_index + 2 * test_count)])
  new_training_data.extend(training_data[(start_index + 2 * test_count):(start_index + count)])
  start_index += count
  
#save data
pickle.dump(test_data, open("test_data.pickle", "wb"))
pickle.dump(validation_data, open("validation_data.pickle", "wb"))
pickle.dump(new_training_data, open("new_training_data.pickle", "wb"))
