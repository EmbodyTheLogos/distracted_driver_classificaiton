IMPORTANT NOTES:

----------------------------------------------------------------------------------------------------------------------------------
1. Prepare the dataset:

	First, download the dataset from https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data

	Unzipped the dataset, and put it in the folder "code to prepare for dataset" folder.

	Run "reformat dataset.py". This will load in the dataset (with the label), resize the images, and convert them to numpy array. The newly formatted datase will be saved as "training_data.pickle"

	Then run "divide dataset.py". This will divide the dataset into train data (60%), validation data (20%), and test data (20%). Three files will be saved "new_training_data.pickle", "validation_data.pickle", and "test_data.pickle" 
	
	Cut and paste the three files to the parent directory.

2. Train the model:

	The codes for training all of our models is in the folder "code for training" folder.

3. Evaluate the model:
	
	The codes for evaluating all of our models is in the folder "evalutation" folder.

	Note that if you want to load in saved models in the "evaluation" folder, you need Tensorflow version 2.9.1.

	Or you can retrain the models with whatever version Tensorflow you have, and use that same Tensorflow version to load in your trained models (this takes long time).

----------------------------------------------------------------------------------------------------------------------------------

RUN MODEL ON RASPBERRY PI 4

1. Update the system:
   
   sudo apt-get update
   
   sudo apt-get upgrade
   
3. Update pip3
   
   pip3 install --upgrade pip
   
5. Install tensorflow lite
   
   pip3 install tflite_runtime==2.9.1
   

