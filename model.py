# contains code for the machine learing model to predict brain tumours
import sys
import os
import tensorflow as tf
import numpy as np
import scipy
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import sklearn
from display import plotAccuracyLoss
from display import displayActivations
import PIL
from PIL import Image, UnidentifiedImageError
from preprocess import crop_image


#crop images to only contain brain section - YES
if 'preprocess' in sys.argv:
	x = 1
	print("---- Cropping Training Images ----")
	for filename in os.listdir('original_data2/train/yes'):
		#print(filename)

		img = cv2.imread(os.path.join('original_data2/train/yes', filename))
		if img is None:
			print('err - cant read img')
			continue

		processed_img = crop_image(img)

		save_filename = 'processed/train/yes/Y' + str(x) + '.jpg'
		x +=1
		cv2.imwrite(save_filename, processed_img)

	#crop images to only contain brain section - NO
	x = 1
	for filename in os.listdir('original_data2/train/no'):
		#print(filename)

		img = cv2.imread(os.path.join('original_data2/train/no', filename))
		if img is None:
			print('err - cant read img')
			continue

		processed_img = crop_image(img)

		save_filename = 'processed/train/no/N' + str(x) + '.jpg'
		x +=1
		cv2.imwrite(save_filename, processed_img)

print("---- Loading Training/Testing Data ----")
img_shape = (200, 200)

# load train images
processed_train_path = "./processed/train"
train_filenames_no = os.listdir(processed_train_path+"/no/")
train_data = []
train_labels = []
for filename in train_filenames_no:
	img = load_img(processed_train_path+"/no/"+filename).resize(img_shape)
	train_data.append(img_to_array(img))
	train_labels.append(0)

train_filenames_yes = os.listdir(processed_train_path+"/yes/")
for filename in train_filenames_yes:
	try:
		img = load_img(processed_train_path+"/yes/"+filename).resize(img_shape)
	except PIL.UnidentifiedImageError:
		print(filename)
	train_data.append(img_to_array(img))
	train_labels.append(1)

# load test images and make a dataframe
processed_test_path = "./original_data/test"
test_filenames_no = os.listdir(processed_test_path+"/no/")
test_data = []
test_labels = []
for filename in test_filenames_no:
	img = load_img(processed_test_path+"/no/"+filename).resize(img_shape)
	test_data.append(img_to_array(img))
	test_labels.append(0)

test_filenames_yes = os.listdir(processed_test_path+"/yes/")
for filename in test_filenames_yes:
	img = load_img(processed_test_path+"/yes/"+filename).resize(img_shape)
	test_data.append(img_to_array(img))
	test_labels.append(1)

# shuffle the data
train_data, train_labels = sklearn.utils.shuffle(train_data, train_labels)

print("---- Creating Callback Functions ----")
#create a logger for our epochs
csv_logger =  CSVLogger("history_log.csv", append=False)

# Save the weights of the model if it is best
bestFilePath = 'best_weights'
checkpoint = tf.keras.callbacks.ModelCheckpoint(
	filepath=bestFilePath,
	monitor='loss',
	verbose=1,
	mode="auto",
	save_best_only=True,
	save_weights_only=True,
)

# stop the model early if the model is overfitting for 10 epochs.
earlyStop = EarlyStopping(patience=10)

# reduce the learning rate when accuracy is not increasing.
learningRateReduction = ReduceLROnPlateau(
	monitor='val_accuracy',
	patience=2,
	verbose=1,
	factor=0.5,
	min_lr=0.00001
)

callbacks = [csv_logger, checkpoint, earlyStop, learningRateReduction]

print("---- Building Model ----")
# build model
model = tf.keras.Sequential([ # sequence of layers of neurons
	tf.keras.layers.Input(shape=(200, 200, 3)), # define the input shape
	tf.keras.layers.experimental.preprocessing.Rescaling(1./255), # prescale all values to 0-1 instead of 0-255
	tf.keras.layers.experimental.preprocessing.RandomRotation((-0.2, 0.3)), # Add a random ration to the image
	tf.keras.layers.experimental.preprocessing.RandomFlip(), # Add a random flip to the image
	tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(
  optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy']
)

history = None
num_epochs = 30

if 'new' in sys.argv:
	print("---- Fitting Model ----")
	history = model.fit(
		x=np.array(train_data),
		y=np.array(train_labels),
		validation_split=0.25,
		batch_size=32,
		shuffle=True,
		epochs=num_epochs, 
		callbacks=callbacks)
else:
	model.load_weights(bestFilePath)

if 'display' in sys.argv:
	if history:
		plotAccuracyLoss(history)
	
	#displayActivations(model, np.array(x_train[0]))

model.summary()

results = np.rint(model.predict(np.array(test_data)))
print("Final Accuracy:")
print(np.mean(results == test_labels))