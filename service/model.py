# contains code for the machine learing model to predict brain tumours
import sys
import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from display import plotAccuracyLoss
from display import displayActivations
from PIL import Image
from preprocess import crop_image


#crop images to only contain brain section - YES
x = 1
for filename in os.listdir('original_data/train/yes'):
	#print(filename)

	img = cv2.imread(os.path.join('original_data/train/yes', filename))
	if img is None:
		print('err - cant read img')
		continue

	processed_img = crop_image(img)

	save_filename = 'processed/train/yes/Y' + str(x) + '.jpg'
	x +=1
	cv2.imwrite(save_filename, processed_img)

#crop images to only contain brain section - NO
x = 1
for filename in os.listdir('original_data/train/no'):
	#print(filename)

	img = cv2.imread(os.path.join('original_data/train/no', filename))
	if img is None:
		print('err - cant read img')
		continue

	processed_img = crop_image(img)

	save_filename = 'processed/train/no/N' + str(x) + '.jpg'
	x +=1
	cv2.imwrite(save_filename, processed_img)


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

#load training data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
	"processed/train",
	image_size=(256, 256),
	batch_size=8,
	label_mode='int',
	shuffle=True,
	color_mode='rgb'
)

#load testing data
test_data = tf.keras.preprocessing.image_dataset_from_directory(
	"processed/test",
	image_size=(256, 256),
	label_mode='int',
	shuffle=True,
	color_mode='rgb'
)

# build model
model = tf.keras.Sequential([ # sequence of layers of neurons
	tf.keras.layers.Input(shape=(256, 256, 3)), # define the input shape
	tf.keras.layers.experimental.preprocessing.Rescaling(1./255), # prescale all values to 0-1 instead of 0-255
	#tf.keras.layers.experimental.preprocessing.RandomRotation((-0.2, 0.3)), # Add a random rotation to the image
	#tf.keras.layers.experimental.preprocessing.RandomFlip(), # Add a random flip to the image
	tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
	tf.keras.layers.MaxPooling2D(),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(2)
])

# compile model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

history = None
num_epochs = 15

if 'fit' in sys.argv:
	model.load_weights(bestFilePath)
	history = model.fit(train_data, epochs=num_epochs, callbacks=[csv_logger, checkpoint])
elif 'new' in sys.argv:
	history = model.fit(train_data, epochs=num_epochs, callbacks=[csv_logger, checkpoint])
else:
	model.load_weights(bestFilePath)

if 'display' in sys.argv:
	if history:
		plotAccuracyLoss(history)
	displayActivations(model, train_data.take(1))

model.summary()

results = np.argmax(model.predict(test_data), axis=1)
labels = np.concatenate([y for x, y in test_data], axis=0)
print("Error:")
print(np.mean(results != labels))


# fit does not work, it re-trains
# can we process the testing data? or only training data?
# display does not work
# model does not work, had to comment out rotation and flip