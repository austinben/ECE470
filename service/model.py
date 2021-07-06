# contains code for the machine learing model to predict brain tumours
import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger
from display import plotAccuracyLoss
from display import displayActivations

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

#load trining data
train_data = tf.keras.preprocessing.image_dataset_from_directory(
	"archive/train",
	image_size=(200, 200),
	batch_size=31
)

#load testing data
test_data = tf.keras.preprocessing.image_dataset_from_directory(
	"archive/test",
	image_size=(200, 200),
)

# build model
model = tf.keras.Sequential([ # sequence of layers of neurons
  tf.keras.layers.Input(shape=(200, 200, 3)),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255), # prescale all values to 0-1 instead of 0-255
  tf.keras.layers.Conv2D(32, 3, activation='relu'), # first convolution layer, relu removes any negative values and returns 0 if x < 0 
  tf.keras.layers.MaxPooling2D(), # pooling condenses image to promote features. Takes max value from 2x2?
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'), # final layer
  tf.keras.layers.Dense(2)
])

# compile model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy']
)

history = None

if 'fit' in sys.argv:
	model.load_weights(bestFilePath)
	history = model.fit(train_data, epochs=10, callbacks=[csv_logger, checkpoint])
elif 'new' in sys.argv:
	history = model.fit(train_data, epochs=10, callbacks=[csv_logger, checkpoint])

if 'display' in sys.argv:
	if history:
		plotAccuracyLoss(history)
	displayActivations(model, train_data.take(1))
	
model.evaluate(test_data)

model.summary()