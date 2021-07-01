import tensorflow as tf
import numpy as np
from tensorflow import keras

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
	batch_size=31
)

# build model
model = tf.keras.Sequential([ # sequence of layers of neurons
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

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(train_data, epochs=10)

model.evaluate(test_data)