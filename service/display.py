# displays model data to graphs
# code from https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import models
from tensorflow.keras.callbacks import CSVLogger
from matplotlib import pyplot as plt

def plotAccuracyLoss(history):
	acc = history.history['accuracy']
	val_acc = history.history['accuracy']
	loss = history.history['loss']
	val_loss = history.history['loss']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

def displayActivations(model, image):
	layer_outputs = [layer.output for layer in model.layers]
	activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
	activations = activation_model.predict(image)
	print(activations[0].shape)
	print()
	for layer_activation in activations[:7]:
		plt.matshow(layer_activation[0, :, :, 2], cmap='viridis')
	plt.show()
