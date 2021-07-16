# displays model data to graphs
# code from https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.callbacks import CSVLogger
from matplotlib import pyplot as plt

def plotAccuracyLoss(history):
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)
	plt.plot(epochs, acc, color='b', label='Training acc')
	plt.plot(epochs, val_acc, color='r', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.figure()
	plt.plot(epochs, loss, color='b', label='Training loss')
	plt.plot(epochs, val_loss, color='r', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.show()

def displayActivations(model, image):
	layer_outputs = [layer.output for layer in model.layers[3:15]]
	print(layer_outputs)
	layer_names = [layer.name for layer in model.layers[3:9]]
	activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
	activations = activation_model.predict(image)

	images_per_row = 4 # number of images per row on the graphs

	for layer_name, layer_activation in zip(layer_names, activations): # for each layer and name
		# (31, size, size, n_features)
		n_features = layer_activation.shape[-1]
		size = layer_activation.shape[1]
		no_cols = n_features // images_per_row
		display_grid = np.zeros((size * no_cols, images_per_row * size))
		for col in range(no_cols):
			for row in range(images_per_row):
				channel_image = layer_activation[0, :, :, col * images_per_row + row]
				display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image
		scale = 1. / size
		plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
		plt.title(layer_name)
		plt.grid(False)
		plt.imshow(display_grid, aspect='auto', cmap='viridis')
	plt.show()
