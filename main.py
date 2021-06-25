#!/usr/bin/env python3

from keras.datasets import mnist
from matplotlib import pyplot




def load_MNIST_dataset():
	"""
		Download the MNIST (Modified National Institute of Standards and Technology)
		image database of handwritten digits. 

		This dataset contains 60,000 training images and 10,000 testing images.

		Each image uses a dimension of 28x28 pixels.

	Returns:
		Tuple of NumPy arrays: (x_train, y_train), (x_test, y_test). 

		x_train (uint8 NumPy array): Gray-scale image data with shapes (60000, 28, 28), containing the training data. 
				 					 Pixel values range from 0 to 255.

		y_train (uint8 NumPy array): Digit labels (integers in range 0-9) 
									 with shape (60000,) for the training data.

		x_test (uint8 NumPy array): Gray-scale image data with shapes (10000, 28, 28), containing the test data. 
									Pixel values range from 0 to 255.

		y_test (uint8 NumPy array): Digit labels (integers in range 0-9) 
									with shape (10000,) for the test data.
	"""
	# Get the data.
	(training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

	# Reshape the data to have a single color channel, reducing the size of the dataset.
	training_images = training_images.reshape((training_images.shape[0], 28, 28, 1)) 
	testing_images = testing_images.reshape((testing_images.shape[0], 28, 28, 1))

	# Use one-hot encoding for the target values.
	training_labels = to_categorical(training_labels)
	testing_labels = to_categorical(testing_labels)

	# Return the data.
	return training_images, training_labels, testing_images, testing_labels




def prepare_pixel_data(training_images, testing_images):
	"""
		Scale down the pixel values for the image data. 

		This is a pre-processing technique known as normalization, which is necessary
		for training the neural network.	

	Args:
		training_images	(uint8 NumPy array): Training images to normalize

		testing_images (uint8 NumPy array): Testing images to normalize

	Returns:
		Tuple of NumPy arrays: (training_images_normalized, testing_images_normalized)
	"""

	# Convert from integers to floats.
	training_images_normalized = training_images.astype('float32')
	testing_images_normalized = testing_images.astype('float32')

	# Normalize pixel values to range 0-1.
	training_images_normalized = training_images_normalized / 255.0
	testing_images_normalized = testing_images_normalized / 255.0

	# Return the data.
	return training_images_normalized, testing_images_normalized




def create_model():

	model = Sequential()

	# 3x3 kernel size with 32 filters is most commonly used for convolutional layers.
	# 2x2 is common for pooling layers.
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2))) 

	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))

	# output layer with 10 classes for digits 0-9
	model.add(Dense(10, activation='softmax'))

	# compile model
	optimizer = SGD(lr=0.01, momentum=0.9) # SGD => Stochastic Gradient Descent
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	return model


# Load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# Summarize loaded dataset
print('Train: X=%s, Y=%s' % (trainX.shape, trainY.shape))
print('Test: X=%s, Y=%s' % (testX.shape, testY.shape))
print(type(trainX))

# Plot and display the first 9 images in the training set.
for i in range(9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
