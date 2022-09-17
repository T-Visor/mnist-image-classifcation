#!/usr/bin/env python3

import numpy
import os
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from matplotlib import image

# Suppress tensorflow log messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GLOBALS
IMAGE_LENGTH = 28
IMAGE_WIDTH = 28
COLOR_CHANNELS = 1      # gray-scale
MAX_PIXEL_VALUE = 255.0 # 8-bit color depth with range 0-255


def main():
    """
        Demonstrate the prediction capability of the trained convolutional
        neural network.
    """
    model = load_model('my_model/')
    image_tensor = load_image_as_compatible_array('sample_image.png')

    # Convert prediction result to a human-readable format
    prediction = model.predict(image_tensor)
    prediction = tensorflow.math.argmax(prediction, axis=-1).numpy()

    display_classification_results('sample_image.png', prediction)


def load_image_as_compatible_array(file_name):
    """
        Convert the input image to a 3D array for compatibility with the
        convolutional neural network. 
    
    Args:
        file_name (string): name of the image file

    Returns:
        a 3D Numpy array representing the image
    """
    image = load_img(file_name, color_mode='grayscale', target_size=(IMAGE_LENGTH, IMAGE_WIDTH))
    image = img_to_array(image)
    image = image.reshape(1, IMAGE_LENGTH, IMAGE_WIDTH, COLOR_CHANNELS)
    image = image.astype('float32')
    image = image / MAX_PIXEL_VALUE

    return image


def display_classification_results(file_name, prediction):
    """
        Display the image and its classification result.

    Args:
        file_name (string): name of the image file
        prediction (string): the result from the convolutional neural network
    """
    prediction = str(prediction).strip('[]')
    formatted_prediction = 'Prediction: ' + prediction

    img = image.imread('sample_image.png')
    pyplot.imshow(img)
    pyplot.suptitle(formatted_prediction, fontsize=20)
    pyplot.show()


main()
