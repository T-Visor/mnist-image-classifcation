#!/usr/bin/env python3

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from matplotlib import pyplot
from matplotlib import image

# GLOBALS
IMAGE_LENGTH = 28
IMAGE_WIDTH = 28
COLOR_CHANNELS = 1      # gray-scale
MAX_PIXEL_VALUE = 255.0 # 8-bit color depth with range 0-255




def run_example():
    model = load_model('my_model/')

    image_tensor = load_image_as_tensor('sample_image.png')

    digit = model.predict_classes(image_tensor)

    display_classification_results('sample_image.png', digit)




def load_image_as_tensor(filename):
    image = load_img(filename, grayscale=True, target_size=(IMAGE_LENGTH, IMAGE_WIDTH))
    image = img_to_array(image)
    image = image.reshape(1, IMAGE_LENGTH, IMAGE_WIDTH, COLOR_CHANNELS)

    image = image.astype('float32')
    image = image / MAX_PIXEL_VALUE

    return image




def display_classification_results(file_name, prediction):
    prediction = str(prediction).strip('[]')
    formatted_prediction = 'Prediction: ' + prediction

    img = image.imread('sample_image.png')
    pyplot.imshow(img)
    pyplot.suptitle(formatted_prediction, fontsize=20)
    pyplot.show()




run_example()
