import json
import numpy as np
import pandas as pd
import cv2
import os
import scipy.misc

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, MaxPooling2D, Lambda
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from PIL import Image
from sklearn.model_selection import train_test_split

# Location of the training data from Udacity.
DATA_FILE = ''

# Load the training data into a pandas dataframe.
columns = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']

# LOAD DATA and SPLIT IT HERE:




def jitter_image():



    return image



def load_image_and_preprocess():
	# PREPROCSES IMAGE



    return image



def batch_generator(images, steering_angles, batch_size=64, augment_data=True):
    # Create an array of sample indices.
    batch_images = []
    batch_steering_angles = []
    sample_count = 0
    SIDE_STEERING_CONSTANT = 0.25
    indices = np.arange(len(images))

    while True:
        # Shuffle indices to minimize overfitting. Common procedure.
        np.random.shuffle(indices)
        for i in indices:
        	# Do Processing of image HERE:





            # Increment the number of samples.
            sample_count += 1

            # If we have processed batch_size number samples or this is the last batch
            # of the epoch, then we submit the batch. Since we augment the data there is a chance
            # we have more than the number of batch_size elements in each batch.
            if (sample_count % batch_size) == 0 or (sample_count % len(images)) == 0:
                yield np.array(batch_images), np.array(batch_steering_angles)
                # Reset
                batch_images = []
                batch_steering_angles = []



def create_model():
	# CREATE KERAS MODEL





	return None



# Create model and data and pass it in here.
model = create_model()



# Code to save model
model.save_weights('model.h5', True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)