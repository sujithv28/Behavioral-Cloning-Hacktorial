# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import argparse
import json
import utils

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils import *


kSEED = 5
SIDE_STEERING_CONSTANT = 0.25
NUM_BINS = 23


def batch_generator(images, angles, augment_data=True, batch_size=64):
    """
    Keras Batch Generator to create a generator of training examples for model.

    :param images: Training image data.
    :param angles: Angle data for images.
    :param batch_size: Batch size of each training run.
    :param augment_data: If the data should be augmented.

    :return: A batch generator.
    """
    batch_images = []
    batch_angles = []
    sample_count = 0

    while True:
        # Shuffle indices to minimize overfitting.
        for i in np.random.permutation(images.shape[0]):

            # Image (1) -> Center image and steering angle.
            center_path = images.iloc[i]['Center Image']
            left_path = images.iloc[i]['Left Image']
            right_path = images.iloc[i]['Right Image']
            angle = float(angles.iloc[i])

            center_image = utils.load_image_and_preprocess(center_path)
            batch_images.append(center_image)
            batch_angles.append(angle)

            sample_count += 1

            # Add augmentation if needed. We do this because our model only runs on
            # our center camera feed and we dont want to modify anything other than
            # the cropping and normalizing for our validation dataset since this should
            # work on raw data.
            if augment_data:
                # Image (2) -> Flip the image and invert angle respectively.
                flipped_image = utils.load_image_and_preprocess(center_path, flip=True, tint=False)
                flipped_angle = -1.0 * angle
                batch_images.append(flipped_image)
                batch_angles.append(flipped_angle)

                # Image (3) -> Tint the center image to random brightness.
                tint_image = utils.load_image_and_preprocess(center_path, flip=False, tint=True)
                tint_angle = angle
                batch_images.append(tint_image)
                batch_angles.append(tint_angle)

                # Image (4) -> Jitter the center image to make it seem like
                # different position on the road.
                jittered_image, jitter_angle = utils.jitter_image(center_path, angle)
                batch_images.append(jittered_image)
                batch_angles.append(jitter_angle)

                # Image (5) -> Load the left image and add steering constant to
                # compensate.
                left_image = utils.load_image_and_preprocess(left_path)
                left_angle = min(1.0, angle + SIDE_STEERING_CONSTANT)
                batch_images.append(left_image)
                batch_angles.append(left_angle)

                # Image (6) -> Load the right image and subtract steering
                # constant to compensate.
                right_image = utils.load_image_and_preprocess(right_path)
                right_angle = max(-1.0, angle - SIDE_STEERING_CONSTANT)
                batch_images.append(right_image)
                batch_angles.append(right_angle)

            # If we have processed batch_size number samples or this is the last batch
            # of the epoch, then we submit the batch. Since we augment the data there is a chance
            # we have more than the number of batch_size elements in each
            # batch.
            if ((sample_count %
                 batch_size == 0) or (sample_count %
                                      len(images) == 0)):
                yield np.array(batch_images), np.array(batch_angles)
                # Reset batch
                batch_images = []
                batch_angles = []


def create_model(lr=1e-3, activation='relu', nb_epoch=15):
    """
    End-to-End Learning Model for Self-Driving based off of Nvidia.

    :param lr: Model learning rate.
    :param activation: Activation function to use for each layer.
    :param nb_epoch: Number of epochs to train for.

    :return: A convolutional neural network.
    """
    model = Sequential()
    # Lambda layer normalizes pixel values between 0 and 1
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    # Convolutional layer (1)
    model.add(Conv2D(24, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    # Convolutional layer (2)
    model.add(Conv2D(36, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    # Convolutional layer (3)
    model.add(Conv2D(48, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    # Convolutional layer (4)
    model.add(Conv2D(64, (3,3), padding='same', activation=activation, strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    # Convolutional layer (5)
    model.add(Conv2D(64, (3,3), padding='same', activation=activation, strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    # Flatten Layer
    model.add(Flatten())
    # Dense Layer (1)
    model.add(Dense(1164, activation=activation))
    # Dense layer (2)
    model.add(Dense(100, activation=activation))
    # Dense layer (3)
    model.add(Dense(50, activation=activation))
    # Dense layer (4)
    model.add(Dense(10, activation=activation))
    # Dense layer (5)
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer=Adam(lr=lr, decay=lr / nb_epoch), loss='mse')
    return model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lr", help="Initial learning rate",
                           type=float, default=1e-3, required=False)
    argparser.add_argument("--nb_epoch", help="Number of epochs to train for",
                           type=int, default=15, required=False)
    argparser.add_argument("--activation", help="Activation function to use",
                           type=str, default='relu', required=False)
    args = argparser.parse_args()

    if not os.path.exists('models'):
        os.makedirs('models/')

    file_name = 'driving_log.csv'
    columns = [
        'Center Image',
        'Left Image',
        'Right Image',
        'Steering Angle',
        'Throttle',
        'Break',
        'Speed']

    print('[INFO] Loading Data.')
    images, angles = utils.load_data(file_name, columns)

    print('[INFO] Creating Training and Testing Data.')
    X_train, X_val, y_train, y_val = train_test_split(
        images, angles, test_size=0.15, random_state=kSEED)

    print('[INFO] Preprocessing Images and Data Augmentation.')
    generator_train = batch_generator(X_train, y_train, augment_data=True)
    generator_val = batch_generator(X_val, y_val, augment_data=False)

    print('[INFO] Creating Model.')
    model = create_model(args.lr, args.activation, args.nb_epoch)
    checkpoint = ModelCheckpoint(
        'models/model-{epoch:03d}.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='auto')

    print('[INFO] Training Model.')
    model.fit_generator(
        generator_train,
        steps_per_epoch=6 * len(X_train),
        epochs=args.nb_epoch,
        validation_data=generator_val,
        callbacks=[checkpoint],
        validation_steps=len(X_val),
        verbose=1)

    print('[INFO] Saving Model')
    model.save_weights('models/model.h5', True)
    with open('models/model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)
