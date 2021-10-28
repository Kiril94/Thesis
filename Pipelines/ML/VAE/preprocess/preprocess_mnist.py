# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 16:26:24 2021

@author: klein
"""

import numpy as np
import tensorflow as tf




def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

def get_mnist_data(train_size, test_size,
                     batch_size, return_batch_ds=True):
    """Returns tf ready train and test dataset as well as the shape"""
    
    (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()             
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images) 

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                     .shuffle(train_size).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                    .shuffle(test_size).batch(batch_size))
    if return_batch_ds:
        return train_dataset, test_dataset, test_images.shape[1:]
    else:
        return train_images, test_images, test_images.shape[1:]

