# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 08:32:52 2021

@author: klein
"""
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def generate_and_save_images(model, epoch, test_sample, model_dir):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
  image_dir = os.path.join(model_dir, 'images')  
  if not os.path.exists(image_dir):
      os.makedirs(image_dir)
      
  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig(f"{image_dir}/image_at_epoch_{epoch:04d}.png")
  plt.show()
  
def plot_latent_images(model, n, digit_size=28):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  image_width = digit_size*n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
      z = np.array([[xi, yi]])
      x_decoded = model.sample(z)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit.numpy()

  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  plt.show()