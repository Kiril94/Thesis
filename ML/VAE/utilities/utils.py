# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 16:48:49 2021

@author: klein
"""
from IPython import display
import tensorflow as tf
from utilities import losses, vis
import time
import os

@tf.function
def train_step(model, x, optimizer):
    """Executes one training step and returns the loss.
    
    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
      loss = losses.compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
def train_model(model, train_dataset, test_dataset, 
                epochs, model_dir, batch_size=32,
                optimizer=tf.keras.optimizers.Adam(1e-4),
                num_examples_to_generate=16):
    
    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
      test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    
    for epoch in range(1, epochs + 1):
      start_time = time.time()
      for train_x in train_dataset:
          train_step(model, train_x, optimizer)
      end_time = time.time()
      loss = tf.keras.metrics.Mean()
      for test_x in test_dataset:
        loss(losses.compute_loss(model, test_x))
      elbo = -loss.result()
      display.clear_output(wait=False)
      print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
            .format(epoch, elbo, end_time - start_time))  
      vis.generate_and_save_images(model, epoch, test_sample, model_dir)
      model.save(os.path.join(model_dir, 'model'))