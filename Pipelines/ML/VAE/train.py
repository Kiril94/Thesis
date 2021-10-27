# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:10:11 2021

@author: klein
"""
# In[import]
from os.path import join
import os
import sys
import tensorflow as tf
script_dir = "D:/Thesis/ML/VAE"
if not script_dir in sys.path:
    sys.path.append(script_dir)
from preprocess import preprocess_mnist
from models import vae_base
import utils
# In[Load and preprocess images]
train_size = 60000
batch_size = 32
test_size = 10000
train_dataset, test_dataset, input_shape = preprocess_mnist.get_mnist_data(
    train_size, test_size, batch_size)
train_images, test_images, input_shape = preprocess_mnist.get_mnist_data(
    train_size, test_size, batch_size, return_batch_ds=(False))
# In[Define]
new_model = True
model_type = 'vae_base'
model_name = 'vae_base_0'
dataset = 'mnist'
comments = ''
model_dir = join(script_dir, 'trained_models', model_name)
description = f"{model_type}\n" + f"dataset: {dataset}\n"\
        + f"batch size = {batch_size}\n" + f"comments: {comments}"
  
# In[Train]
epochs = 10
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 10
num_examples_to_generate = 4
# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal([num_examples_to_generate,
                                                 latent_dim])
if model_type=='vae_base':
    model = vae_base.CVAE(latent_dim, input_shape)
    model.build((1,*input_shape))
else: 
    print('model not implemented')

# In[]
utils.train_model(model, train_dataset, test_dataset,
                  epochs, model_dir)




