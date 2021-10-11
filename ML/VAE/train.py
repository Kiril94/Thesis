# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 15:10:11 2021

@author: klein
"""
# In[]

import tensorflow as tf
import tensorflow.keras as k
from models import vae_base
import os
# In[]

base_dir = os.dir(__file__)
print(base_dir)
# %%
inputs = k.Input(shape=(3,))
x = k.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = k.layers.Dense(5, activation=tf.nn.softmax)(x)
model = k.Model(inputs=inputs, outputs=outputs)
# In[]
x_train = tf.random.uniform((100,3))
y_train = tf.random.uniform((100,5))
model.compile(optimizer=k.optimizers.RMSprop(learning_rate=1e-3),
              loss=k.losses.CategoricalCrossentropy())
model.fit(x_train, y_train,
          batch_size=32, epochs=10)              