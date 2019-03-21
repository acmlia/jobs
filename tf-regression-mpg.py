#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:31:04 2019

@author: dvdgmf
"""
# https://www.tensorflow.org/tutorials/keras/basic_regression
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Define the Dataframe and input/output variables:

#path = '/home/david/DATA/'
path = '/media/DATA/tmp/datasets/brazil/brazil_qgis/csv/'
file = 'yrly_br_under_c1_over_c3c4_10pct.csv'
dataset = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')
#x, y= df.loc[:,['36V', '89V', '166V', '186V', '190V', '36VH', '89VH',
#                        '166VH', '183VH', 'PCT36', 'PCT89']], df.loc[:,['sfcprcp']]

# Split the data into train and test
# Now split the dataset into a training set and a test set.
# We will use the test set in the final evaluation of our model.

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data
# Have a quick look at the joint distribution of a few pairs of columns from the training set.

sns.pairplot(train_dataset[["sfcprcp", "36V", "89V","PCT89"]], diag_kind="kde")

# Also look at the overall statistics:

train_stats = train_dataset.describe()
train_stats.pop("sfcprcp")
train_stats = train_stats.transpose()
train_stats

# Split features from labels:
# Separate the target value, or "label", from the features. This label is the value that you will train the model to predict.

y_train = train_dataset.pop('sfcprcp')
y_test = test_dataset.pop('sfcprcp')

# Normalize the data:
# Look again at the train_stats block above and note how different the ranges 
# of each feature are.

# It is good practice to normalize features that use different scales and ranges.
# Although the model might converge without feature normalization, it makes 
# training more difficult, and it makes the resulting model 
# dependent on the choice of units used in the input. 

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Build the model:

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

# Inspect the model:
# Use the .summary method to print a simple description of the model

model.summary()

# Now try out the model. 
# Take a batch of 10 examples from the training
# data and call model.predict on it.

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

# It seems to be working, and it produces a result 
# of the expected shape and type.

# Train the model:
# Train the model for 1000 epochs, and record the training
# and validation accuracy in the history object.

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, y_train,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

# Visualize the model's training progress using the stats
# stored in the history object.

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(history)

# This graph shows little improvement, or even degradation 
# in the validation error after about 100 epochs. 
# Let's update the model.fit call to automatically stop 
# training when the validation score doesn't improve.
# We'll use an EarlyStopping callback that tests a training 
# condition for every epoch. If a set amount of epochs 
# elapses without showing improvement, then automatically stop the training.

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

# The graph shows that on the validation set, the average error 
# is usually around +/- 2 MPG. Is this good? 
# We'll leave that decision up to you.

# Let's see how well the model generalizes by using 
# the test set, which we did not use when training the model. 
# This tells us how well we can expect the model to predict 
# when we use it in the real world.

loss, mae, mse = model.evaluate(normed_test_data, y_test, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# Make predictions
# Finally, predict MPG values using data in the testing set:

test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])

# It looks like our model predicts reasonably well. 
# Let's take a look at the error distribution.

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

# It's not quite gaussian, but we might expect that because 
# the number of samples is very small.

# Conclusion:

# This notebook introduced a few techniques to handle a regression problem.

# >> Mean Squared Error (MSE) is a common loss function used for regression problems (different loss functions are used for classification problems).

# >> Similarly, evaluation metrics used for regression differ from classification. A common regression metric is Mean Absolute Error (MAE).

# >> When numeric input data features have values with different ranges, each feature should be scaled independently to the same range.

# >> If there is not much training data, one technique is to prefer a small network with few hidden layers to avoid overfitting.

# >> Early stopping is a useful technique to prevent overfitting.

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
