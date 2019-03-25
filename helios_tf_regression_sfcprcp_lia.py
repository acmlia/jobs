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

path = '/media/DATA/tmp/datasets/brazil/brazil_qgis/csv/'
#file = 'yrly_br_under_c1_over_c3c4.csv'
file = 'yrly_br_under_c1.csv'
file_name = os.path.splitext(file)[0]


dataset_orig = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')
dataset_orig=dataset_orig.drop(columns=['CLASSE'])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Split the data into train and test
# Now split the dataset into a training set and a test set.
# We will use the test set in the final evaluation of our model.

train_dataset = dataset_orig.sample(frac=0.8,random_state=0)
test_dataset = dataset_orig.drop(train_dataset.index)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Inspect the data:
# Have a quick look at the joint distribution of a few pairs of columns from the training set.

colunas = list(dataset_orig.columns.values)
sns.pairplot(dataset_orig[[colunas]], diag_kind="kde")
sns.pairplot(train_dataset[[colunas]], diag_kind="kde")
sns.pairplot(test_dataset[[colunas]], diag_kind="kde")

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Also look at the overall statistics:

train_stats = train_dataset.describe()
train_stats.pop("sfcprcp")
train_stats = train_stats.transpose()
train_stats
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Split features from labels:
# Separate the target value, or "label", from the features. This label is the value that you will train the model to predict.

y_train = train_dataset.pop('sfcprcp')
y_test = test_dataset.pop('sfcprcp')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Inspect the model:
# Use the .summary method to print a simple description of the model

model.summary()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Now try out the model. 
# Take a batch of 10 examples from the training
# data and call model.predict on it.

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# It seems to be working, and it produces a result 
# of the expected shape and type.

# Train the model:
# Train the model for 1000 epochs, and record the training
# and validation accuracy in the history object.

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Visualize the model's training progress using the stats
# stored in the history object.

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [sfcprcp]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  ylim_max = hist.val_mean_absolute_error.max()+10
  plt.ylim([0,ylim_max])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$scfprcp^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  ylim_max = hist.val_mean_squared_error.max()+10
  plt.ylim([0,20])
  plt.legend()
  #plt.show()
  fig_name = os.path.splitext(file)[0] + "error_per_epochs_history.png"
  plt.savefig(fig_name)
  
plot_history(history)

  
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Ploting again, but with the EarlyStopping apllied:

def plot_history_EarlyStopping(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [sfcprcp]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$sfcprcp^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,30])
  plt.legend()
  #plt.show()
  fig_name = os.path.splitext(file)[0] + "error_per_epochs_EarlyStopping.png"
  plt.savefig(fig_name)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

history = model.fit(normed_train_data, y_train, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history_EarlyStopping(history)

# The graph shows that on the validation set, the average error 
# is usually around +/- 2 MPG. Is this good? 
# We'll leave that decision up to you.
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Let's see how well the model generalizes by using 
# the test set, which we did not use when training the model. 
# This tells us how well we can expect the model to predict 
# when we use it in the real world.

loss, mae, mse = model.evaluate(normed_test_data, y_test, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} sfcprcp".format(mae))

# Make predictions
# Finally, predict SFCPRCP values using data in the testing set:

test_predictions = model.predict(normed_test_data).flatten()

plt.figure()
plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [sfcprcp]')
plt.ylabel('Predictions [sfcprcp]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
fig_name = os.path.splitext(file)[0] + "_plot_scatter_y_test_vs_y_pred.png"
plt.savefig(fig_name)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# It looks like our model predicts reasonably well. 
# Let's take a look at the error distribution.

error = test_predictions - y_test
fig = plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [sfcprcp]")
_ = plt.ylabel("Count")
fig_name = os.path.splitext(file)[0] + "_prediction_error.png"
plt.savefig(path+fig_name)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------


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
