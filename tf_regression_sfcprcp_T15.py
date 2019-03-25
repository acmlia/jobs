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

import os
import time
import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import GaussianNoise
from keras.layers import GaussianDropout
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_yaml


print(tf.__version__)

# --------------------------
# DROP DATA OUTSIDE INTERVAL
# --------------------------
def keep_interval(keepfrom:0.0, keepto:1.0, dataframe, target_col:str):
    keepinterval = np.where((dataframe[target_col] >= keepfrom) &
                             (dataframe[target_col] <= keepto))
    result = dataframe.iloc[keepinterval]    
    return result

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#def tic():
#    global _start_time
#    _start_time = time.time()
#
#def tac():
#    t_sec = round(time.time() - _start_time)
#    (t_min, t_sec) = divmod(t_sec, 60)
#    (t_hour, t_min) = divmod(t_min, 60)
#    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))
    
# Fix random seed for reproducibility:
seed = 7
np.random.seed(seed)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Path, file and load DATAFRAME

file = 'yearly_br_underc1_0956.csv'
path = '/media/DATA/tmp/datasets/brazil/brazil_qgis/csv/'
fig_title = 'tf_regression_T15_undc1_0956_'
path_fig = '/media/DATA/tmp/git-repositories/jobs/tf_regression_figures/'

df_orig = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')

#path = '/home/david/DATA/'
#file = 'yrly_br_under_c1.csv'
#path_fig = '/home/david/DATA/'
#file = 'yrly_br_under_c1_over_c3c4.csv'
#file_name = os.path.splitext(file)[0]

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#### PRE - PROCESSING:
# Count the number of pixels by classs:

#colunas = list(df_orig.columns.values)
#df_orig = df_orig.loc[:,colunas]
#x, y = df_orig.loc[:,colunas], df_orig.loc[:,['CLASSE']]
#x_arr = np.asanyarray(x)
#y_arr = np.asanyarray(y)
#y_arr = np.ravel(y_arr)
#print('Original dataset shape %s' % Counter(y_arr))

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#dataset=df_orig.drop(columns=['lat','lon','sfccode', 'T2m','tcwv','skint',
#                              'cnvprcp','10V','10H','18V','18H','23V','36H',
#                              '89H','166H','10VH','18VH','SSI','delta_neg',
#                              'delta_pos','MPDI','MPDI_scaled','PCT10','PCT18',
#                              'TagRain', 'CLASSE'])

df_input=df_orig.loc[:,['10V','10H','18V','18H','36V','36H','89V','89H',
                    '166V','166H','183VH','sfccode','T2m','tcwv']]

colunas = ['10V','10H','18V','18H','36V','36H','89V','89H',
           '166V','166H','183VH','sfccode','T2m','tcwv']
scaler = StandardScaler()
normed_input = scaler.fit_transform(df_input) 
df_normed_input = pd.DataFrame(normed_input[:],
                                columns = colunas)
ancillary=df_normed_input.loc[:,['183VH','sfccode','T2m','tcwv']]
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------   
# Choosing the number of components:

TB1 = df_normed_input.loc[:,['10V','10H','18V','18H']]
TB2 = df_normed_input.loc[:,['36V','36H','89V','89H','166V','166H']]

#------------------------------------------------------------------------------
# Verifying the number of components that most contribute: 
pca = PCA()
pca1 = pca.fit(TB1)
plt.plot(np.cumsum(pca1.explained_variance_ratio_))
plt.xlabel('Number of components for TB1')
plt.ylabel('Cumulative explained variance');
plt.savefig("PCA_TB1.png")
#---  
pca_trans1 = PCA(n_components=2)
pca1=pca_trans1.fit(TB1)
TB1_transformed = pca_trans1.transform(TB1)
print("original shape:   ", TB1.shape)
print("transformed shape:", TB1_transformed.shape)
#------------------------------------------------------------------------------
pca = PCA()
pca2 = pca.fit(TB2)
plt.plot(np.cumsum(pca2.explained_variance_ratio_))
plt.xlabel('Number of components for TB2')
plt.ylabel('Cumulative explained variance');
plt.savefig("PCA_TB2.png")
#---
pca_trans2 = PCA(n_components=2)
pca2=pca_trans2.fit(TB2)
TB2_transformed = pca_trans2.transform(TB2)
print("original shape:   ", TB2.shape)
print("transformed shape:", TB2_transformed.shape)
#------------------------------------------------------------------------------
# JOIN THE TREATED VARIABLES IN ONE SINGLE DATASET AGAIN:

PCA1=pd.DataFrame()

PCA1 = pd.DataFrame(TB1_transformed[:],
                    columns = ['pca1_1','pca1_2'])
PCA2 = pd.DataFrame(TB2_transformed[:],
                    columns = ['pca2_1','pca2_2'])

dataset=PCA1.join(PCA2, how='right')
dataset=dataset.join(ancillary, how='right')
dataset=dataset.join(df_orig.loc[:,['sfcprcp']], how='right')
    
#df_orig['sfcprcp']=df_orig[['sfcprcp']].astype(int)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#dataset = keep_interval(0.5, 100.0, dataset, 'sfcprcp')
# scale the output between 0 and 1 for the colorbar
# y = minmax_scale(y_full)

# --------------------------------------
# Transform pd.DataFrame column to array
# --------------------------------------
#x_position = dataset.loc[:,['sfcprcp']]
#x_array = np.asanyarray(x_position)
#plt.plot(x_array)
#plt.show()    

threshold_rain =0.1
rain_pixels = np.where((dataset['sfcprcp'] >= threshold_rain))
dataset=dataset.iloc[rain_pixels]        

#         
# ----------------------------------------
# SUBSET BY SPECIFIC CLASS (UNDERSAMPLING)
n = 0.90
to_remove = np.random.choice(
        dataset.index,
        size=int(dataset.shape[0]*n),
        replace=False)
dataset = dataset.drop(to_remove)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Split the data into train and test
# Now split the dataset into a training set and a test set.
# We will use the test set in the final evaluation of our model.

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Inspect the data:
# Have a quick look at the joint distribution of a few pairs of columns from the training set.

colunas = list(dataset.columns.values)
#sns.pairplot(df_orig[colunas], diag_kind="kde")
#sns.pairplot(train_dataset[colunas], diag_kind="kde")
#sns.pairplot(test_dataset[colunas], diag_kind="kde")

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

#def norm(x):
#  return (x - train_stats['mean']) / train_stats['std']
#
#normed_train_data = norm(train_dataset)
#normed_test_data = norm(test_dataset)


#scaler=QuantileTransformer(output_distribution='uniform')
scaler = StandardScaler()
normed_train_data = scaler.fit_transform(train_dataset)
normed_test_data = scaler.fit_transform(test_dataset)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Build the model:

#def build_model():
#    model = keras.Sequential([
#            layers.Dense(24, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
#            layers.Dense(12, activation=tf.nn.relu),
#            layers.Dense(1)
#            ])
#    optimizer = tf.keras.optimizers.Adam(0.001)
#    model.compile(loss='mean_squared_error',
#                  optimizer=optimizer,
#                  metrics=['mean_absolute_error', 'mean_squared_error'])
#    return model

def build_model():
    model = Sequential()
    model.add(GaussianNoise(0.01, input_shape=[len(train_dataset.keys())] ))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',
              optimizer='adam',
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
print(history.history.keys())

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
    plt.ylim([0,ylim_max])
    plt.legend()
    #plt.show()
    fig_name = fig_title + "_error_per_epochs_history.png"
    plt.savefig(path_fig+fig_name)
  
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
    ylim_max = hist.val_mean_absolute_error.max()+10
    plt.ylim([0,ylim_max])
    #plt.ylim([0,5])
    plt.legend()
      
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$sfcprcp^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label = 'Val Error')
    ylim_max = hist.val_mean_squared_error.max()+10
    plt.ylim([0,ylim_max])
    #plt.ylim([0,30])
    plt.legend()
    #plt.show()
    fig_name = fig_title + "_error_per_epochs_EarlyStopping.png"
    plt.savefig(path_fig+fig_name)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

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
plt.plot([-100, 100], [-100, 100])
fig_name = fig_title + "_plot_scatter_y_test_vs_y_pred.png"
plt.savefig(path_fig+fig_name)
plt.clf()
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# It looks like our model predicts reasonably well. 
# Let's take a look at the error distribution.

error = test_predictions - y_test
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [sfcprcp]")
plt.ylabel("Count")
fig_name = fig_title + "_prediction_error.png"
plt.savefig(path_fig+fig_name)
plt.clf()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Saving a model
#if __name__ == '__main__':
#    _start_time = time.time()
#
#    tic()

# serialize model to YAML
model_yaml = model.to_yaml()
with open("tf_regression_T15.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("tf_regression_T15.h5")
print("Saved model to disk")
# 
## later...
# 
## load YAML and create model
#yaml_file = open('model.yaml', 'r')
#loaded_model_yaml = yaml_file.read()
#yaml_file.close()
#loaded_model = model_from_yaml(loaded_model_yaml)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
# 
## evaluate loaded model on test data
#loaded_model.compile(loss='mean_squared_error',
#              optimizer='adam',
#              metrics=['mean_absolute_error', 'mean_squared_error'])
#score = loaded_model.evaluate(normed_test_data, y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]))
#    
#    training_model = history
#    #grid_result = training_model.run_TuningRegressionPrecipitation()
#    joblib.dump(training_model, 'teste.pkl')
##    loaded_model = joblib.load('/media/DATA/tmp/git-repositories/jobs/model_trained_regression_precipitation_W1.pkl')
#    
#    
#    
#    tac()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------


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
