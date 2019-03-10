# ------------------------------------------------------------------------------
# Loading the libraries to be used: import numpy
import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
from keras.constraints import maxnorm
# ------------------------------------------------------------------------------

def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


class TuningDropoutRegularization:

    def create_model(self, dropout_rate=0.0, weight_constraint=0):

        # create model
        model = Sequential()
        model.add(Dense(5, input_dim=9, kernel_initializer='uniform', activation='tanh',
                        kernel_constraint=maxnorm(weight_constraint)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_initializer='uniform', activation='linear'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def run_TuningDropoutRegularization(self):

        # Fix random seed for reproducibility:
        seed = 7
        np.random.seed(seed)

        # Load dataset:
        path = '/home/david/DATA/'
        file = 'yearly_br_rain_var2d_OK.csv'
        df = pd.read_csv(os.path.join(path, file), sep=',', decimal='.')

        # Split into input (X) and output (Y) variables:
        df2 = df[['36V', '36H', '89V', '89H', '166V', '166H', '190V', 'PCT36', 'PCT89']]
        # x = df2.reindex(columns=cols)
        x = df2[['36V', '36H', '89V', '89H', '166V', '166H', '190V', 'PCT36', 'PCT89']]
        y = df[['sfcprcp']]

        # Scaling the input paramaters:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Split the dataset in test and train samples:
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=101)

        # Create the instance for KerasRegressor:
        model = KerasRegressor(build_fn=self.create_model, epochs=100, batch_size=10, verbose=0)

        # define the grid search parameters:

        weight_constraint = [1, 2, 3, 4, 5]
        dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
        grid_result = grid.fit(x_train, y_train)

        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Saving a model
if __name__ == '__main__':
    _start_time = time.time()

    tic()

    training_model = TuningDropoutRegularization()
    grid_result = training_model.run_TuningDropoutRegularization()
    joblib.dump(grid_result, 'model_trained_dropout_regularization_A.pkl')

    tac()

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

