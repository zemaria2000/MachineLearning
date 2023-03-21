# %%

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, d2_pinball_score
from settings import CORR_GROUP, DATA_DIR
import math
from FixedParameters import TRAIN_SPLIT, PREVIOUS_STEPS, TRAINING, VARIABLES
import keras_tuner as kt
from keras_tuner import HyperParameters as hp
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

# ---------------------------------- LOADING THE DATASET ------------------------------------ #

# Loading the dataset with all the variables
input_df = pd.read_csv("gold_price_data.csv")
# retrieving the variables in which we are interested
df = input_df[['Date', 'Value']]
# Converting the date
df['Date'] = pd.to_datetime(df['Date'])


# Normalizing the data
# scaler = MinMaxScaler()
scaler = RobustScaler()
# scaler = PowerTransformer()
scaler = scaler.fit(df[['Value']])

df['Value'] = scaler.transform(df[['Value']])

# -------------------------------- TRAINING vs TEST SPLITS -------------------------------- #

# splitting data between training and testing
train_data_size = int(TRAIN_SPLIT * len(df)) 
train_data = df[:train_data_size]
test_data = df[train_data_size:len(df)]

# %%
# Function that divides our dataset according to the previous_steps number
# watch this - https://www.youtube.com/watch?v=6S2v7G-OupA&t=888s&ab_channel=DigitalSreeni
# this function will take our data and for each 'previous_steps' timestamps, the next one is saved in the y_values
def divide_time_series(x, y, prev_steps):
    x_values = []
    y_values = []

    for i in range(len(x)-prev_steps):
        x_values.append(x.iloc[i:(i+prev_steps)].values)
        y_values.append(y.iloc[i+prev_steps])

    return np.array(x_values), np.array(y_values)

# Defining our train and test splits
train_X, train_y = divide_time_series(x = train_data['Value'],
                                y = train_data['Value'],
                                prev_steps = PREVIOUS_STEPS)
test_X, test_y = divide_time_series(x = test_data['Value'],
                                y = test_data['Value'],
                                prev_steps = PREVIOUS_STEPS)


# %%
# --------------------------------- BUILDING OUR AUTOENCODER WITH KERAS-TUNER ----------------------------- #

# 1. Function that the tuner executes everytime it iterates a new model (with different hyperparameters)
def build_model(hp):

    # generating the model
    model = tf.keras.Sequential()
    
    # generating the initializer
    initializer = tf.keras.initializers.GlorotNormal(seed = 13)

    # Choosing an activation function
    global hp_activation 
    hp_activation = hp.Choice('activation', values = ['relu', 'tanh', 'swish'])

    # defining our input layer
    model.add(tf.keras.layers.Input(shape = (PREVIOUS_STEPS, )))
    # model.add(tf.keras.layers.Dense(PREVIOUS_STEPS, activation = hp_activation, kernel_initializer=initializer))

    # Defining the number of layers
    global hp_layers 
    hp_layers = hp.Int('layers', min_value = 3, max_value = 10)

    # Defining the dropout rate for each layer
    global hp_dropout 
    hp_dropout = np.zeros(hp_layers)
    for i in range(hp_layers):
        hp_dropout[i] = hp.Float(f'dropout{i}', min_value = 1.2, max_value = 1.7)

    # Defining the different layer dimensions
    global hp_layer_dimensions
    hp_layer_dimensions = np.zeros(hp_layers)
    for i in range(hp_layers):
        if i == 0:
            hp_layer_dimensions[i] = int(PREVIOUS_STEPS/hp_dropout[i])
        else:
            hp_layer_dimensions[i] = int(hp_layer_dimensions[i-1]/hp_dropout[i])

    # Building our encoder
    for i in range(hp_layers):
        if i == 0 or i == 1:
            model.add(tf.keras.layers.Dense(hp_layer_dimensions[i], activation=hp_activation, kernel_initializer=initializer))
            model.add(tf.keras.layers.Dropout(0.2))
        else:
            model.add(tf.keras.layers.Dense(hp_layer_dimensions[i], activation=hp_activation, kernel_initializer=initializer))

    # Building our decoder
    for i in range(hp_layers-1, -1, -1):
        if i == 0 or i == 1:
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(hp_layer_dimensions[i], activation=hp_activation, kernel_initializer=initializer))
        else:
            model.add(tf.keras.layers.Dense(hp_layer_dimensions[i], activation=hp_activation, kernel_initializer=initializer))

    # output layer
    model.add(tf.keras.layers.Dense(PREVIOUS_STEPS, activation=hp_activation, kernel_initializer=initializer))
    
    # defining a series of learning rates
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-4, 1e-6])
    # compiling our model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss = 'mean_squared_error',
                  metrics = ['accuracy'])
    
    return model


# 2. Using keras_tuner - https://www.youtube.com/watch?v=6Nf1x7qThR8&ab_channel=GregHogg
tuner = kt.Hyperband(build_model,
                     objective = 'val_loss',
                     max_epochs = 10,
                     factor = 3, 
                     directory = 'AutoML_Experiments_Other_Datasets',
                     project_name = 'Project',
                     overwrite = True
)
# tuner = kt.BayesianOptimization(build_model,
#                      objective = 'val_loss',
#                      max_trials = 100, 
#                      directory = 'AutoML_Experiments',
#                      project_name = 'Project')

# callback that stops the search if the results aren't improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0.0001,
    patience = 20,
    verbose = 1, 
    mode = 'min',
    restore_best_weights = True)

# equivalent to model.fit - it will do the model training for each of it's iterations in order to find the best fitting model created
# with the tuner above
tuner.search(train_X, train_y, epochs = 10, validation_split = 0.1, callbacks = [early_stop])

# summary with the best results
tuner.results_summary()

# 3. Final fitting and training of the best model
# getting the besst hyper parameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# best model
model = tuner.hypermodel.build(best_hps)
# fitting/training the final model
history = model.fit(train_X, train_y, epochs = 50, validation_data = (test_X, test_y), callbacks = [early_stop]).history
# summary with the model's best features
model.summary()

# %%

# ----------------------------- PREDICTING ON TRAINING_DATA -------------------------- #
# making a prediction of the X variable
train_predict = model.predict(train_X)
# Generating a vector with the y predictions
train_predict_y = []
for i in range(len(train_predict)):
    train_predict_y.append(train_predict[i][PREVIOUS_STEPS-1])

# As our train_predict_y is a list, we'll turn our train_y in a list as well
train_predict_y = np.array(train_predict_y)

# ----------------------------- PREDICTING ON TEST_DATA -------------------------- #
# making a prediction of the X variable
test_predict = model.predict(test_X)
# Generating a vector with the y predictions
test_predict_y = []
for i in range(len(test_predict)):
    test_predict_y.append(test_predict[i][PREVIOUS_STEPS-1])

# As our test_predict_y is a list, we'll turn our test_y in a list as well
test_predict_y = np.array(test_predict_y)


# %%
# ------------------------------ ANALYSING OUR DATA (WHILE NORMALIZED) ------------------------------ #

# Some important indicators
mse = mean_squared_error(test_y, test_predict_y)
rmse = math.sqrt(mse)
r2 = r2_score(test_y, test_predict_y)
pinball = d2_pinball_score(test_y, test_predict_y)
list = {'model': f'{model}', 'MSE': [mse], 'RMSE': [rmse], 'R2': [r2], 'Pinball': [pinball]}
print(list)


# %%
# ----------------------------- INVERSE SCALING OUR DATA -------------------------- #
train_y, train_predict_y = scaler.inverse_transform(train_y.reshape(-1, 1)), scaler.inverse_transform(train_predict_y.reshape(-1, 1))
test_y, test_predict_y = scaler.inverse_transform(test_y.reshape(-1, 1)), scaler.inverse_transform(test_predict_y.reshape(-1, 1))


# %%

# plotting the differences between our test and training data 
plt.subplot(1, 2, 1)
plt.plot(train_y, linewidth = 0.5)
plt.plot(train_predict_y, linewidth = 0.5)
plt.title('Training Data')
plt.legend(['Real Values', 'Predicted Values'])

plt.subplot(1, 2, 2)
plt.plot(test_y, linewidth = 0.5)
plt.plot(test_predict_y, linewidth = 0.5)
plt.title('Testing Data')
plt.legend(['Real Values', 'Predicted Values'])
# plt.savefig(f"{FIGURES_DIR}{model}_RealVSPrediction.png")
plt.show()

# printing some of the final Autoencoder characteristics
print('Different Hyperparameters: \n')
print(f"Layer's dimensions: {hp_layer_dimensions}\n")
print(f"Drop between each layer: {hp_dropout}\n")


# %%
