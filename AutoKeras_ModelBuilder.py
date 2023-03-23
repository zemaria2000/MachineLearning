import pandas as pd
import numpy as np
import tensorflow as tf
from settings import DATA_DIR
from settings import TRAIN_SPLIT, PREVIOUS_STEPS, TRAINING, VARIABLES
import keras_tuner as kt
from keras_tuner import HyperParameters as hp
from sklearn.preprocessing import MinMaxScaler


# 1. CHOOSING WHICH OF PEDRO'S VARIABLES I WANT TO PREDICT
# Chossing a model to train based on the models available for now
var_to_predict = input(f'Choose a model from the following list\n {VARIABLES}: ')
if var_to_predict not in VARIABLES:
    print("Model spelt incorretly / model doesn't exist in the list given")
    var_to_predict = input(f"Please select a model from the list \n {VARIABLES}: ")


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. LOADING THE DATASET
# Loading the dataset with all the variables
input_df = pd.read_csv(f"{DATA_DIR}{var_to_predict}.csv")
input_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
# retrieving the variables in which we are interested
df = input_df[['Date', f'{var_to_predict}']]
# Converting the date
df['Date'] = pd.to_datetime(df['Date'])


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. TRAINING AND TEST SPLITS

# splitting data between training and testing
train_data_size = int(TRAIN_SPLIT * len(df)) 
train_data = df[:train_data_size]
test_data = df[train_data_size:len(df)]

#----------------------------------------------------------------------------------------------------------------------------
# NORMALIZING THE DATA
scaler = MinMaxScaler()
df[f'{var_to_predict}'] = scaler.fit_transform(np.array(df[f'{var_to_predict}']).reshape(-1, 1))

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

# Defining our train and test datasets based on the previous function
train_X, train_y = divide_time_series(x = train_data[f'{var_to_predict}'],
                                y = train_data[f'{var_to_predict}'],
                                prev_steps = PREVIOUS_STEPS)
test_X, test_y = divide_time_series(x = test_data[f'{var_to_predict}'],
                                y = test_data[f'{var_to_predict}'],
                                prev_steps = PREVIOUS_STEPS)


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. AUTOENCODER WITH HYPERPARAMETER TUNING

# First we need to define a function (with a parameter hp), that will iteratively be called by the tuner to build new models with different characteristics
# Information about the tuner - https://keras.io/keras_tuner/ 

def build_model(hp):

    # DEFINING A SERIES OF HYPERPARAMETERS TO BE TUNED
    # Choosing an activation function
    global hp_activation 
    hp_activation = hp.Choice('activation', values = ['relu', 'tanh', 'swish'])
    # Defining the number of layers
    global hp_layers 
    hp_layers = hp.Int('layers', min_value = 2, max_value = 10)
    # Defining the dropout rate for each layer
    global hp_dropout 
    hp_dropout = np.zeros(hp_layers)
    for i in range(hp_layers):
        hp_dropout[i] = hp.Float(f'dropout{i}', min_value = 1.05, max_value = 1.95)
    # Defining the different layer dimensions
    global hp_layer_dimensions
    hp_layer_dimensions = np.zeros(hp_layers)
    for i in range(hp_layers):
        if i == 0:
            hp_layer_dimensions[i] = int(PREVIOUS_STEPS/hp_dropout[i])
        else:
            hp_layer_dimensions[i] = int(hp_layer_dimensions[i-1]/hp_dropout[i])
    # defining a series of learning rates
    hp_learning_rate = hp.Choice('learning_rate', values = [1e-4, 1e-5, 1e-6, 1e-7])

    # BUILDING OUR AUTOENCODER MODEL
    # Instantiating the model
    model = tf.keras.Sequential()
    # generating the initializer
    initializer = tf.keras.initializers.GlorotNormal(seed = 13)
    # defining our input layer
    model.add(tf.keras.layers.Dense(PREVIOUS_STEPS, activation = hp_activation, kernel_initializer=initializer))
    # Building the encoder layers
    for i in range(hp_layers):
        if i == 0 or i == 1:
            model.add(tf.keras.layers.Dense(hp_layer_dimensions[i], activation=hp_activation, kernel_initializer=initializer))
            model.add(tf.keras.layers.Dropout(0.2))
        else:
            model.add(tf.keras.layers.Dense(hp_layer_dimensions[i], activation=hp_activation, kernel_initializer=initializer))
    # Building the decoder layers
    for i in range(hp_layers-1, -1, -1):
        if i == 0 or i == 1:
            model.add(tf.keras.layers.Dropout(0.2))
            model.add(tf.keras.layers.Dense(hp_layer_dimensions[i], activation=hp_activation, kernel_initializer=initializer))
        else:
            model.add(tf.keras.layers.Dense(hp_layer_dimensions[i], activation=hp_activation, kernel_initializer=initializer))
    # Building the output layer
    model.add(tf.keras.layers.Dense(PREVIOUS_STEPS, activation=hp_activation, kernel_initializer=initializer))
    
    # compiling our model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss = 'mean_squared_error',
                  metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    return model


# After defining the tuner function, we have to instantiate our hyperparameter tuner - we have 2 different options in here (there are many more - https://keras.io/api/keras_tuner/tuners/)
# tuner = kt.Hyperband(build_model,
#                      objective = 'val_loss',
#                      max_epochs = 20,
#                      factor = 3, 
#                      directory = 'AutoML_Experiments',
#                      project_name = 'Project',
#                     #  overwrite = True
#                      )
tuner = kt.BayesianOptimization(build_model,
                     objective = 'val_loss',
                     max_trials = 100, 
                     directory = 'AutoML_Experiments',
                     project_name = 'Project',
                     overwrite = True
                     )

# Defining a callback that stops the search if the results aren't improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0.0001,
    patience = 20,
    verbose = 1, 
    mode = 'min',
    restore_best_weights = True)
# Defining a callback that saves our model
cp = tf.keras.callbacks.ModelCheckpoint(filepath = f"models/{var_to_predict}.h5",
                                mode = 'min', monitor = 'val_loss', verbose = 2 , save_best_only = True)
# Initializing the tuner search - that will basically iterate over a certain number of different combinations (defined in the tuner above)
tuner.search(train_X, train_y, epochs = 20, validation_split = 0.1, callbacks = [early_stop])

# Printing a summary with the results obtained during the tuning process
tuner.results_summary()


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------
# 5. RETRIEVING THE BEST MODEL AND FITTING IT TO OUR DATA
# getting the best hyper parameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# CREATING THE RESPECTIVE MODELmodel
model = tuner.hypermodel.build(best_hps)
# fitting/training the final model
history = model.fit(train_X, train_y, epochs = TRAINING['EPOCHS'], validation_split = 0.15, callbacks = [early_stop, cp]).history
# summary with the model's  features
model.summary()


