
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from settings import CORR_GROUP, DATA_DIR
import math
from FixedParameters import TRAIN_SPLIT, PREVIOUS_STEPS, TRAINING


# Chossing a model to train based on the models available for now
model = input(f'Choose a model from the following list\n {CORR_GROUP.keys()}: ')
if model not in CORR_GROUP.keys():
    print("Model spelt incorretly / model doesn't exist in the list given")
    model = input(f"Please select a model from the list \n {CORR_GROUP.keys()}: ")

# --------------------------------- BUILDING OUR AUTOENCODER ----------------------------- #

# Setting all the hyperparameters (explained in my notebook, 03/03)
nb_epoch = TRAINING['EPOCHS']       #default: 50
batch_size = TRAINING['BATCH_SIZE']     #default: 32
input_dim = PREVIOUS_STEPS #num of columns, default = train_X.shape[1]
encoding_dim = input_dim / 2   #default: 14
hidden_dim_1 = encoding_dim / 2 #
hidden_dim_2 = hidden_dim_1 / 2  #default: 4
learning_rate = 1e-7    # default = 1e-7

# Creating the Autoencoder
# Input Layer
input_layer = tf.keras.layers.Input(shape=(input_dim, ))  

# Encoder
encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh", activity_regularizer=tf.keras.regularizers.l2(learning_rate))(input_layer)                       
encoder = tf.keras.layers.Dropout(0.2)(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
encoder = tf.keras.layers.Dense(hidden_dim_2, activation=tf.nn.leaky_relu)(encoder)

# Decoder
decoder = tf.keras.layers.Dense(hidden_dim_1, activation='relu')(encoder)
decoder = tf.keras.layers.Dropout(0.2)(decoder)
decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(decoder)
decoder = tf.keras.layers.Dense(input_dim, activation='tanh')(decoder)

#Autoencoder
autoencoder = tf.keras.Model(inputs = input_layer, outputs = decoder)
autoencoder.summary()

# Some callbacks for checkpoints and early stopping
cp = tf.keras.callbacks.ModelCheckpoint(filepath = "autoencoder.h5",
                                mode = 'min', monitor = 'val_loss', verbose =2 , save_best_only = True)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0.0001,
    patience = 20,
    verbose = 1, 
    mode = 'min',
    restore_best_weights = True)

# Compiling the autoencoder
autoencoder.compile(metrics = ['accuracy'],
                    loss = 'mean_squared_error',
                    optimizer = 'adam')

# ---------------------------------- LOADING THE DATASET ------------------------------------ #
# Loading the dataset with all the variables
input_df = pd.read_csv(f"{DATA_DIR}{model}.csv")
input_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
# retrieving the variables in which we are interested
df = input_df[['Date', f'{model}']]
# Converting the date
df['Date'] = pd.to_datetime(df['Date'])



# -------------------------------- TRAINING vs TEST SPLITS -------------------------------- #

# splitting data between training and testing
train_data_size = int(0.8 * len(df)) 
train_data = df[:train_data_size]
test_data = df[train_data_size:len(df)]


# -------------------------------- CLEARING ANOMALIES -------------------------------- #

# Eliminating anomalies from our training dataset (our model must learn the normal behaviour)
# I considered an anomaly to be a value below 0.1
anomaly_filter = (train_data[f'{model}'] <= 0.1)
normal_train_data = train_data.loc[~anomaly_filter]

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

# Defining our train, test and test_training splits
train_X, train_y = divide_time_series(x = normal_train_data[f'{model}'],
                                y = normal_train_data[f'{model}'],
                                prev_steps = PREVIOUS_STEPS)
test_X, test_y = divide_time_series(x = test_data[f'{model}'],
                                y = test_data[f'{model}'],
                                prev_steps = PREVIOUS_STEPS)


# ---------------------------------- TRAINING THE AUTOENCODER --------------------------------- #
history = autoencoder.fit(x = train_X,
                            y = train_y,
                            epochs = nb_epoch,
                            batch_size = batch_size,
                            shuffle = True,
                            # validation_split = 0.1,
                            validation_data = (test_X, test_y),
                            verbose = 1,
                            callbacks = [cp, early_stop]).history

# Plot training and testing loss
plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.savefig(f"{FIGURES_DIR}{model}_Loss_Epoch.png")
plt.show()

# ----------------------------- PREDICTING ON TEST_DATA -------------------------- #
# making a prediction of the X variable
test_predict = autoencoder.predict(test_X)
# Generating a vector with the y predictions
test_predict_y = []
for i in range(len(test_predict)):
    test_predict_y.append(test_predict[i][PREVIOUS_STEPS-1])


# ------------------------------ ANALYSING OUR DATA ------------------------------ #

# Some important indicators
mse, rmse = [], []
for i in range(len(test_y)):
    mse.append(np.square(np.subtract(test_y[i],test_predict_y[i])).mean())
# mse = mean_squared_error(test_y, np.array(test_predict_y))
# rmse = math.sqrt(mse)
# r2 = r2_score(test_y, test_predict_y)
# pinball = d2_pinball_score(test_y, test_predict_y)
# list = {'model': f'{model}', 'MSE': [mse], 'RMSE': [rmse]}
# print(list)

# plt.hist(mse, bins = 30)

# --------------------------- ANOMALY DETECTION --------------------------- #

# value of mse from which we consider an anomaly - completely arbitrary for now
anomaly_threshold = 0.1

# DataFrame with the anomalies labeled
anomalies_df = pd.DataFrame({'Real y values': test_y, 'Predicted y values': test_predict_y, 'MSE': mse})

# filtering the anomalies
anomaly_filter = (anomalies_df['MSE'] > anomaly_threshold)
anomalies_df['Anomaly'] = anomaly_filter

# seperating the anomalies
anomalies = anomalies_df.loc[anomalies_df['Anomaly'] == True]

# plotting the results and highlighting the anomalies
plt.subplot(1, 2, 1)
plt.plot(test_y, linewidth = 0.5)
plt.plot(test_predict_y, linewidth = 0.5)
plt.title('Predictions VS Real Values')
plt.legend(['Real Values', 'Predicted Values'])
# plotting our data
plt.subplot(1, 2, 2)
plt.plot(test_y, linewidth = 0.5)
plt.scatter(anomalies.index, anomalies['Real y values'], facecolors = None, edgecolors = 'r')
plt.title('Anomalies')
# plt.savefig(f"{FIGURES_DIR}{model}_RealVSPrediction.png")
plt.show()