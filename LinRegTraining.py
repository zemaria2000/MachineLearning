# In this file I'm just going to train all the models through a linear regression, in order to see if they are better thab the AutoML ones
# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from settings import DATA_DIR
from settings import TRAIN_SPLIT, PREVIOUS_STEPS, TRAINING, VARIABLES, LIN_REG_VARS
from sklearn.linear_model import LinearRegression
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import joblib
from sklearn.preprocessing import MinMaxScaler

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


for var_to_predict in LIN_REG_VARS:

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # LOADING THE DATASET
    # Loading the dataset with all the variables
    input_df = pd.read_csv(f"unnormalized datasets/{var_to_predict}.csv")
    input_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
    # retrieving the variables in which we are interested
    df = input_df[['Date', f'{var_to_predict}']]
    # Converting the date
    # df['Date'] = pd.to_datetime(df['Date'])

    #----------------------------------------------------------------------------------------------------------------------------
    # NORMALIZING THE DATA
    scaler = MinMaxScaler()
    df[f'{var_to_predict}'] = scaler.fit_transform(np.array(df[f'{var_to_predict}']).reshape(-1, 1))
    joblib.dump(scaler, f'my scalers/{var_to_predict}.scale')
    print(df[f'{var_to_predict}'].head())


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    # TRAINING AND TEST SPLITS

    # splitting data between training and testing
    train_data_size = int(TRAIN_SPLIT * len(df)) 
    train_data = df[:train_data_size]
    test_data = df[train_data_size:len(df)]

    # Defining our train and test datasets based on the divide_time_series function
    train_X, train_y = divide_time_series(x = train_data[f'{var_to_predict}'],
                                    y = train_data[f'{var_to_predict}'],
                                    prev_steps = PREVIOUS_STEPS)
    test_X, test_y = divide_time_series(x = test_data[f'{var_to_predict}'],
                                    y = test_data[f'{var_to_predict}'],
                                    prev_steps = PREVIOUS_STEPS)


    model = LinearRegression().fit(train_X, train_y)
    test_predictions = model.predict(test_X)

    # plt.plot(test_predictions, linewidth = 0.5)
    # plt.plot(test_y, linewidth = 0.5)
    # plt.show()
    # plt.plot(test_predictions)
    # plt.show()

    joblib.dump(model, f'models/{var_to_predict}.h5')
    
    # tf.keras.models.save_model(model = model,
    #                            filepath = f'./models/linear_{var_to_predict}',
    #                            save_format = 'tf')

    rmse = np.sqrt(mse(test_predictions, test_y))
    print(f'{var_to_predict} = {rmse}')


# %%
