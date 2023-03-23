# percentage of the dataset used for training
TRAIN_SPLIT = 0.9

# Number of timestamps to look back in order to make a prediction
PREVIOUS_STEPS = 30

# now some training parameters

TRAINING = {
    'EPOCHS': 200,
    'BATCH_SIZE': 50
}

# Optimizer used by the algorithm through it's epochs to minimize the cost function
OPTIMIZER = 'adam'
# OPTIMIZER = 'sgd'
# OPTIMIZER = 'adadelta'


# Variables to predict
VARIABLES = {
    'P_SUM',
    'U_L1_N',
    'I_SUM',
    'H_TDH_I_L3_N',
    'F',
    'ReacEc_L1',
    'C_phi_L3',
    'ReacEc_L3',
    'RealE_SUM',
    'H_TDH_U_L2_N'
}

# Variables to which a linear regression is more suitable
LIN_REG_VARS = {
    'RealE_SUM', 
    'ReacEc_L1', 
    'ReacEc_L3'
}

# Datasets directory
DATA_DIR = './unnormalized datasets/'

# my influxDB settings 
INFLUXDB = {
    'URL': "http://localhost:8086",
    'Token': "bqKaz1mOHKRTJBQq6_ON4qk89U02e99xFc2jBN89M4OMaDOyYMHR7q7DDKR7PPiX7wKCiXC8X_9NbF27-aW7wg==",
    'Org': "UA",
    'Bucket_Data': "ML_Data",
    'Bucket_Predictions': "ML_Predictions"
}

# for now, some simulation configurations
INJECT_TIME_INTERVAL = 30   #time, in seconds, between each inject

# For now, a fixed value that is equal to every variable (just for testing basically)
AD_THRESHOLD = 0.05     # RMSE value above which we consider a point to be an anomaly


# Excel reports directory
EXCEL_DIR = './Reports/'

