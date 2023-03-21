# percentage of the dataset used for training
TRAIN_SPLIT = 0.9

# Number of timestamps to look back in order to make a prediction
PREVIOUS_STEPS = 30

# now some training parameters

TRAINING = {
    'EPOCHS': 100,
    'BATCH_SIZE': 32
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