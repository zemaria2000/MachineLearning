# %%
import pandas as pd
import numpy as np
from settings import DATA_DIR
from settings import VARIABLES
import joblib

#%%

for var in VARIABLES:

    # Loading the dataset with all the variables
    input_df = pd.read_csv(f"{DATA_DIR}{var}.csv")
    input_df.rename(columns = {'Unnamed: 0': 'Date'}, inplace = True)
    # retrieving the variables in which we are interested
    df = input_df[['Date', f'{var}']]
    # Converting the date

    # using Pedro's scalers to take the data back to it's original form
    var_scaler = joblib.load(f'scalers/{var}.scale')
    result = var_scaler.inverse_transform(np.array(df[f'{var}']).reshape(-1, 1))
    result.tolist()

    df[f'{var}'] = result

    df.to_csv(f'unnormalized datasets/{var}.csv')


