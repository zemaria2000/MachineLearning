# %%
from settings import INFLUXDB, INJECT_TIME_INTERVAL, DATA_DIR, VARIABLES
from datetime import datetime
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import time

# %%
df = pd.DataFrame()

for var in VARIABLES:
    new_df = pd.read_csv(f'{DATA_DIR}{var}.csv', index_col = 'Unnamed: 0')
    new_df = new_df[f'{var}']
    df = pd.concat([df, new_df], axis = 1)

original_size_df = df.shape[0]

# instantiating our InfluxDB client
client = influxdb_client.InfluxDBClient(
    url=INFLUXDB['URL'],
    token=INFLUXDB['Token'],
    org=INFLUXDB['Org']
)
write_api = client.write_api(write_options=SYNCHRONOUS)


# Defining the function to send a value to the db every 10 seconds
def send_values_to_influx(var, df_to_send):
    msg = influxdb_client.Point(var) \
        .tag("Model", var) \
        .field("value", df_to_send.iloc[0][f'{var}']) \
        .time(datetime.utcnow(), influxdb_client.WritePrecision.NS)
            
    write_api.write(bucket=INFLUXDB['Bucket_Data'], org=INFLUXDB['Org'], record=msg)


while df.shape[0] > 0:

    for var in VARIABLES:       
        send_values_to_influx(var, df_to_send = df)

    # removing the value sent from the dataframe
    df = df.iloc[1:]   

    print('New values sent')
    print(df.shape[0])
    
    time.sleep(INJECT_TIME_INTERVAL)




    
    
