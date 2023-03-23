#%%
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from settings import INFLUXDB, PREVIOUS_STEPS, INJECT_TIME_INTERVAL, VARIABLES
from keras.models import load_model
from datetime import datetime, timedelta
import numpy as np
import time
import joblib

#%%
# -----------------------------------------------------------------------------
# 1. Instantiating the InfluxDB client

# Instantiate the InfluxDB client
client = influxdb_client.InfluxDBClient(
    url = INFLUXDB['URL'],
    token = INFLUXDB['Token'],
    org = INFLUXDB['Org']
)
# Instantiate the write api client
write_api = client.write_api(write_options = SYNCHRONOUS)
# Instantiate the query api client
query_api = client.query_api()


#%%
while True:

    for var in VARIABLES:

        # defining the two buckets where we will retrieve/insert data in the database
        data_bucket = INFLUXDB['Bucket_Data']
        predictions_bucket = INFLUXDB['Bucket_Predictions']

        # -----------------------------------------------------------------------------
        # Retrieving the necessary data to make a prediction (based on some of the settings)
        # influxDB documentation - https://docs.influxdata.com/influxdb/cloud/api-guide/client-libraries/python/
        # query to retrieve the data from the bucket
        query = f'from(bucket:"{data_bucket}")\
            |> range(start: -1h)\
            |> sort(columns: ["_time"], desc: true)\
            |> limit(n: {PREVIOUS_STEPS})\
            |> filter(fn:(r) => r._measurement == "{var}")\
            |> filter(fn:(r) => r.Model == "{var}")\
            |> filter(fn:(r) => r._field == "value")'

        # Send the query defined above retrieving the needed data from the database
        result = query_api.query(org = INFLUXDB['Org'], query = query)

        # getting the values
        results = []
        for table in result:
            for record in table.records:
                results.append((record.get_value()))

        # normalizing our data (because in the database we are injecting data in it's normal state)
        # loading the scaler
        scaler = joblib.load(f'my scalers/{var}.scale')
        results = scaler.fit_transform(np.array(results).reshape(-1, 1))
        print(results + '\n\n')

        # reverse the vector so that the last measurement is the last timestamp
        results.reverse()

        # Turning them into a numpy array, and reshaping so that it has the shape that we used to build the model
        array = np.array(results).reshape(1, PREVIOUS_STEPS)
        
        # Loading our AutoEncoder model
        model = load_model(f'models/{var}.h5')

        # Making a prediction based on the values that were retrieved
        test_predict = model.predict(array)
        # Retrieving the y prediction
        test_predict_y = test_predict[0][PREVIOUS_STEPS - 1]

        # Putting our value back on it's unnormalized form
        test_predict_y = scaler.inverse_transform(np.array(test_predict_y).reshape(-1, 1))
    
        # getting the future timestamp
        actual_ts = datetime.utcnow()
        future_ts = actual_ts + timedelta(seconds = INJECT_TIME_INTERVAL)

        # Sending the current prediction to a bucket 
        msg = influxdb_client.Point(var) \
            .tag("Model", var) \
            .field("value", test_predict_y) \
            .time(future_ts, influxdb_client.WritePrecision.NS)
        write_api.write(bucket = predictions_bucket, org = INFLUXDB['Org'], record = msg)

        # Debugging the prediction
        print(f'{var} = {test_predict_y}')

    print('Predictions successfully sent to the database... Waiting 30 secs for the next predictions')
    
    # Wait for the next injection of data
    time.sleep(INJECT_TIME_INTERVAL)


# %%
