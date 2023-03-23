# This file tries to join the 'RealTimeAnomalyDetection.py' + 'RealTimePrediction.py' in one basically. The purpose is to be able to make a prediction almost as the same time as we recieve new data from the database

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from settings import INFLUXDB, PREVIOUS_STEPS, INJECT_TIME_INTERVAL, VARIABLES, AD_THRESHOLD
from keras.models import load_model
from datetime import datetime, timedelta
import numpy as np
import time
import pandas as pd
import schedule
import os
from ClassAssistant import Email_Intelligent_Assistant
import joblib


# ------------------------------------------------------------------------------
# 1. SETTING SOME FIXED VARIABLES

data_bucket = INFLUXDB['Bucket_Data']
predictions_bucket = INFLUXDB['Bucket_Predictions']
# My email address and password (created by gmail) - see tutorial How to Send Emails Using Python - Plain Text, Adding Attachments, HTML Emails, and More
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')


# -----------------------------------------------------------------------------
# 2. INSTANTIATING THE INFLUXDB CLIENT

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


# ------------------------------------------------------------------------------
# 3. STARTING OUR EMAIL ASSISTANT OBJECT
email_assistant = Email_Intelligent_Assistant(EMAIL_ADDRESS=EMAIL_ADDRESS, EMAIL_PASSWORD=EMAIL_PASSWORD)


# -------------------------------------------------------------------------------------------------
# 4. SCHEDULLING SOME FUNCTIONS TO BE EXECUTED
# now we have 5 mins, but this is meant to be an hour or more
schedule.every(5).minutes.do(email_assistant.send_email_notification)
schedule.every(5).minutes.do(email_assistant.save_report)
schedule.every(5).minutes.do(email_assistant.generate_blank_excel)
#schedule.every().hour.do()


# generating the first blank excel before the infinite cycle
email_assistant.generate_blank_excel()



# -----------------------------------------------------------------------------
# 5. DEFINING THE PREDICTIONS FUNCTION

def predictions():

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

    print('Predictions successfully sent to the database... Waiting 30 secs for the next predictions \n')


# -----------------------------------------------------------------------------
# 6. DEFINING THE ANOMALY DETECTION FUNCTION

def anomaly_detection():
# Checks if the scheduled functions are readyy to be implemented
    # schedule.run_pending()


    # ------------------------------------------------------------------------------
    # RETRIEVING THE LAST 2 PREDICTED VALUES AND THE LAST REAL VALUE FOR EACH MEASUREMENT

    # Creating 2 empty dictionaries - one for the predicted values and one for the real values
    real_vals = dict()
    predicted_vals = dict()

    # query to retrieve the last 2 forecasted values for
    # why the range tahta way? because we can now retrieve the 2nd to last prediction, which will be related to the last REAL timestamp in the database
    query_pred = f'from(bucket:"{predictions_bucket}")\
        |> range(start: -2m, stop: -25s)\
        |> last()\
        |> filter(fn:(r) => r._field == "value")'

    # query to retrieve the last actual value
    query_last = f'from(bucket:"{data_bucket}")\
        |> range(start: -1h)\
        |> last()\
        |> filter(fn:(r) => r._field == "value")'

    result_pred = query_api.query(org = INFLUXDB['Org'], query = query_pred)
    result_last = query_api.query(org = INFLUXDB['Org'], query = query_last)

    # getting the values for the forecasts and the real values
    results_pred = []
    results_last = []
    for table in result_pred:
        for record in table.records:
            results_pred.append((record.get_value(), record.get_time()))
    for table in result_last:
        for record in table.records:
            results_last.append((record.get_value(), record.get_time()))

    # Getting the timestamp of the values (to then put in the report)
    ts = list()
    for i in range(len(results_last)):
        ts.append(results_last[i][1].strftime("%m/%d/%Y, %H:%M:%S"))

    # getting the 2nd to last prediction to the predicted values dictionary - for now, I'm trying to use the 2nd to last prediction to then compare it to the last value timestamp - it may need to be improved
    i = 0
    mse, rmse = [], []
    for var in VARIABLES:
        # quick explanation - the vector results_pred will have 20 values, 2 for each variable, in a row. So, as we want only the  second one for each variable, we do the 2*i+1 to get the element 1, 3, 5, 7... of the vector
        predicted_vals[f'{var}'] = results_pred[i][0]
        real_vals[f'{var}'] = results_last[i][0]
        # getting the mse and rmse values as well
        mse.append(np.square(np.subtract(real_vals[f'{var}'], predicted_vals[f'{var}'])).mean())
        rmse.append(np.sqrt(mse[i]))
        i += 1

        # sending the rmse values to the database as well, so then we can visualize them in the Grafana interface
        msg = influxdb_client.Point(var) \
            .tag("Model", var) \
            .field("value", rmse[i]) \
            .time(datetime.now(), influxdb_client.WritePrecision.NS)
        write_api.write(bucket = 'RMSE', org = INFLUXDB['Org'], record = msg)


    # -----------------------------------------------------------------------------------------------------------
    # COMPARING THE TWO RESULTS IN ORDER TO DETECT AN ANOMALY
    # for this I'll create a pandas DataFrame with some important columns, which can then be more easily used to send the reports, etc

    df = pd.DataFrame(index = VARIABLES)
    df[['Timestamp', 'Predicted Value', 'Real Value', 'RMSE']] = [ts, predicted_vals.values(), real_vals.values(), rmse]

    # setting up an anomaly filter
    anomaly_filter = (df['RMSE'] > AD_THRESHOLD)
    # getting the anomalies
    anomaly_df = df.loc[anomaly_filter]

    # adding the anomalies to the report
    email_assistant.add_anomalies(anomaly_dataframe = anomaly_df)

    print('The add_anomalies function is working... \n')



# ---------------------------------------------------------------------------------
# 7. INFINITE CYCLE

while True:

    schedule.run_pending()

    predictions()
    anomaly_detection()

    time.sleep(INJECT_TIME_INTERVAL)