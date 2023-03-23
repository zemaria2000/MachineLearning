# In here I'll have to make probably 2 queries: one to retrieve the latest prediction and one to retrieve the real value associated with that prediction
# (which probably means that I'll have to choose the 2nd to last prediction and the last real value). Then I need to compare those two values. Defining a 
# threshold, if I have a difference that is bigger than the threshold, I'll say that that is an anomaly.
# Each anomaly will be saved, hourly, and then a report will be sent via email with all the anomalies that were registered during the previous hour.
# The next step will be to triggr some kind of a warning whenever an anomaly is detected - not exactly via email, just a notification via smartphone or
# even a warning in Grafana

# Next steps - I'll need to analyse (jupyter notebbok maybe) the datasets and define for each variable what can be considered to be an anomaly (probably through the rmse of the predictions)
           # - Then I'll build the queries to retrieve informations from the database
           # - After data, build a class with all the functions needed to send emails and so on
           # - Try and build a grafana interface with graphs of the predicted and real values, as well as warnings to when an anomaly is detected
           # - Try and generate the reports for all the variables that are being monitored
# %%

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from settings import INFLUXDB, PREVIOUS_STEPS, INJECT_TIME_INTERVAL, VARIABLES, AD_THRESHOLD, EXCEL_DIR
import time
import numpy as np
import pandas as pd
from datetime import datetime
import schedule
import os
import smtplib
from email.message import EmailMessage

# ------------------------------------------------------------------------------
# 1. SETTING SOME FIXED VARIABLES

data_bucket = INFLUXDB['Bucket_Data']
predictions_bucket = INFLUXDB['Bucket_Predictions']
# My email address and password (created by gmail) - see tutorial How to Send Emails Using Python - Plain Text, Adding Attachments, HTML Emails, and More
EMAIL_ADDRESS = os.environ.get('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD')

#%%
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
# 3. DEFINING SOME FUNCTIONS

# Generating a new blank excel
def generate_blank_excel():
    cols = {'Timestamp': [], 'Predicted Value': [], 'Real Value': [], 'RMSE': [], 'Notes': []}
    new_file = pd.DataFrame(cols)
    new_file.to_excel('Current_Report.xlsx')

# inputting new anomalies into the report
def add_anomalies(anomaly_dataframe):
    
    notes = []
    for var in anomaly_dataframe.index:
        if anomaly_dataframe.loc[var]['Predicted Value'] > anomaly_dataframe.loc[var]['Real Value']:
            notes.append('The real value was far to low than the prediction given')
        else:
            notes.append('The real value was to high when compared with the prediction given')

    anomaly_dataframe['Notes'] = notes

    # "Original File" 
    file1 = pd.read_excel('Current_Report.xlsx', index_col = 'Unnamed: 0')
    # Adding the new data
    new_file = pd.concat([file1, anomaly_dataframe], ignore_index=False)
    # Saving the new excel file
    new_file.to_excel('Current_Report.xlsx')

# Saving the hourly report
def save_report():
    ts = datetime.now().strftime("%Y-%m-%d__%H-%M")
    report = pd.read_excel('Current_Report.xlsx')
    report.to_excel(f'{EXCEL_DIR}{str(ts)}.xlsx')

# Generating the email message to be sent
def send_email_notification():
        
    # Loading the excel file
    df = pd.read_excel('Current_Report.xlsx', index_col = 'Unnamed: 0')

    # some data processing...
    # number of anomaly occurrences per variable
    anomaly_occur = df.groupby(df.index).size()
    number_anomalies = df.shape[0]
    high_anomalies = len(df[df['Notes'] == 'The real value was far to low than the prediction given'])
    low_anomalies = len(df[df['Notes'] == 'The real value was to high when compared with the prediction given'])
    # Writing a text that summarizes the data gathered
    msg_to_send = f"In total there were {number_anomalies} detected during the last hour \n {high_anomalies} were values that were far too big when compared to the predictions and {low_anomalies} were values too low compared to the predictions \n\n In terms of the variables that had anomalies, we had: \n"

    for var in VARIABLES:
        if var in anomaly_occur:
            msg_to_send += f'   - {var} with {anomaly_occur[var]} anomalies;\n'

    msg_to_send += '\n\n Attached to this email we have an Excel file with all the anomalies, their respective timestamps and some more useful information'

    # Instantiating the EmailMessage object
    msg = EmailMessage()

    # mail sender and receiver
    msg['From'] = EMAIL_ADDRESS
    # msg['To'] = 'zemaria-sta@hotmail.com'

    # Sending emails to several contacts
    contacts = ['zemaria-sta@hotmail.com', 'josemaria@ua.pt']
    msg['To'] = ', '.join(contacts)

    # Subject and content of the message
    msg['Subject'] = 'Hourly report'
    msg.set_content(msg_to_send)
    # Opening the excel report...
    with open('Current_Report.xlsx', 'rb') as file:
        file_data = file.read()
        file_name = "HourlyReport.xlsx"

    # Adding the excel file as an attachment to the email
    msg.add_attachment(file_data, maintype = 'application', subtype = 'xlsx', filename = file_name)

    # sending the email
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        # login to our email server
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)       # Password that is given by google when enabling 2 way authentication 
        # sending the message
        smtp.send_message(msg)
        print('Message sent')


# -------------------------------------------------------------------------------------------------
# 4. SCHEDULLING SOME FUNCTIONS TO BE EXECUTED
# now we have 5 mins, but this is meant to be an hour or more
schedule.every(5).minutes.do(send_email_notification)
schedule.every(5).minutes.do(save_report)
schedule.every(5).minutes.do(generate_blank_excel)
#schedule.every().hour.do()

# Before entering in the infinite loop, generate a first blank excel, that then will be automatically generated
generate_blank_excel()

#%%
#----------------------------------------------------------------------------------------------------
# INFINITE LOOP

while True:
    
    # Checks if the scheduled functions are readyy to be implemented
    schedule.run_pending()


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
        
        # sending the rmse values to the database as well, so then we can visualize them in the Grafana interface
        msg = influxdb_client.Point(var) \
            .tag("Model", var) \
            .field("value", rmse[i]) \
            .time(datetime.now(), influxdb_client.WritePrecision.NS)
        write_api.write(bucket = 'RMSE', org = INFLUXDB['Org'], record = msg)
        
        i += 1

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
    add_anomalies(anomaly_dataframe = anomaly_df)

    print('Working...')

    time.sleep(INJECT_TIME_INTERVAL)


# %%
