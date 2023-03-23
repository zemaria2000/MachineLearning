In this repository we have a series of files and folders that allow us to build ML predictive models for some variables, test them, and use them for prediction and anomaly detection tasks. Additionaly, we have an 'Intelligent Assistant' that has the ability 
to generate reports from time to time about the anomalies occured. All the files and folders are explained in the section below.


=========================================================================================================================================

Firstly the folders. There are a bunch of them:
    
    - unnormalized datasets - this folder contains a series of csv files (each for a different variable that I am trying to predict). This folder comes from the 'datasets' folder in Pedro's files. In that, he had all the variables already normalized. That's why I tried to create a bunvh of non-normalized files, to better emulate the conditions in which the data will eventualy arrive from the equipments, which is non-normalized;

    - models - this is the folder where the ML models, after trained, will be saved. Then, whenever we want to use them to make a prediction, we just need to load them up from this folder;

    - scalers - this folder contains the scalers used by Pedro in his work. I basically used these scalers in order to regenerate the non-normalized data, back to it's somewhat 'original state' (this process of de-normalization is ran in the "DenormalizingData.py" file);

    - my scalers - in this folder I stored my own scalers, which are used in the normalization processes that happen before I train the models, or before I apply a specific model, and the denormaliztion processes that are ran whenever we need to send data back to the database, for instance;

    - Reports - folder with some excel files, that try and simulate reports that are sent to the workers / data analysts / maintenance engineers regarding the anomalies that occured in a previous shift / previous hour of work


Now for the python scripts:

    - 'settings.py' - important configuration file, where I've put a series of different fixed parameters that are used amongst the different scripts contained in this folder

    - 'AutoKeras_ModelBuilder_All.py' - in this script we are able to train each one of the models. We have a keras-tuner, which is basically an AutoML tool, that iterates over a series of parameters and tries to build the best possible model in the amount of time / tries we give it. After the model is built, it is saved in an 'h5' file format, in the models folder, where it can then be loaded back to make some predictions (if we want just to train a specific model to predict a specific variable, we can always use the "AutoKeras_ModelBuilder.py" file);

    - 'AutoKeras_ModelTester.py' - after building the model, we can alwaysrun a simple test using this script. This script will take the data, split it into a training and testing split, and use the testing one to compare the predictions to the real values of the testing dataset. We also have some graphical analysis in here...

    - 'AE_AnomalyDetector.py' - It's just the testing script, but with the capability of also giving us a graphical sense of detected anomalies (which are big differences between the real and predicted values)

    - RealTimePrediction.py' - this script allows us to load the modes that were created in the ModelBuilder file, and apply them to make in real time predictions. First it queries the needed values from the InfluxDB bucket, then it normalizes the data, applies the model, de-normalizes the data back to it's "original" magnitude, and finally sends the predicion of the future timestamp for the InfluxDB bucket (for all the variables)

    - 'RealTimeAnomalyDetection.py' - in this case, this script loads the penultimate predictions and the last real values (because the last prediction is relative to the next timestamp in the future). Then it compares the difference between the values. If the difference is bigger than a certain threshold, then something is wrong and is labeled as an anomaly. This script also has programmed the intelligent assistant that puts the anomalies in an excel file and then sends the report every X minutes/hours...;

    - 'IntelligentAssistant.py' - this file has the content of the previous two. That said, it is capable of doing the predictions, the anomaly detection and the email sending operation. When we implement the code, instead of seperately implementing the "TimePredictions" and "AnomalyDetection" files, we just need to run this script. It is important to note that this script depends on the "ClassAssistant.py' one;

    - 'ClassAssistant.py' - It contains some important functions that are used in the 'IntelligentAssistant.py' script, such has saving the report, sending the email or adding a new anomaly to the report;

    - 'DenormalizingData.py' - just a file that I used to denormalize the data and generate the non-normalized csv files;

    - 'LinRegTraining.py' - file where I tested a Linear Regression in every variable, to see if it was worth it in some of them to build a full Autoencoder model;

    - 'InjectData.py' - script that I'm using to inject data in the Database so that I can then use the Intelligent Assistant to collect the data and make the predictions and anomaly detection tasks;


==========================================================================================================

To implement this, we don't need to run all the scripts that were mentioned above. We'll just need the following:

    1. AuroKeras_ModelBuilder_All.py - this will be executed one time - TO TRAIN AND BUILD THE MODELS TO IMPLEMENT. For now, there's no need to re-train the models, as the csv files are always the same. When implementing this for real, we can every week for instance, with new data, train the models in order for them to be adapted to new data, new tendencied, etc.

    2. InjectData.py - for now, in a simulated environment, we need to be injecting data to then be able to collect it and test if the intelligent assistant is working;

    3. IntelligentAssistant.py - with the models trained, we just need to run this script and the prediction and anomaly detection tasks will automatically start. 