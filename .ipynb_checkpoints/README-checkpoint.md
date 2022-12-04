# Udacity Data Science Nanodegree

# Disaster Response Pipeline

## Introduction

This project is based on using ML, NLP and real life disaster response messages to classify the new messages into 36 different categories. </br>

This project contains 3 different folders:
- data : This folder contains the raw data in the form of two csv files. It also conatins a python script by the name process_data.py that extracts, transforms and load the data into a sqlite database (DisasterResponse.db).
- models : This folder contains train_classifoer.py that trains the model and out put the best model as a pickle file.
- app : This folder contains the necessary files for creating the app that predicts any new message.

Folder structure : 

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md



## Packages used

- Data transformation : pandas, numpy
- sqlite database : sqlalchemy
- NLP : nltk, re
- Machine Learning : sklearn
- Model saving : pickle


## Executing the files

- ETL : python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

- Model : python train_classifier.py ../data/DisasterResponse.db classifier.pkl

- app : python run.py
