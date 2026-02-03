


import pandas as pd 
from prophet import Prophet


trainingData = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_trainingData_train.csv")

trainingData.head()



#format data for prophet
trainingDataC = trainingData[['Timestamp', 'trips']]
trainingDataC.columns = ['ds', 'y']
trainingDataC.head()
model = Prophet()
modelFit = model.fit(trainingDataC)

#import and format test data
test = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv")
test.head()
testC = test[['Timestamp']]
testC.head()
testC.columns=['ds']

#forecast
forecast = modelFit.predict(test_p)
pred = forecast['yhat'].values
forecast['yhat'].head()








s