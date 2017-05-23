from sklearn.naive_bayes import GaussianNB
# apply the GaussianNB for Naive_bayes
import numpy as np
import pandas as pd

# import the train data
Train_data=pd.read_csv('Bcancer_Train.csv')

# copy the data into application array
Train_detail=Train_data[:];
# take out the label which is M or B
Train_lable=Train_detail.pop('Condition');
# genenrate the data set for machine learning
Train_detail.pop('Name');

# apply Gaussian distribution for naive bayesian in model
model=GaussianNB();
model.fit(Train_detail,Train_lable);

# input and copy the test data
test_data=pd.read_csv('Bcancer_Test.csv');
test_detail=test_data[:];
# trim the data for test use
test_label=test_detail.pop('Condition');
test_detail.pop('Name');

# get the result after the training
result=model.predict(test_detail);

# count the correct rate
counter=0;
for i in range(0,len(result)):
    if result[i]==test_label[i]:
        counter+=1;

rate=float(counter/len(result));
# output the result
print ('correct rate is %f',rate);