'''This production code implements linear-regression model with L2 regularization for prediction.
   For detailed explanation, please check the notebook file.
   NOTE: in this code, I use mean absolute erro and R-square as metric. BUt in notebook, I show RMSE.
'''
import numpy as np
import pandas as pd
from datetime import datetime
from time import time
import time
import copy
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

def R_squared(predict, true):
    '''This function is used to compute MAE and R-square'''
    if len(predict) != len(true): return False
    meanTrue = np.mean(true)
    RSS = 0
    TSS = 0
    MAE = 0
    for i in range(len(predict)):
        MAE += abs(predict[i] - true[i])
        RSS += (predict[i] - true[i])**2
        TSS += (true[i] - meanTrue)**2
    return MAE/len(predict), 1.0-RSS/TSS


def preprocess_time(data):
    '''This convert timestampt to local time, days, month and weekday'''
    months = []
    hours = []
    wdays = []
    for i in data['start_timestamp']:
        localTime = time.localtime(i)
        month = localTime.tm_mon
        hour = localTime.tm_hour
        wday = localTime.tm_wday
        months.append(month)
        hours.append(hour)
        wdays.append(wday)

    data['month']=pd.Series(months)
    data['wday']=pd.Series(wdays)
    data['hour']=pd.Series(hours)
    del data['start_timestamp']
    del data['row_id']



class duration_model():
    
    def __init__(self, data_source):
        '''prepare data and preprocessing,  training datasets.
        '''
        self.trainData = pd.read_csv(data_source)
        #print (self.trainData.shape)

        self.trainData['man_dist'] = pd.Series(abs(self.trainData['start_lng']-self.trainData['end_lng'])+\
                    abs(self.trainData['start_lat']-self.trainData['end_lat']), index=self.trainData.index)
        ## this generate Mahanttan distance ------

        preprocess_time(self.trainData)
        #print (self.trainData.shape)


# -----------------------------------------------------------------------------------------


    def training(self):
        '''train the linear-regression model with L2 regularization. Since the data is big enough
           I won't use cross-validation to save time in training models'''

        self.trainData['month'] = self.trainData['month'].astype('category');
        self.trainData['wday']  = self.trainData['wday'].astype('category');
        self.trainData['hour']  = self.trainData['hour'].astype('category');

        self.trainData = pd.get_dummies(self.trainData)

        #print (self.trainData.head())

        y = self.trainData['duration'];
        X = self.trainData;
        del X['duration']


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        y_test = y_test.tolist()

        print ('training data size: ', X_train.shape, y_train.shape)


        print (' ----- start to training model ----- ')

        best_ridge_model = None
        max_rscore = -1
        best_lambda = -1
        print ('lambda,      MAE,      R-square' )
        for lambda_i in [0.001, 1e-2, 0.05, 0.1, 0.2, 1.0, 2.0]:
            ridge = linear_model.Ridge(alpha=lambda_i)
            ridge.fit(X_train, y_train)
            pred = ridge.predict(X_test)
            MAE, R_sq = R_squared(pred, y_test)
            if R_sq > max_rscore:
                max_rscore = R_sq
                best_ridge_model = ridge
                best_lambda = lambda_i
            print (lambda_i, MAE, R_sq)

        print ('best parameter:', best_lambda, max_rscore)

        return best_ridge_model


## --------------------------------------------------------------------------


    def predict(self, data_source, best_model):
        '''read "test.csv" file and then predict duration using the linear model'''
        testData = pd.read_csv(data_source)
        #print (testData.shape)

        testData['man_dist'] = pd.Series(abs(testData['start_lng']-testData['end_lng'])+\
                    abs(testData['start_lat']-testData['end_lat']), index=testData.index)

        preprocess_time(testData)

        testData['month'] = testData['month'].astype('category');
        testData['wday']  = testData['wday'].astype('category');
        testData['hour']  = testData['hour'].astype('category');

        testData = pd.get_dummies(testData)
        ## data preprocessing (convert categorical to OEH, and local time) is done ------ 


        print ('test data size: ', testData.shape)

        prediction = best_model.predict(testData)

        file = open("duration3.csv", "w")
        file.write('row_id,duration\n')
        for i in range(len(prediction)):
            file.write(str(i)+','+str(prediction[i])+'\n')
        file.close









O = duration_model('train.csv')
models = O.training()

O.predict('test.csv', models)

#print (models)
