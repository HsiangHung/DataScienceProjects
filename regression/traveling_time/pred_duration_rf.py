'''This code implements random forest for regression models on each time. There are 24 hours a day, so
   there are 24 various models. The R^2 is minimum at 3PM and then raises until midnight.
   The best model happens for time= 9AM.
'''
import numpy as np
import pandas as pd
from datetime import datetime
from time import time
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import model_selection

def R_squared(predict, true):
    '''This function is used to compute RMSE and R square'''
    if len(predict) != len(true): return False
    meanTrue = np.mean(true)
    RSS =0
    TSS =0
    for i in range(len(predict)):
        RSS += (predict[i] - true[i])**2
        TSS += (true[i] - meanTrue)**2
    return np.sqrt(RSS/len(predict)), 1.0-RSS/TSS


def preprocess_time(data):
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
        print (self.trainData.shape)

        self.trainData['man_dist'] = pd.Series(abs(self.trainData['start_lng']-self.trainData['end_lng'])+\
                    abs(self.trainData['start_lat']-self.trainData['end_lat']), index=self.trainData.index)

        preprocess_time(self.trainData)
        print (self.trainData.shape)





    def training(self):
        '''train the random-forest-regression model with/without grid search
           Note here I separate the whole data to 24 models for different times
           Each of them is independently trained.'''

        best_models = {}
        #for hour in [3, 4, 13, 23]:
        for hour in range(24):


            X = self.trainData[self.trainData['hour'] == hour]
            y = self.trainData[self.trainData['hour'] == hour]['duration']
            del X['duration']
            del X['hour']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            y_test = y_test.tolist()

            #rfReg = RandomForestRegressor();
            #parameters = {'n_estimators': [4,8,12],'max_depth':[10,20],'min_samples_leaf':[2,5,10]};
            #model_cv_grid = model_selection.GridSearchCV(rfReg,parameters,verbose=2,n_jobs=-1);
            #model_cv_grid.fit(X_train, y_train);
            #best_rf_model = model_cv_grid.best_estimator_
            #best_rf_model.fit(X_train, y_train)
            #predictedY = best_rf_model.predict(X_test)
            #best_models[hour] = best_rf_model
            #print (' (RMSE, Rsquare) = ',R_squared(predictedY, y_test))

            rfReg = RandomForestRegressor(n_estimators=9, n_jobs=-1, random_state=100, verbose=0, warm_start=False)
            rfReg.fit(X_train, y_train)
            rf_pred = rfReg.predict(X_test)
            print (hour, R_squared(rf_pred, y_test))

            best_models[hour] = rfReg

        return best_models




    def predict(self, data_source, models):
        '''read test.csv file and then predict duration'''
        testData = pd.read_csv(data_source)
        print (testData.shape)

        testData['man_dist'] = pd.Series(abs(testData['start_lng']-testData['end_lng'])+\
                    abs(testData['start_lat']-testData['end_lat']), index=testData.index)

        preprocess_time(testData)


        print (testData.shape)
        print (testData.head())

        duration = []
        for i in range(testData.shape[0]):
            test = testData.iloc[i,:7]
            hour = testData.iloc[i,7]
            duration.append([i, models[int(hour)].predict(test.reshape(1, -1))[0]])
            if i% 10000==0: print (i,models[int(hour)].predict(test.reshape(1, -1))[0])

        df3 = pd.DataFrame(np.array(duration), columns=['row_id', 'duration'])
        #df3 = pd.Series(np.array(duration), columns=['row_id', 'duration'])

        print(df3.head())

        df3.to_csv('duration2.csv')








O = duration_model('train.csv')
models = O.training()

O.predict('test.csv', models)

#print (models)
