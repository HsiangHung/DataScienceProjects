'''This python code integrates all parts including data preprocessing,
   training the random-forest-model with grid search and evaluate the optimal values of bid amount.
   Note here I default train the model with feature selections; thus only considering 
   "traffic_source", "device", "country" and "PUBLISHER_BID" as input.
   The analysis why I selected the features is based on the iPython note.
'''

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
#from sklearn import grid_search
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


def compute_revenue(model, adv_cpc, traffic, device, country, publisher_bid):
    win_rate = model.predict(np.array([[traffic, device, country, publisher_bid]]))[0]
    #win_rate = model.predict(np.array([[8, 23, traffic, device, country, publisher_bid, 126]]))[0]
    return (adv_cpc-publisher_bid)*win_rate*0.15


class win_rate_model():
    
    def __init__(self, data_source):
        '''prepare data and preprocessing, removing outliers, feature selection, and preparing
           training and test datasets.
        '''
        #data = pd.read_csv('DATA_SCIENCE_CHALLENGE.csv')
        data = pd.read_csv(data_source)
        data['win_rate'] = pd.Series(data['WINS']/data['RESPONSES'], index=data.index)
        del data['WINS']
        data = data[data.win_rate <= 1]
        data = data[data.win_rate >=0]
        
        ## feature selection: 3: 'traffic source', 4: 'device', 5: 'country', 6: 'publisher_bid'
        X = data.iloc[:,[3,4,5,6]]   ## here we mean to do feature selection
        #X = data.iloc[:,1:8]   ## here we consider all features to train model
        y = data.iloc[:,8]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        self.y_test = self.y_test.tolist()


    
    def training(self):
        '''train the random-forest-regression model with grid search'''
        rfReg = RandomForestRegressor()
        parameters = {'n_estimators': [2,4],'max_depth':[10,20,30,40,50],'min_samples_leaf':[2,5,10]}
        model_cv_grid = model_selection.GridSearchCV(rfReg,parameters,verbose=2,n_jobs=-1)
        model_cv_grid.fit(self.X_train,self.y_train)
        best_rf_model = model_cv_grid.best_estimator_
        best_rf_model
        best_rf_model.fit(self.X_train, self.y_train)
        predictedY = best_rf_model.predict(self.X_test)
        print (' (RMSE, Rsquare) = ',R_squared(predictedY, self.y_test))
        return best_rf_model



    def get_optimal_bid(self, model, traffic, device, country):
        '''searching for optimal bid amount. The bid amount varies in increment of 0.01'''
        adv_cpc = [0.01, 0.02, 0.10, 0.15, 0.30, 0.70]
        print ('cpc, optimal_bid, max_net_revenue')
        for cpc in adv_cpc:
            optimal_bid = -1
            max_revenue = -1
            for i in range(1,100):
                bid = 0.01*i
                net_revenue = compute_revenue(model, cpc, traffic, device, country, bid)
                if net_revenue > max_revenue: optimal_bid, max_revenue = bid, net_revenue
            ## this shows optimal value of bid and the corresponding net revenue
            print (cpc, optimal_bid, max_revenue)
        





O = win_rate_model('DATA_SCIENCE_CHALLENGE.csv')


best_model = O.training()
predictedY = best_model.predict(O.X_test)

## ---- this gives features -----
#traffic, device, country = 274040327, 26, 9
#traffic, device, country = 93920677, 22, 156
traffic, device, country = 216145324,22,6


O.get_optimal_bid(best_model, traffic, device, country)

