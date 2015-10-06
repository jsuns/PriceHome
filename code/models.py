import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import cPickle

def linear_regression(X_train, y_train):
	'''
    INPUT: Pandas dataframe, Pandas series
    OUTPUT: Pickled model
    '''
	lr = LinearRegression()
	lr.fit(X_train, y_train)
	cPickle.dump(lr, open('../models/lr.pkl', 'wb'))
	return lr
	
def random_forest(X_train, y_train):
	'''
    INPUT: Pandas dataframe, Pandas series
    OUTPUT: Pickled model
    '''
	rf = RandomForestRegressor(n_estimators=50, criterion='mse', min_samples_split=5, min_samples_leaf=5, max_features=10)
	rf.fit(X_train, y_train)
	cPickle.dump(rf, open('../models/rf.pkl', 'wb'))
	return rf

def gradient_boosting(X_train, y_train):
	'''
    INPUT: Pandas dataframe, Pandas series
    OUTPUT: Pickled model
    '''
	gb = GradientBoostingRegressor(n_estimators=50, min_samples_split=5, min_samples_leaf=5, max_features=10)
	gb.fit(X_train, y_train)
	cPickle.dump(gb, open('../models/gb.pkl', 'wb'))
	return gb
