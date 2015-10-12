import pandas as pd
import numpy as np
import csv
import sys
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from featurize import get_median_neighbors, get_median_neighbors_no_adjust, run_nmf, clean, tokenize, get_tfidf
from models import linear_regression, random_forest, gradient_boosting
import cPickle
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn

def add_latent_features(df, n_topics):
	'''
	INPUT: Pandas dataframe, number of topics
	OUTPUT: Pandas dataframe with latent features
	'''
	description=clean(df.description)
	vectorizer, X= get_tfidf(description)
	latent_weights = run_nmf(X, vectorizer, n_topics=n_topics, print_top_words=True)
	latent_df      = pd.DataFrame(latent_weights, columns=(
	                            ['Latent Feature %s' % (i + 1) for i in range(n_topics)]))
	
	concat_df = pd.concat([df, latent_df], axis = 1)
	return concat_df

def remove_outlier(df):
	'''
	INPUT: Pandas dataframe
	OUTPUT: Pandas dataframe after removing outliers
	'''
	df= df[df['lot size'] < 8000]
	df= df[df['sqft'] < 4000]
	df = df[df['last sale price'] < 2500000]
	return df

def cal_month(df):
    df['month'] = [item.month - 9 if item.year == 2012 else item.month + 3 \
                   if item.year == 2013 else item.month + 15 if item.year == 2014 \
                   else item.month + 27 for item in df['last sale date']]
def cal_quarter(df):
    df['quarter'] = [(int(item)/3) + 1 for item in df['month']]

def prepare(df, n_topics, n_nbs):
	'''
	INPUT: Pandas dataframe, number of topics, number of neighbors
	OUTPUT: Pandas dataframe with features ready to use
	'''
	df = df.drop('Unnamed: 0', axis = 1)

	df['id'] = df['rid']
	df = add_latent_features(df, n_topics)
    

	df = remove_outlier(df)

	df['last sale date'] = pd.to_datetime(df['last sale date'])
	df = df[df['list_date_2'] > 0]
	df['list_date_2'] = pd.to_datetime(df['list_date_2'])

	df['days_on_market'] = df['last sale date']- df['list_date_2']
	df['days_on_market'] = [item.days for item in df['days_on_market']]
	df = df[df['days_on_market']>0]
	df['sale_d'] = datetime(2015, 10, 5)-df['last sale date']
	df['sale_y'] = [item.year for item in df['last sale date']]
	df['sale_m'] = [item.month for item in df['last sale date']]
	df['2012'] = df['sale_y'] == 2012
	df['2013'] = df['sale_y'] == 2013
	df['2014'] = df['sale_y'] == 2014
   

	df['sale_d'] = [item.days for item in df['sale_d']]
  
	df = df.reset_index() # The index need to be reset every time before get median
	
	df = get_median_neighbors(df, n_nbs, 0.15)
	#df = get_median_neighbors_no_adjust(df, n_nbs)
	
	df['med_neighbor_price'] = df['med_neighbor_price'].fillna(df['med_neighbor_price'].median())
	print 'MAE is:', mean_absolute_error(df['last sale price'], df['med_neighbor_price'])
	print 'MAPE is:', np.mean(np.absolute(df['last sale price'] - df['med_neighbor_price']) / df['last sale price'])
	
    
	df['list_over_actual']= df['list price']*1.0/df['last sale price']
	#df.to_csv('../data/process/data_med.csv')
	cal_month(df)
	cal_quarter(df)
	df.to_csv('for_pred.csv')

	return df

def create_test(df):
	'''
	INPUT: Pandas dataframe
	OUTPUT: Create different data set for training and testing; use the past six month data for testing.
	'''
	df_train = df[pd.to_datetime(df['last sale date']) <= datetime(2015, 4, 4)]
	df_test = df[pd.to_datetime(df['last sale date']) > datetime(2015, 4, 4)]
	return df_train, df_test

def create_test_price_no_latent_mnp(df):
	'''
	INPUT: Pandas dataframe
	OUTPUT: Filter the dataframe to have the predictors and target, return X and y
	'''
	y = df['last sale price']
	X = df[['med_neighbor_price', 'days_on_market', 'month', 'quarter', 'sale_y', 'sale_m', 'beds','baths', 'sqft', 'lot size', 'age', 'condo', 'sfh']]
	return X, y

def create_test_price_latent_mnp(df):
	'''
	INPUT: Pandas dataframe
	OUTPUT: Filter the dataframe to have the predictors and target, return X and y
	'''
	y = df['last sale price']
	X = df[['med_neighbor_price','days_on_market', 'month', 'quarter', 'HOA_amount', 'Latent Feature 1', 'Latent Feature 2','Latent Feature 3', 'Latent Feature 4', 'sale_y', 'sale_m', 'beds','baths', 'sqft', 'lot size', 'age', 'condo', 'sfh']]
	return X, y

def create_test_price_latent_no_mnp(df):
	'''
	INPUT: Pandas dataframe
	OUTPUT: Filter the dataframe to have the predictors and target, return X and y
	'''
	y = df['last sale price']
	X = df[['days_on_market', 'month', 'quarter', 'HOA_amount', 'Latent Feature 1', 'Latent Feature 2','Latent Feature 3', 'Latent Feature 4', 'sale_y', 'sale_m', 'beds','baths', 'sqft', 'lot size', 'age', 'condo', 'sfh']]
	return X, y

def create_test_dom(df):
	'''
	INPUT: Pandas dataframe
	OUTPUT: Filter the dataframe to have the predictors and target, return X and y
	'''
	y = df['days_on_market']
	X = df[['HOA_amount','list_over_actual', 'Latent Feature 1', 'Latent Feature 2','Latent Feature 3', 'Latent Feature 4',\
	        'beds','baths', 'sqft', 'lot size', 'age', 'condo', 'sfh', 'recent_reduction']]
	return X, y

def predict(model_name, X, y):
	'''
	INPUT: Model name, X, y
	OUTPUT: Print MAE (Mean Absolute Error) of the model
 	'''
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
	model = model_name(X_train, y_train)
	y_pred = model.predict(X_test)
	y_p_train = model.predict(X_train)
	print 'Testing MAE is:', np.mean(np.absolute(y_test - y_pred))
	print 'Testing MAPE is:', np.mean(np.absolute(y_test - y_pred) / y_test)
	print 'Training MAE is:', np.mean(np.absolute(y_train - y_p_train))
	print 'Training MAPE is:', np.mean(np.absolute(y_train - y_p_train) / y_train)
	
	return model, y_pred, y_test, X_test, X_train



if __name__=='__main__':
	data = pd.read_csv('../data/process/clean_data.csv')
	df = prepare(data, 4, 10)

	#X, y = create_test_price_no_latent_mnp(df)
	#model, y_pred, y_test, X_test, X_train = predict(random_forest, X, y)
	#X, y = create_test_price_latent_no_mnp(df)
	#model, y_pred, y_test, X_test, X_train = predict(random_forest, X, y)
	
	X, y = create_test_price_latent_mnp(df)
	model, y_pred, y_test, X_test, X_train = predict(gradient_boosting, X, y)
