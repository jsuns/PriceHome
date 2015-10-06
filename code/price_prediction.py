import pandas as pd
import csv
import sys
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error
from featurize import get_median_neighbors, run_nmf, clean, tokenize, get_tfidf
from models import linear_regression, random_forest
import cPickle
#import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn

def add_latent_features(df, n_topics):
	'''
	INPUT: Pandas dataframe, number of topics
	OUTPUT: Pandas dataframe with latent features
	'''
	description=clean(df.description)
	vectorizer, X= get_tfidf(description)
	latent_weights = run_nmf(X, vectorizer, n_topics=n_topics, print_top_words=True)
	latent_df      = pd.DataFrame(latent_weights, columns=(
	                            ['Latent Feature %s' % (i+1) for i in range(n_topics)]))
	
	concat_df = pd.concat([df, latent_df], axis=1)
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

def prepare(df, n_topics, n_nbs):
	'''
	INPUT: Pandas dataframe, number of topics, number of neighbors
	OUTPUT: Pandas dataframe with features ready to use
	'''
	df = df.drop('Unnamed: 0', axis= 1)

	
	df = add_latent_features(df, n_topics)

	df = remove_outlier(df)
	df = df.reset_index() # The index need to be reset every time before get median
	df = get_median_neighbors(df, n_nbs)
	df['med_neighbor_price'] = df['med_neighbor_price'].fillna(df['med_neighbor_price'].mean())


	df['last sale date'] = pd.to_datetime(df['last sale date'])
	df = df[df['list_date_2']>0]
	df['list_date_2'] = pd.to_datetime(df['list_date_2'])

	df['days_on_market'] = df['last sale date']- df['list_date_2']
	df['days_on_market']= [item.days for item in df['days_on_market']]
	df = df[df['days_on_market']>0]
	df['list_over_actual']= df['list price']*1.0/df['last sale price']
	df.to_csv('../data/data_med.csv')

	return df

def create_test_price(df):
	'''
	INPUT: Pandas dataframe
	OUTPUT: Filter the dataframe to have the predictors and target, return X and y
	'''
	y = df['last sale price']
	X = df[['HOA_amount','days_on_market', 'med_neighbor_price','Latent Feature 1', 'Latent Feature 2','Latent Feature 3', 'Latent Feature 4',\
	        'beds','baths', 'sqft', 'lot size', 'age', 'condo', 'recent_reduction']]
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
	print mean_absolute_error(y_test, y_pred)
	print mean_absolute_error(y_test, y_pred)/y_test.mean()



if __name__=='__main__':
	data = pd.read_csv('../data/clean_data.csv')
	df = prepare(data, 4, 10)
	#X, y = create_test_dom(df)
	X, y = create_test_price(df)

	predict(random_forest, X, y)
	#plt.figure(figsize=(20,10))
	#plt.scatter(X['list_over_actual'], y)
	



