import pandas as pd
import numpy as np 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.punkt import PunktLanguageVars
import re
from sklearn.decomposition import NMF
from sklearn.neighbors import KDTree 
import cPickle
'''
This is written based on Jon Oleson's code

'''

def clean(descriptions):
	'''
	INPUT: List of descriptions
	OUTPUT: List of cleaned descriptions
	'''
	clean_ds = []
	for desc in descriptions:
	    clean_desc = re.sub("[^a-zA-Z]"," ", str(desc)) #grabs only letters
	    clean_desc = re.sub(' +',' ', clean_desc) #removes multiple spaces
	    clean_desc = clean_desc.lower() #converts to lower case
	    clean_ds.append(clean_desc)
	return clean_ds


def tokenize(desc):
	'''
	INPUT: List of cleaned descriptions
	OUTPUT: Tokenized and stemmed list of words from the descriptions 
	'''
	plv = PunktLanguageVars()
	snowball = SnowballStemmer('english')
	return [snowball.stem(word) for word in plv.word_tokenize(desc.lower())]


def get_tfidf(clean_desc): 
	'''
	INPUT: List of cleaned descriptions
	OUTPUT: TF-IDF vectorizer object, vectorized array of words from the corpus
	'''
	vectorizer = TfidfVectorizer('corpus', tokenizer = tokenize,
	                              stop_words=stopwords.words('english'), 
	                              strip_accents='unicode', norm='l2')
	vectorizer.fit(clean_desc)
	cPickle.dump(vectorizer, open('../models/tfidf.pkl', 'wb'))
	X = vectorizer.transform(clean_desc)
	return vectorizer, X


def run_nmf(X, vectorizer, n_topics=5, print_top_words=False):
	'''
	INPUT: Vectorized word array, vectorizer object, number of latent 
	features to uncover, whether to print the top words from each latent
	feature
	OUTPUT: Saves pickled NMF model, returns latent weights matrix that
	can be concatenated with our dataset as additional features  
	'''
	nmf = NMF(n_components=n_topics)
	nmf.fit(X)
	cPickle.dump(nmf, open('../models/nmf.pkl', 'wb'))
	H = nmf.transform(X)

	if print_top_words == True:
		feature_names = vectorizer.get_feature_names()
		n_top_words = 15
		for topic_idx, topic in enumerate(nmf.components_):
			print("Topic #%d:" % topic_idx)
			print(" ".join([feature_names[i]
			                for i in topic.argsort()[:-n_top_words - 1:-1]]))
			print()
	
	return H 

def get_median_neighbors(df, n_neighbors, adj_r):
	'''
	INPUT: Pandas dataframe, and the number of comparable neighbors
	of each listing we'll take the median price of in adding the 
	median_neighbor_prices feature
	OUTPUT: Pandas dataframe with the median prices of the n_neighbors
	closest comparables added as a feature. This is accomplished using a 
	KD-Tree model to search for nearest-neighbors
	'''
	kd_df = df[['latitude', 'longitude']]
	kdvals = kd_df.values
	kd = KDTree(kdvals, leaf_size = 1000)
	cPickle.dump(kd, open('../models/kd_tree.pkl', 'wb'))
	neighbors = kd.query(kdvals, k=100)

	median_neighbor_prices = []
	
	for i in xrange(len(df)):
	    listing_neighbors = neighbors[1][i]
	    listing_id = df.ix[i,'id']
	    n_beds = df.ix[i,'beds']
	    sale_y = df.ix[i, 'sale_y']
	   
	    sub_df = df[(df.index.isin(listing_neighbors))]
	    sub_df = sub_df[
	        (sub_df['beds']  == n_beds)  &
	        (sub_df['id']    != listing_id)
	        ]

	    comp_listings = [item for item in listing_neighbors if item in sub_df.index]
	    df_filtered = pd.DataFrame()
	    df_filtered['last sale price']= df['last sale price'][comp_listings][:n_neighbors]
	    df_filtered['sale_y'] = df['sale_y'][comp_listings][:n_neighbors]

	    df_filtered['price adjusted'] = df_filtered['last sale price'] * (1.0 + (sale_y - df_filtered['sale_y']) * adj_r)
	    med_price = df_filtered['price adjusted'].median()
	    if med_price > 0:
	        median_neighbor_prices.append(med_price)
	    else:
			df_filtered = pd.DataFrame()
			df_filtered['last sale price']= df['last sale price'][comp_listings][:n_neighbors+10]
			df_filtered['sale_y'] = df['sale_y'][comp_listings][:n_neighbors+10]

			df_filtered['price adjusted'] = df_filtered['last sale price'] * (1.0 + (sale_y - df_filtered['sale_y']) * adj_r)
			med_price = df_filtered['price adjusted'].median()
			
			if med_price > 0:
			    median_neighbor_prices.append(med_price)
			else:
				df['price adjusted'] = df['last sale price'] * (1.0 + (sale_y - df['sale_y']) * adj_r)
				med_price = df['price adjusted'][comp_listings].median()
				median_neighbor_prices.append(med_price)

	df['med_neighbor_price'] = median_neighbor_prices
	   
	rmse = np.mean((df['med_neighbor_price'] - df['last sale price'])**2)**0.5
	print 'RMSE is ', rmse
	return df    

def get_median_neighbors_no_adjust(df, n_neighbors):
	'''
	INPUT: Pandas dataframe, and the number of comparable neighbors
	of each listing we'll take the median price of in adding the 
	median_neighbor_prices feature
	OUTPUT: Pandas dataframe with the median prices of the n_neighbors
	closest comparables added as a feature. This is accomplished using a 
	KD-Tree model to search for nearest-neighbors
	'''
	kd_df = df[['latitude', 'longitude']]
	kdvals = kd_df.values
	kd = KDTree(kdvals, leaf_size = 1000)
	cPickle.dump(kd, open('../models/kd_tree.pkl', 'wb'))
	neighbors = kd.query(kdvals, k=100)

	median_neighbor_prices = []
	
	for i in xrange(len(df)):
	    listing_neighbors = neighbors[1][i]
	    listing_id = df.ix[i,'id']
	    n_beds = df.ix[i,'beds']
	    sale_y = df.ix[i, 'sale_y']
	   
	    sub_df = df[(df.index.isin(listing_neighbors))]
	    sub_df = sub_df[
	        (sub_df['beds']  == n_beds)  &
	        (sub_df['sale_y'] == sale_y) &
	        (sub_df['id']    != listing_id)
	        ]

	    comp_listings = [item for item in listing_neighbors if item in sub_df.index]
	    med_price = df['last sale price'][comp_listings][:n_neighbors].median()	    

	    if med_price > 0:
	        median_neighbor_prices.append(med_price)
	    else:
			med_price = df['last sale price'][comp_listings][:(n_neighbors+10)].median()
			
			if med_price > 0:
			    median_neighbor_prices.append(med_price)
			else:
				med_price = df['last sale price'][comp_listings].median()
				median_neighbor_prices.append(med_price)

	df['med_neighbor_price'] = median_neighbor_prices
	   
	rmse = np.mean((df['med_neighbor_price'] - df['last sale price'])**2)**0.5
	print 'RMSE is ', rmse
	return df    
