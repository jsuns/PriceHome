import pandas as pd
import csv
import re

style = ['Contemporary', 'Modern/High Tech', 'Victorian', 'Edwardian', 'Traditional', 'Georgian', \
         'RanchView', 'Conversion', 'Art Deco', 'Custom', 'Spanish/Mediterranean', 'Bungalow', \
         'Tudor', 'Colonial', 'Spanish', 'Arts & Crafts', 'French', 'English']

month_dict = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,\
              'August':8, 'September':9, 'October':10, 'November':11, 'December':12}


def clean_basic(df):
	'''
	INPUT: Pandas dataframe
	OUTPUT: Pandas dataframe containing cleaned basic data
	'''
	df = df.dropna(subset=['latitude', 'longitude'])
	df['type'] = ['sfh' if item == 'Single Family Residential' else 'condo' if item =='Condo/Coop' else 'th' for item in df['home type']]
	df= pd.concat([df, pd.get_dummies(df['type'])], axis = 1)
	df = df.drop('type', axis = 1)
	df['garage'] = [True if item == 'Garage' else False for item in df['parking type']]
	df['recent_reduction'] = [True if len(str(item))>6 else False for item in df['recent reduction date']]
	df['short_sale']= [True if item ==True else False for item in df['is short sale']]
	df = df.drop(['parking type','sale type','home type','address','city','state','location','days on market','status',\
		          'next open house date','next open house start time','next open house end time',\
		          'recent reduction date','original list price',\
		          'url (see http://www.redfin.com/buy-a-home/comparative-market-analysis for info on pricing)',\
		          'source','original source','favorite','interested'], axis = 1)
	df['lot size'] = df['lot size'].fillna(0)
	df['year built'] = df['year built'].fillna(df['year built'].mean())
	df['parking spots'] = df['parking spots'].fillna(0)
	df = df[df['list price'] > 0]
	df = df[df['sqft'] > 0]
	df = df.drop(['zip'], axis = 1)
	#df = pd.concat([d, pd.get_dummies(df['zip'])], axis=1)
	df['age'] = 2015 - df['year built']
	df = df[df['beds'].notnull()]
	df['baths'] = df['baths'].fillna(df['beds']/2)
	df['beds'] = [int(round(item)) for item in df['beds']]
	df['baths'] = [int(round(item)) for item in df['baths']]
	df = df.reset_index()
	df = df.drop('index', axis = 1)
	return df

def check_style(line):
	'''
	INPUT: String
	OUTPUT: For each style, check if it exits in the string, return a list of styles or None
	'''
	all_styles = []
	for s in style:
	    if s in line:
	        all_styles.append(s)
	if len(all_styles) > 0:
	    return ', '.join(all_styles)
	else:
		return None

def change_date(line):
    '''
    INPUT: String
    OUTPUT: Convert string to appropriate date time format, return string in 2015-01-01 format
    '''
    if line == 'N':
        return None
    else:
        mon_day_year = line.split(',')
        mon_day = mon_day_year[1].split(' ')
        month = month_dict[mon_day[1]]
        day = mon_day[2]
        year = mon_day_year[2].strip()
        date = str(year) + '-' + str(month) + '-' + str(day)
        return date

def clean_meta(df):
    '''
    INPUT: Pandas dataframe
    OUTPUT: Pandas dataframe containing cleaned meta data
    '''
    df['desc_length'] = [len(str(item)) for item in df['description']]
    df['HOA_amount'] = [re.search('HOA\sDues\$(\d*)\/', str(item)).group(1) \
                              if re.search('HOA\sDues\$(\d*)\/', str(item)) else 0 for item in df['HOA']]
    df['transport'] = [re.search('\s*(\d*\+*)\s*B', item).group(1) if re.search('\s*(\d*\+*)\s*B', item) \
                             else 5 for item in df['transportation']]
    df['transport'] = [4 if item == '+' else item for item in df['transport']]
    df['shop'] = [re.search('\s*(\d*\+*)\s*B', item).group(1) if re.search('\s*(\d*\+*)\s*B', item)\
                        else 5 for item in df['shopping']]
    df['shop'] = [4 if item == '4+' else item for item in df['shop']]
    df = df.drop(['HOA', 'transportation', 'shopping'], axis=1)
    df['style2'] = df['style'].apply(lambda x: check_style(x))
    df['style2'] = df['style2'].fillna('Other')

    df = df.drop('style', axis = 1)
    df['desc'] = [str(unicode(item).encode('ascii', 'ignore')) for item in df['description']]
    df['list_date_2'] = df['list_date'].apply(lambda x: change_date(x))
    
    df = df.drop('list_date', axis = 1)
    return df

def combine(df1, df2):
	'''
    INPUT: Two Pandas dataframe
    OUTPUT: Pandas dataframe
    '''
	df = pd.merge(df1, df2, how='inner', left_on='rid', right_on='id')
	df.to_csv('../data/clean_data.csv')
	return df

if __name__=='__main__':
	basic_data = pd.read_csv('../data/basic.csv')
	data1 = clean_basic(basic_data)
	meta_data = pd.read_csv('../data/meta.csv')
	data2 = clean_meta(meta_data)
	data1 = data1.drop('Unnamed: 0', axis = 1)
	data2 = data2.drop('Unnamed: 0', axis = 1)
	data2 = data2.drop_duplicates()

	clean_data = combine(data1, data2)
	

