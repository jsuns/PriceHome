
import pandas as pd
import requests
import re
import csv
import time
from bs4 import BeautifulSoup, UnicodeDammit
import glob
import os

def get_rid(urls):
	'''
    INPUT: Pandas series of urls
    OUTPUT: List of house id (file names)
    '''
	fns = []
	for url in urls.values:
		rid = str(url)[-8:]
		rid = rid.replace('e/', '')
		rid = rid.replace('/', '')
		fns.append(rid)
	return fns    

def get_basic_info(zipcode):
	'''
    INPUT: List of zip code
    OUTPUT: Save basic information downloaded from Redfin.com to a csv file, 
            return Pandas series of urls
    '''
	basic_data = pd.DataFrame()
	files = glob.glob("../data/*.csv")

	for filename in sorted(files):
	    print filename
	    d = pd.read_csv(filename)
	    basic_data = pd.concat([basic_data, d], axis = 0)
	basic_data.columns = [item.lower() for item in basic_data.columns]
	basic_data= basic_data.drop_duplicates()
	urls = basic_data['url (see http://www.redfin.com/buy-a-home/comparative-market-analysis for info on pricing)']
	rid = get_rid(urls)
	basic_data['rid'] = rid

	basic_data.to_csv('../data/process/basic.csv')

	return urls

def download(urls):
	'''
    INPUT: Pandas series of urls
    OUTPUT: Save web pages to a folder
    '''
	for url in urls.values:
	    # user_agent = {'User-agent': 'Mozilla/5.0'}
	    # response = requests.get(url, headers = user_agent)    
	    url = url.replace('http:', 'https:')
	    rid = str(url)[-8:]
	    rid = rid.replace('e/', '')
	    rid = rid.replace('/', '')
	    filename = '../webpage/'+ rid
	    cmd = "curl -H 'User-Agent: my browser' " + url + " > " + filename
	    print cmd
	    if os.path.exists(filename) and os.stat(filename).st_size > 100000:
	    	continue
	    time.sleep(2.1)
	    os.system(cmd)
	    # ff = open(filename, 'w+')
	    # ff.write(response.content)
	    # ff.close
  

def scrape(filenames):
	'''
    INPUT: List of file names
    OUTPUT: Save Pandas dataframe of meta information to a csv file 
            (including id, description, HOA, style, transportation, 
            shopping information and listing date)
    '''
	data = []
	for fn in filenames:
		filename = '../webpage/' + str(fn)
		ff = open (filename, 'rw')
		soup = BeautifulSoup(ff, from_encoding='UTF-8')
		row = []
		'''get id'''
		row.append(str(fn))

		'''get description'''
		row_desc = soup.find('div', {'class':'remarks'})
		if row_desc:
			row.append(unicode(row_desc.text))
		else:
			row.append('N')

		'''get style'''
		if re.search('Style', unicode([each.text for each in soup.findAll('tbody')][1])):
			row.append(unicode([each.text for each in \
				       soup.findAll('tbody')][1]).split('Style')[1].split('Community')[0])
		else:
			row.append('N')

		'''get HOA'''
		row.append(unicode([each.text for each in soup.findAll('tbody')][1]).split('Style')[0])

		'''get transportation information'''
		if len(soup.findAll('div', {'class': 'super-group-content'}))>=2:
			if re.search('Shopping', soup.findAll('div', {'class': 'super-group-content'})[1].text):
				row.append(unicode(soup.findAll('div', {'class': 'super-group-content'})[1].text.split('Shopping:')[0][-13:]))
				'''get shopping information'''
				row.append(unicode(soup.findAll('div', {'class': 'super-group-content'})[1].text.split('Shopping:')[1][:14]))
			else:
				row.append('N')
				row.append('N')
		else:
			row.append('N')
			row.append('N')

		'''get listing date'''
		text = soup.findAll('div', {'class': 'super-group-content'})
		match = re.search('\$ON\_MARKET\_DATE\.1\"\>(.*?)\<', str(text))
		if match:
			row.append(match.group(1))
		else:
			row.append('N')
		data.append(row)

	metadata = pd.DataFrame(data = data, columns=['id', 'description', \
		                    'style', 'HOA', 'transportation', 'shopping', 'list_date'])
	
	metadata['description']= [str(unicode(item).encode('ascii', 'ignore')) \
	                          for item in metadata['description']]
	
	metadata.to_csv('../data/process/meta.csv', encoding='utf-8')   

if __name__=='__main__':
	zipcode = [94102, 94103, 94104, 94105, 94107, 94108, 94109, 94110, \
	           94111, 94112, 94114, 94115, 94116, 94117, 94118, 94121, \
	           94122, 94123, 94124, 94127, 94131, 94132, 94133, 94134, 94158]
	urls = get_basic_info(zipcode)
	print urls.shape
	download(urls)
	# fns = get_rid(urls)
	# scrape(fns)
