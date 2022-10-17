import time
import os
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, skew
from tqdm import tqdm
from utils import get_taskparams, load_data, send_email
from utils1 import flatten_metrics, open_entities, get_mapping_dict, mmap_subcat, glove_model, within_category_deviation, level1_estimate, pos_based_catagorization
import stanza
from google.oauth2 import service_account
from google.cloud import language, bigquery
import pandas_gbq
import configparser
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


config = configparser.ConfigParser()
config.read('variables.ini')
task_params = get_taskparams(config['Default']['sheet_id'], config['Default']['sheet_name'])

#testing
#xls = pd.ExcelFile('abc.xlsx')
#df = pd.read_excel(xls, 'fan')



class Sentiment_Buckets():
	def __init__(self, data):
		self.data = data

	def get_review_buckets(self):
		self.data.Entity = self.data.Entity.astype(str)
		sadf = self.data.copy()
		
		# Getting level Categories

		mm = get_mapping_dict(sheet_id=mapping_sheet_id, sheet_name='level1')
	    #category
		def mmap(x):
			r = ''
			for k in mm.keys():
			    if k in str(x).lower():
			        return mm[k]
			        r = mm[k]
			        break
			    else:
			    	pass
			if r == '':
				return 'NA'

		sadf['Categories'] = sadf.Entity.apply(lambda x: mmap(x))
		sadf['Categories'] = sadf['Categories'].fillna('NA')
		#sadf.loc[sadf.Entity == 'NA'].Categories = 'IR'

		# getting estimate for remaining level 1 categories

		sadf.Entity = sadf.Entity.apply(lambda x: x if x == 'NA' else x.lower())

		mm_general = get_mapping_dict(sheet_id=mapping_sheet_id, sheet_name='level1_general')
		dev_df = within_category_deviation(glovedf, mm_general)
		def predict_level1_category(x, level1_dict):
		    try:
		        return level1_estimate(glovedf, x, level1_dict, dev_df = dev_df)[1]
		    except:
		        return 'NA'

		#getting categories for entities which could not be mapped by the level1 ductionary
		sadf['Categories'] = sadf.apply(lambda x: predict_level1_category(x['Entity'], mm_general) if (x['Categories'] == 'NA') & (x['Entity'] != 'NA') else x['Categories'], axis = 1)

		#getting categories for reviews for whihc entities were not detected
		sadf['Categories'] = sadf.apply(lambda x: pos_based_catagorization(glovedf, x['Reviews'], mm_general, dev_df = dev_df) if x['Entity'] == 'NA' else x['Categories'], axis = 1)

		#getting categories for which only one entity was detected, i.e, salience = 1
		sadf['Categories'] = sadf.apply(lambda x: pos_based_catagorization(glovedf, x['Reviews'], mm_general, dev_df = dev_df) if (str(x['Salience']) == '1') & (x['Categories'] in ['NA', 'IR']) else x['Categories'], axis = 1)
		
		# Getting level 2 Categories
		#sadf =  mmap_subcat(sadf)

		#combining categories, eg: Features1 and Features2, Usability1 and Usability2
		sadf['Categories'] = sadf['Categories'].apply(lambda x: str(x)[:-1] if x not in ('NA','IR') else x)
		#return sadf

		return sadf


	def run(self):
		sadf = self.get_review_buckets()
		sadf = sadf.astype(str)
		sadf = sadf.fillna("")
		credentials = service_account.Credentials.from_service_account_file(out_key_file,)
		try:
			pandas_gbq.to_gbq(sadf, destination_table=destination_table, project_id=out_project_id, credentials=credentials, if_exists='replace')
			return df
		except Exception as e:
			send_email(from_email=task_params.loc[task_id,'From Email'],                   #send an email if there is an error in uploading
                        from_email_pass=task_params.loc[task_id,'From Email Password'],
                        to_email=task_params.loc[task_id,'To Email'],
                        subject='Error in Uploading file(s)', 
                        body_text=f'There was an error in uploading the file to {destination_table} : {e}')
			print('--Error--\n')
			print(e)




if __name__ == '__main__':
	task_id_list = task_params.index.to_list()
	model_path = 'C:/Users/ADMIN/Desktop/sentiment analysis module/glove.6B.100d.txt'
	glovedf = glove_model(model_path)		# importing model
	
	for task_id in task_id_list:

		mapping_sheet_id = task_params.loc[task_id, 'Mapping Sheet']

		sa_key_file =  task_params.loc[task_id, 'Sentiment Analysis Key File']	
		sa_project_id = task_params.loc[task_id, 'Sentiment Analysis Project ID']		
		sa_dataset = task_params.loc[task_id, 'Sentiment Analysis Dataset']			
		sa_table = task_params.loc[task_id, 'Sentiment Analysis Table']	

		out_key_file =  task_params.loc[task_id, 'Sentiment Analysis Key File']	
		out_project_id = task_params.loc[task_id, 'Sentiment Analysis Project ID']		
		out_dataset = task_params.loc[task_id, 'Sentiment Analysis Dataset']			
		out_table_suffix = task_params.loc[task_id, 'Sentiment Analysis Table Suffix']	

		df = load_data(key_path=sa_key_file, project_id=sa_project_id, dataset=sa_dataset, t=sa_table)       #importing sentiment analysis table
		print(df.shape)
		df = df.sort_values(by = 'Review_ID')
		df = flatten_metrics(df)
		#df = df.head(20)

		#rr_tables = task_params.loc[task_id, 'Raw Reviews Tables'].split(',')
		#sa_tables = [t + '_' + sa_table_suffix for t in rr_tables]

		distinct_product_types = df.Product_Category.unique()
		for t in distinct_product_types:
			print(t)

			table = t + '_' + out_table_suffix
			destination_table = f'{out_dataset}.{table}'
			
			dft = df[df.Product_Category == t]
			obj = Sentiment_Buckets(dft)
			print(obj.data.head())
			print(" ")
			obj.run()
			print(task_params.loc[task_id, 'Client'], f'{t} Bucketing Done')

		print('All',task_params.loc[task_id, 'Client'],'Done')
	print('--Done--')