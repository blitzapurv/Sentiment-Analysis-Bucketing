import time
from fileinput import filename
import pandas as pd
import numpy as np
from utils import analyze_text_entities, analyze_text_sentiment, send_email, load_data, get_taskparams
from google.oauth2 import service_account
from google.cloud import language, bigquery
import pandas_gbq
from tqdm import tqdm
import configparser



config = configparser.ConfigParser()
config.read('variables.ini')
task_params = get_taskparams(config['Default']['sheet_id'], config['Default']['sheet_name'])



class Sentiment_Analysis():
	def __init__(self, data):
		self.review_id = data['Review_ID']
		self.review = data['Comment']
		self.date = data['Date']
		self.platform = data['Platform']
		self.view = data['View']
		self.product_category = data['Product_Category']
	def get_entities(self):
		review_ids = []		#Review IDs
		reviews = []		#Reviews
		entities = []		#Entity and Entity metrics
		for i in range(len(self.review)):
			try:
				te = analyze_text_entities(text=self.review[i], client=NL_client)		#getting entities and entity metrics
				review_ids.extend([self.review_id[i]])
				reviews.extend([self.review[i]])
				entities.extend([te])
			except Exception as e:
				print(i, e)
				review_ids.extend([self.review_id[i]])
				reviews.extend([self.review[i]])
				entities.extend(['NA'])							#append 'NA' if there is an error
			#time.sleep(0.1) #atleast 0.1s to remain below 600 requests/sec quota
		sadf = pd.DataFrame({'Review_ID':review_ids, 'Reviews':reviews, 'Entities':entities})
		print(sadf.shape)
		return sadf

	def get_overall_sentiment(self):
		review_ids = []
		views = []
		product_categories = []
		overall_sentiment_scores = []
		overall_sentiment_magnitudes = []
		for i in range(len(self.review)):
			try:
				ts = analyze_text_sentiment(text=self.review[i], client=NL_client)      #getting overall sentiment score and magnitude
				review_ids.extend([self.review_id[i]])
				views.extend([self.view[i]])
				product_categories.extend([self.product_category[i]])
				overall_sentiment_scores.extend([ts['sentiment_score']])
				overall_sentiment_magnitudes.extend([ts['sentiment_magnitude']])
			except Exception as e:
				print(i, e)
				review_ids.extend([self.review_id[i]])             
				views.extend([self.view[i]])
				product_categories.extend([self.product_category[i]])
				overall_sentiment_scores.extend(['NA'])			#append 'NA' if there is an error
				overall_sentiment_magnitudes.extend(['NA'])		#append 'NA' if there is an error
			#time.sleep(0.1) #atleast 0.1s to remain below 600 requests/sec quota
		oss = pd.DataFrame({'Review_ID':review_ids, 'Overall_Sentiment_Score': overall_sentiment_scores, 'Overall_Sentiment_Magnitude': overall_sentiment_magnitudes, 
							'View': views, 'Product_Category': product_categories})
		print(oss.shape)
		return oss

	def run(self):
		sadf = self.get_entities()
		#time.sleep(30)
		oss = self.get_overall_sentiment()
		df = sadf.merge(oss, on='Review_ID', how='inner')  # merging

		credentials = service_account.Credentials.from_service_account_file(sa_key_file,)
		df = df.astype(str)
		df = df.fillna("")
		print(df.shape,'\n', df.head())
		try:
			pandas_gbq.to_gbq(df, destination_table=destination_table, project_id=sa_project_id, credentials=credentials, if_exists='replace')
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

	for task_id in task_id_list:

		rr_key_file =  task_params.loc[task_id, 'Raw Reviews Key File']		# Raw Reviews
		rr_project_id = task_params.loc[task_id, 'Raw Reviews Input'].split('.')[0]		#config['data_upload']['project_id']
		rr_dataset = task_params.loc[task_id, 'Raw Reviews Input'].split('.')[1]			#config['data_upload']['dataset']
		rr_table = task_params.loc[task_id, 'Raw Reviews Input'].split('.')[2]	#config['data_import']['table'].split(',')

		NL_client = language.LanguageServiceClient.from_service_account_json(task_params.loc[task_id, 'NL API Key File'])
		
		sa_key_file =  task_params.loc[task_id, 'Sentiment Analysis Key File']		#config['data_upload']['key_file']
		sa_project_id = task_params.loc[task_id, 'Sentiment Analysis Project ID']		#config['data_upload']['project_id']
		sa_dataset = task_params.loc[task_id, 'Sentiment Analysis Dataset']			#config['data_upload']['dataset']
		sa_table = task_params.loc[task_id, 'Sentiment Analysis Table']	#config['data_import']['table'].split(',')
		
		df = load_data(key_path=rr_key_file, project_id=rr_project_id, dataset=rr_dataset, t=rr_table)       #importing review data from source table
		#df = df.head(150)#testing
		df = df.astype(str)
		df = df.fillna("")
		
		print(df.columns)
		print(df.shape)
		
		#distinct_product_types = df.Product_Category.unique()
		#for t in distinct_product_types:
		#sa_table = t + '_' + sa_table_suffix		#config['data_upload']['table']
		destination_table = f'{sa_dataset}.{sa_table}'
		
		#df1 = df[df.Product_Category == t].copy()
		obj = Sentiment_Analysis(df)
		obj.run()
		#print(task_params.loc[task_id, 'Client'], f'{t} Sentiment Analysis Done')
		print('All',task_params.loc[task_id, 'Client'],'Done')	
	print('All Done')