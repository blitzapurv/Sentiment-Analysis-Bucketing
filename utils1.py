import os
import ast
import zipfile
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import language, bigquery
import pandas_gbq
import numpy as np
import pandas as pd
from scipy import spatial
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import zipfile
import configparser
from tqdm import tqdm
import stanza


config = configparser.ConfigParser()
config.read('variables.ini')



def open_entities(data):
    data.Entities = data.Entities.astype(str)
    data.Entities = data.Entities.replace('NA', '{}')
    data.Entities = data.Entities.apply(lambda x: ast.literal_eval(x))
    data['Entity'] = data['Entities'].apply(lambda x: x.keys())
    d1 = data.explode('Entity')
    d2 = data['Entities'].apply(lambda x: [x.get(k) for k in x.keys()]).explode().values
    d1['Entity_Metrics'] = d2
    #d1.Entity = d1.Entity.fillna('{}')
    #d1.Entity_Metrics = d1.Entity_Metrics.fillna('{}')
    #d1 = d1.dropna()
    d1 = d1.drop('Entities', axis = 1)
    return d1


def flatten_metrics(df):
    data = df.copy()
    df_entity = open_entities(data)
    df_entity.Entity = df_entity.Entity.fillna('NA')
    df_entity['Salience'] = df_entity['Entity_Metrics'].apply(lambda x: x['salience'] if isinstance(x, dict) else 'NA')
    df_entity['Sentiment_Score'] = df_entity['Entity_Metrics'].apply(lambda x: x['sentiment_score'] if isinstance(x, dict) else 'NA')
    df_entity['Sentiment_Magnitude'] = df_entity['Entity_Metrics'].apply(lambda x: x['sentiment_magnitude'] if isinstance(x, dict) else 'NA')
    df_entity = df_entity.drop('Entity_Metrics', axis = 1)
    return df_entity


#df = load_data('data_import_sentiment', 'MixerGrinder_Sentiment_Table')


# Buckting function
## get mapping dictionary from gsheet
def get_mapping_dict(sheet_id, sheet_name):
    sheet_id = sheet_id     #config['mapping_sheet']['sheet_id']
    sheet_name = sheet_name
    csv_url = "https://docs.google.com/spreadsheets/d/" + sheet_id + f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    mapping_dict = pd.read_csv(csv_url, header = None, index_col = 0)[1].to_dict()
    return mapping_dict


# Dictionary for Level 1
#mm = get_mapping_dict('level1')

# Dictionary for Level 1 general key value pairs
#mm_general = get_mapping_dict('level1_general')


def mmap_subcat(data):
    df = data.copy()
    df['Sub-Categories'] = 'NA'

    #Functionality
    def funtionality_map(data):
        if 'Functionality' in str(data['Categories']):
            if data['Sentiment_Score'] == 'NA':         #condition if entity sentiment score is NA, look at overall sentiment score
                if data['Overall_Sentiment_Score'] == 'NA':
                    return data['Sub-Categories']
                if float(data['Overall_Sentiment_Score']) > 0.1:
                    return 'Effective'
                elif float(data['Overall_Sentiment_Score']) < -0.1:
                    return 'Ineffective'
                else:
                    return 'Neutral'
            elif float(data['Sentiment_Score']) > 0.1:
                return 'Effective'
            elif float(data['Sentiment_Score']) < -0.1:
                return 'Ineffective'
            else:
                return 'Neutral'
        else:
            return data['Sub-Categories']
    
    #Features
    def features_map(data):
        if data['Categories'] == 'Features1':
            return 'Sound/Noise'
        elif data['Categories'] == 'Features2':
            return 'Electricity consumption'
        elif data['Categories'] == 'Features3':
            return 'Components'
        else:
            return data['Sub-Categories']
    
    #Product Perception
    def product_perception_map(data):
        if 'Product Perception' in str(data['Categories']):
            if ('thank' in str(data['Entity']).lower()) | ('tq' in str(data['Entity']).lower()) | ('tnx' in str(data['Entity']).lower()) | ('thx' in str(data['Entity']).lower()):
                return 'Appreciation'
            elif ('fake' in str(data['Reviews']).lower()) | ('duplicate' in str(data['Reviews']).lower()):
                return 'Claim of counterfeit product'
            elif data['Sentiment_Score'] == 'NA':       #condition if entity sentiment score is NA, look at overall sentiment score
                if data['Overall_Sentiment_Score'] == 'NA':
                    return data['Sub-Categories']
                elif float(data['Overall_Sentiment_Score']) > 0.1:
                    return 'Positive Feedback'
                elif float(data['Overall_Sentiment_Score']) < -0.1:
                    return 'Negative Feedback'
                else:
                    return 'Neutral'
            elif float(data['Sentiment_Score']) > 0.1:
                return 'Positive Feedback'
            elif float(data['Sentiment_Score']) < -0.1:
                return 'Negative Feedback'
            else:
                return 'Neutral'
        else: 
            return data['Sub-Categories']
    
    #Usability
    def utility_map(data):
        if 'Usability' in str(data['Categories']):
            if data['Sentiment_Score'] == 'NA':         #condition if entity sentiment score is NA, look at overall sentiment score
                if data['Overall_Sentiment_Score'] == 'NA':
                    return data['Sub-Categories']
                if float(data['Overall_Sentiment_Score']) > 0:
                    return 'Easy to Use'
                elif float(data['Overall_Sentiment_Score']) <= 0:
                    return 'Inconvenient'
                else:
                    return 'Neutral'
            elif float(data['Sentiment_Score']) > 0:
                return 'Easy to Use'
            elif float(data['Sentiment_Score']) < 0:
                return 'Inconvenient'
            else:
                return 'Neutral'
        else: 
            return data['Sub-Categories']

    #Pricing
    def pricing_map(data):
        if 'Pricing' in str(data['Categories']):
            if data['Sentiment_Score'] == 'NA':         #condition if entity sentiment score is NA, look at overall sentiment score
                if data['Overall_Sentiment_Score'] == 'NA':
                    return data['Sub-Categories']
                if float(data['Overall_Sentiment_Score']) > 0:
                    return 'Value for Money'
                elif float(data['Overall_Sentiment_Score']) <= 0:
                    return 'Expensive'
                else:
                    return 'Neutral'
            elif float(data['Sentiment_Score']) > 0:
                return 'Value for Money'
            elif float(data['Sentiment_Score']) < 0:
                return 'Expensive'
            else:
                return 'NA'
        else: 
            return data['Sub-Categories']

    #Shipping Experience
    def shipping_experience_map(data):
        if data['Categories'] == 'Shipping Experience3':
            return 'Damaged/Unsealed'
        elif data['Categories'] == 'Shipping Experience2':
            if data['Sentiment_Score'] == 'NA':
                if data['Overall_Sentiment_Score'] == 'NA':
                    return data['Sub-Categories']
                if float(data['Overall_Sentiment_Score']) <= 0:
                    return 'Bad/Unsatisfactory Package/Delivery'
                else:
                    return 'Good/Satisfactory Package/Delivery'
            elif float(data['Sentiment_Score']) <= 0:
                return 'Bad/Unsatisfactory Package/Delivery'
            else:
                return 'Good/Satisfactory Package/Delivery'
        elif data['Categories'] == 'Shipping Experience1':
            return data['Sub-Categories']
        else:
            return data['Sub-Categories']

    #Manufacturing & Design
    def manufacturing_design_map(data):
        if data['Categories'] == 'Manufacturing & Design1':
            return 'Quality'
        elif data['Categories'] == 'Manufacturing & Design2':
            return 'Size/Design'
        else:
            return data['Sub-Categories']
    


    df['Sub-Categories'] =  df.apply(funtionality_map, axis = 1)
    df['Sub-Categories'] =  df.apply(features_map, axis = 1)
    df['Sub-Categories'] =  df.apply(product_perception_map, axis = 1)
    df['Sub-Categories'] =  df.apply(utility_map, axis = 1)
    df['Sub-Categories'] =  df.apply(pricing_map, axis = 1)
    df['Sub-Categories'] =  df.apply(shipping_experience_map, axis = 1)
    df['Sub-Categories'] =  df.apply(manufacturing_design_map, axis = 1)
    
    return df




#=================================================================

# Importing pretrained word embeddings (vector representation), Glove

def glove_model(path = 'glove.6B.100d.txt'):
    glove100d = path # 100d vector representation

    # laoding word vectors
    embeddings_dict = {} # We create a dictionary of word -> embedding
    f = open(glove100d, encoding="utf8") # Open file

    # In the dataset, each line represents a new word embedding
    # The line starts with the word and the embedding values follow
    words = []
    vectors = []
    for line in f:#tqdm(f):
        values = line.split()
        word = values[0] # The first value is the word, the rest are the values of the embedding
        words.append(word)
        embedding = np.asarray(values[1:], dtype='float32') # Loading embedding
        vectors.append(embedding)
        embeddings_dict[word] = embedding # Adding embedding to our embedding dictionary
    f.close()
    print('\nFound %s word vectors.' % len(embeddings_dict))
    glove_df = pd.DataFrame(embeddings_dict).T
    return glove_df


#Function to output within category deviations for distances

def within_category_deviation(glovedf, level1_dict):
    mm = level1_dict
    category_list = list(set([mm[k] for k in mm.keys()]))
    dev_df = pd.DataFrame(columns = ['MAD', 'SD', 'max_dist', 'min_dist'])
    for category in category_list:
        xx = glovedf.loc[[k for k in mm.keys() if mm[k] == category],:].copy()
        mean_vec = xx.mean()
        xx1 = xx - mean_vec
        dev_df.loc[category,'MAD'] = xx1.pow(2).T.sum().apply(np.sqrt).mean()
        dev_df.loc[category,'SD'] = np.sqrt(xx1.pow(2).T.sum().mean())
        dev_df.loc[category,'max_dist'] = xx1.pow(2).T.sum().apply(np.sqrt).max()
        dev_df.loc[category,'min_dist'] = xx1.pow(2).T.sum().apply(np.sqrt).min()
        #ldf.loc[category,'median dist'] = xx1.pow(2).T.sum().apply(np.sqrt).median()
    return dev_df


# Function to estimate bucket/category given an entity
def level1_estimate(glovedf, word, level1_dict, dev_df, threshold_metric = 'MAD'):
    mm = level1_dict
    category_list = list(set([mm[k] for k in mm.keys()]))
    ldf = pd.DataFrame(columns = ['Centroid'])
    for category in category_list:
        ldf.loc[category,'Centroid'] = glovedf.loc[[k for k in mm.keys() if mm[k] == category],:].mean().values
    
    x_vec = glovedf.loc[word,:].values
    ldf['dist'] = ldf['Centroid'].apply(lambda x: spatial.distance.euclidean(x_vec,x))

    min_index = ldf['dist'].idxmin()
    if ldf['dist'].min() > dev_df.loc[min_index, threshold_metric]: #k*mad
        estimated_category = 'IR'
    else:
        estimated_category = ldf['dist'].idxmin()

    return ldf, estimated_category



# Using POS ('VERB','NOUN','ADJ') for catagorising reviews for which entities were not detected
def pos_based_catagorization(glovedf, text, level1_dict, dev_df, threshold_metric = 'max_dist'):
    mm = level1_dict
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos', verbose = False)
    doc = nlp(text)

    dft = pd.DataFrame(columns = ['word','lemma','pos','upos','xpos'])
    for i, sent in enumerate(doc.sentences):    
        for word in sent.words:
            dft = dft.append({'word':word.text, 'lemma':word.lemma, 'pos':word.pos, 'upos':word.upos, 'xpos': word.xpos}, ignore_index=True)
    key_words = dft[dft.pos.isin(['NOUN','VERB'])].word.tolist()
    key_words = [k.lower() for k in key_words if k.lower() in list(glovedf.index)]

    #if there are only one or 0 nouns or verbs, then include adjectives as well
    if len(key_words) <= 0:
        key_words = dft[dft.pos.isin(['NOUN','VERB','ADJ'])].word.tolist()
        key_words = [k.lower() for k in key_words if k.lower() in list(glovedf.index)]
    else:
        pass
    #print(key_words)
    if len(key_words) > 0:
        out_vec = 0
        for k in key_words:
            out_vec = out_vec + glovedf.loc[k,:].values
        out_vec = out_vec/len(key_words)

        category_list = list(set([mm[k] for k in mm.keys()]))
        ldf = pd.DataFrame(columns = ['Centroid'])
        for category in category_list:
            ldf.loc[category,'Centroid'] = glovedf.loc[[k for k in mm.keys() if mm[k] == category],:].mean().values
        
        ldf['dist'] = ldf['Centroid'].apply(lambda x: spatial.distance.euclidean(out_vec,x))

        min_index = ldf['dist'].idxmin()
        if ldf['dist'].min() > dev_df.loc[min_index, threshold_metric]: #k*mad
            estimated_category = 'IR'
        else:
            estimated_category = min_index
        
        return estimated_category
    else:
        return 'NA'