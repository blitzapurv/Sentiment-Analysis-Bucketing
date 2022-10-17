import os
import zipfile
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud import language, bigquery
import pandas_gbq
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import zipfile
import configparser


config = configparser.ConfigParser()
config.read('variables.ini')

def get_taskparams(sheet_id, sheet_name):
    sheet_id = sheet_id
    sheet_name = sheet_name
    csv_url = "https://docs.google.com/spreadsheets/d/" + sheet_id + f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    task_params = pd.read_csv(csv_url, index_col = 0)
    task_params = task_params.astype(str)
    return task_params


def load_data(key_path, project_id, dataset, t):
    KEY_PATH = key_path#       # json key file
    project_id = project_id    # project_id
    dataset = dataset          # dataset
    table = t                  # table
    CREDS = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials=CREDS, project=CREDS.project_id)
    # SQL Query
    Q1 = f"""
    SELECT * 
    FROM `{project_id}.{dataset}.{table}`
    """
    query_job1 = client.query(Q1)
    df = query_job1.to_dataframe()
    return df



#KEY_PATH1 = config['NL_api']['key_file']
#client = language.LanguageServiceClient.from_service_account_json(KEY_PATH1)

# Entity Sentiment
def analyze_text_entities(text, client):
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
    #document = {"type_": "PLAIN_TEXT", "language": "EN", "content": text}
    encoding_type = 'UTF8'
    response = client.analyze_entity_sentiment(document=document, encoding_type = encoding_type)
    entities = {}
    for entity in response.entities:        #creating a dictionary of metrics
        entity_metrics = dict(salience = entity.salience,
                sentiment_score = entity.sentiment.score,
                sentiment_magnitude = entity.sentiment.magnitude
                )
        entities[entity.name] = entity_metrics      #creating parent dictionary of entitities, {entity: entity_metrics}
    return entities


# Overall Sentiment
def analyze_text_sentiment(text, client):
    document = language.Document(content=text, type_=language.Document.Type.PLAIN_TEXT)
    #document = { "type_": "PLAIN_TEXT", "language": "EN", "content": text}
    response = client.analyze_sentiment(document=document)
    sentiment = response.document_sentiment
    score = {}      #dictionary to store overall sentiment score and magnitude
    score['sentiment_score'] = sentiment.score              #overall sentiment score
    score['sentiment_magnitude'] = sentiment.magnitude      #overall sentiment magnitude
    return score


# Funtion for Sending email
import smtplib
def send_email(from_email, from_email_pass, to_email, subject, body_text):
    #Ports 465 and 587 are intended for email client to email server communication - sending email
    server = smtplib.SMTP('smtp.gmail.com', 587)
    #starttls() is a way to take an existing insecure connection and upgrade it to a secure connection using SSL/TLS.
    server.starttls()
    #Next, log in to the server
    server.login(from_email, from_email_pass)#("prreporting@pivotroots.com", "vwcuvekcfubywxcg")
    msg = MIMEMultipart()
    msg["Subject"] = subject
    body = MIMEText(body_text)
    msg["From"] = from_email#"prreporting@pivotroots.com"
    msg["To"] = to_email#"ekagra.singh@pivotroots.com"
    msg.attach(body)
    server.sendmail(msg["From"], msg["To"],msg.as_string())
    server.quit()

#Send the mail
#
